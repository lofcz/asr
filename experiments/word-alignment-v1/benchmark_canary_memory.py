"""Compare numeric precision without changing tokenizer, crops or NFA decoding."""
import argparse
import json
import os
from pathlib import Path
import statistics
import threading
import time

from align_canary import load_model, align_segments
from align_existing import atomic_json, decode, sha

HERE = Path(__file__).resolve().parent


class ProcessMemory:
    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.api = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]))
        self.peak = 0
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def current(self):
        return max((p.usedGpuMemory for p in self.api.nvmlDeviceGetComputeRunningProcesses(self.handle)
                    if p.pid == os.getpid()), default=0)

    def poll(self):
        while not self.halt.is_set():
            self.peak = max(self.peak, self.current())
            self.halt.wait(.02)

    def start(self):
        self.thread.start()

    def stop(self):
        self.halt.set()
        self.thread.join()
        self.peak = max(self.peak, self.current())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['float32', 'float16', 'bfloat16', 'int8', 'nf4'], required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    variant = args.variant
    precision = variant if variant in ('float32', 'float16', 'bfloat16') else 'float16'
    quantization = variant if variant in ('int8', 'nf4') else 'none'
    started = time.perf_counter()
    model = load_model('cuda', precision, quantization)
    loading = time.perf_counter()-started
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    post_load_allocated = torch.cuda.memory_allocated()/2**30
    post_load_process = ProcessMemory().current()/2**30
    tasks = []
    base = HERE/'benchmarks/soniox-v1'
    for sample in json.loads((base/'manifest.json').read_text()):
        tasks.append({'id': sample['id'], 'audio': Path(sample['remote_audio']),
                      'transcript': base/(sample['id']+'.transcript.json'),
                      'baseline': base/(sample['id']+'.canary.json')})
    b = HERE/'benchmarks/bilingual-v1'
    tasks.append({'id': 'bilingual', 'audio': b/'audio.wav', 'transcript': b/'transcript.json', 'baseline': b/'canary.json'})
    for task in tasks:
        audio = decode(task['audio'])
        segments = json.loads(task['transcript'].read_text())
        baseline = json.loads(task['baseline'].read_text())
        assert sha(task['audio']) == baseline['audio_sha256']
        assert sha(task['transcript']) == baseline['transcript_sha256']
        memory = ProcessMemory()
        torch.cuda.empty_cache()
        memory.start()
        align_segments(segments[:1], audio, model, 'cuda')
        torch.cuda.reset_peak_memory_stats()
        runs = [align_segments(segments, audio, model, 'cuda') for _ in range(3)]
        memory.stop()
        assert all(r['segments'] == runs[0]['segments'] for r in runs), 'Nondeterministic word timings'
        result = runs[0]
        times = [r['summary']['alignment_seconds'] for r in runs]
        result['summary'].update({'alignment_seconds': statistics.median(times), 'repeat_seconds': times,
            'realtime_factor': statistics.median(times)/(len(audio)/16000), 'warmed_up': True,
            'post_load_allocated_gpu_gib': post_load_allocated, 'post_load_process_gpu_gib': post_load_process,
            'peak_process_gpu_gib_nvml': memory.peak/2**30,
            'peak_reserved_gpu_gib': torch.cuda.max_memory_reserved()/2**30})
        result.update({'variant': variant, 'precision': precision, 'quantization': quantization,
                       'model': json.loads((HERE/'canary-aux/provenance.json').read_text()),
                       'sample': task['id'], 'model_load_seconds': loading,
                       'audio_sha256': sha(task['audio']), 'transcript_sha256': sha(task['transcript']),
                       'method': 'Official NeMo encoder/tokenizer/NFA; FP32 mel frontend and Viterbi; optional reduced encoder/decoder precision',
                       'logical_parameters': model._alignment_logical_parameters,
                       'language_routing': False})
        atomic_json(HERE/f"benchmarks/memory-v1/{task['id']}.{variant}.json", result)
        print(json.dumps({'sample': task['id'], 'variant': variant, **result['summary']}), flush=True)


if __name__ == '__main__':
    main()
