"""Repeat the existing WhisperX path after warmup for comparable timings."""
import argparse
import json
from pathlib import Path
import statistics
import time
from align_existing import align_segments, atomic_json, decode, load_aligner, sha, WHISPERX_COMMIT

p = argparse.ArgumentParser()
p.add_argument('--language', required=True)
p.add_argument('--audio', type=Path, required=True)
p.add_argument('--transcript', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
import torch
torch.set_num_threads(2)
torch.manual_seed(0)
t = time.perf_counter()
model, metadata, pin = load_aligner(a.language, 'cuda')
loading = time.perf_counter()-t
audio = decode(a.audio)
segments = json.loads(a.transcript.read_text())
align_segments(segments[:1], audio, model, metadata, 'cuda')
torch.cuda.reset_peak_memory_stats()
results = [align_segments(segments, audio, model, metadata, 'cuda') for _ in range(3)]
assert all(r['segments'] == results[0]['segments'] for r in results)
r = results[0]
times = [x['summary']['alignment_seconds'] for x in results]
r['summary'].update({'repeat_seconds': times, 'alignment_seconds': statistics.median(times),
                     'realtime_factor': statistics.median(times)/(len(audio)/16000), 'warmed_up': True})
r.update({'language': a.language, 'model': pin, 'whisperx_commit': WHISPERX_COMMIT,
          'transcript_sha256': sha(a.transcript), 'audio_sha256': sha(a.audio),
          'model_load_seconds': loading, 'precision': 'float32', 'method': 'WhisperX forced alignment only'})
atomic_json(a.out, r)
print(json.dumps(r['summary']), flush=True)
