"""Compare reduced-precision timestamps with FP32 and the fixed Soniox pairs."""
import json
from pathlib import Path
import statistics

from align_existing import atomic_json, sha
from benchmark_soniox import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE/'benchmarks/memory-v1'
VARIANTS = ('float16', 'int8', 'nf4')


def flatten(run):
    return [w for s in run['segments'] for w in s['words']]


def get_samples():
    base = HERE/'benchmarks/soniox-v1'
    samples = []
    for item in json.loads((base/'manifest.json').read_text()):
        sid = item['id']
        baseline = json.loads((base/f'{sid}.canary.json').read_text())
        pairs = [(r['index'], r['reference']) for r in json.loads((base/f'{sid}.comparison.json').read_text())['words']]
        samples.append((sid, baseline, pairs))
    b = HERE/'benchmarks/bilingual-v1'
    ref = json.loads((b/'reference.json').read_text())
    samples.append(('bilingual', json.loads((b/'canary.json').read_text()),
                    [(i,w) for i,w in enumerate(ref) if not w['issues']]))
    return samples


def main():
    summary = {'reference': 'Same fixed Soniox pairs as the previous benchmarks; no quality-based exclusion',
               'quantization': 'bitsandbytes 0.46.0 Linear8bitLt or Linear4bit NF4 on 217 encoder Linear layers; other encoder/decoder layers FP16; frontend and NFA FP32. These are not GGUF Q8_0/Q4_K.',
               'samples': [], 'aggregate': {}}
    accum = {name: {'errors': [], 'deltas': [], 'runtime': []} for name in ('float32',)+VARIANTS}
    for sid, baseline, pairs in get_samples():
        bw = flatten(baseline)
        source_words = [w['word'] for w in bw]
        metrics = {}
        runs = {'float32': baseline}
        for variant in VARIANTS:
            p = ROOT/f'{sid}.{variant}.json'
            if p.exists():
                runs[variant] = json.loads(p.read_text())
        for variant, run in runs.items():
            assert run['audio_sha256'] == baseline['audio_sha256']
            assert run['transcript_sha256'] == baseline['transcript_sha256']
            words = flatten(run)
            assert [w['word'] for w in words] == source_words
            assert [{k:s[k] for k in original if k!='words'} for s,original in zip(run['segments'],baseline['segments'])] == [{k:v for k,v in s.items() if k!='words'} for s in baseline['segments']]
            errors, deltas = [], []
            for i, reference in pairs:
                assert words[i]['start'] is not None and words[i]['end'] is not None, 'Reduced precision lost a previously scored word'
                for key in ('start', 'end'):
                    errors.append(1000*(words[i][key]-reference[key]))
                    deltas.append(1000*(words[i][key]-bw[i][key]))
            metrics[variant] = {'reference_agreement': stats(errors), 'fp32_deviation': stats(deltas),
                'identical_to_fp32_pct': 100*sum(abs(x)<1e-7 for x in deltas)/len(deltas),
                'runtime_and_memory': run['summary']}
            accum[variant]['errors'].extend(errors)
            accum[variant]['deltas'].extend(deltas)
            accum[variant]['runtime'].append(run['summary'])
        summary['samples'].append({'sample':sid, 'paired_words':len(pairs), 'models':metrics})
    for variant, value in accum.items():
        if len(value['runtime']) != len(summary['samples']):
            continue
        rs = value['runtime']
        summary['aggregate'][variant] = {
            'paired_words':len(value['errors'])//2, 'reference_agreement':stats(value['errors']),
            'fp32_deviation':stats(value['deltas']),
            'identical_to_fp32_pct':100*sum(abs(x)<1e-7 for x in value['deltas'])/len(value['deltas']),
            'segmented_peak_allocated_gpu_gib':max(r['peak_gpu_gib'] for r in rs[:-1]),
            'continuous_6min_peak_allocated_gpu_gib':rs[-1]['peak_gpu_gib'],
            'segmented_peak_process_gpu_gib':max((r.get('peak_process_gpu_gib_nvml',0) for r in rs[:-1]), default=0) or None,
            'continuous_6min_peak_process_gpu_gib':rs[-1].get('peak_process_gpu_gib_nvml'),
            'resident_allocated_gpu_gib':rs[0].get('post_load_allocated_gpu_gib'),
            'resident_process_gpu_gib':rs[0].get('post_load_process_gpu_gib'),
            'warm_total_seconds':sum(r['alignment_seconds'] for r in rs),
            'first_full_pass_seconds':sum(r.get('repeat_seconds',[r['alignment_seconds']])[0] for r in rs),
        }
    atomic_json(ROOT/'summary.json',summary)
    print(json.dumps(summary['aggregate'],indent=2))


if __name__=='__main__':
    main()
