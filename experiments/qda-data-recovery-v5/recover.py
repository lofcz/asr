"""Recover stored, paired Soniox data without quality-heuristic admission rules.

Never writes to the paid sources, v4 selection, or active training plan. Verifies
the original receipts, bytes and exact token stream; emits a new JSONL catalog.
"""
import argparse
import bisect
import collections
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ASR = HERE.parents[1]
V4 = HERE.parent / 'qda-full-r64-v4'
OLD = ASR / 'datasets/releases/qda-full-r64-v4'
OUT = ASR / 'datasets/releases/qda-recovered-v5'
sys.path.insert(0, str(V4))
from contracts import atomic_json, sha, target_with_spans
from targets import independent_verify


def safe_cuts(segments, duration):
    cuts = [0.0]
    extent = 0.0
    for a, b in zip(segments, segments[1:]):
        extent = max(extent, a['end'])
        if b['start'] >= extent:
            cuts.append((extent + b['start']) / 2)
    return sorted(set(cuts + [duration]))


def initialize():
    global tokenizer, processor, policy
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['USE_TORCH'] = '0'
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(V4 / 'selector-tokenizer')
    processor = json.loads((V4 / 'selector-processor.json').read_text())
    policy = json.loads((HERE / 'policy.json').read_text())


def window_segments(segments, start, end, source_duration):
    """Preserve zero-duration Soniox spans at a cut in exactly one window."""
    chosen = []
    for segment in segments:
        a, b = segment['start'], segment['end']
        intersects = a < end and b > start
        if a == b:
            intersects = start <= a < end or (end == source_duration and a == end)
        if intersects:
            if a < start or b > end:
                raise ValueError('Window boundary cuts speech')
            chosen.append(segment)
    if not chosen:
        raise ValueError('Empty target crop')
    speakers = {}
    return [{'Start': round(s['start'] - start, 2), 'End': round(s['end'] - start, 2),
             'Speaker': speakers.setdefault(s['speaker'], len(speakers)), 'Content': s['text']}
            for s in chosen]


def select_windows(source, label):
    """Every source second is covered once; confidence/speaker count never reject."""
    segments = label['segments']
    duration = source['audio_duration']
    cuts = safe_cuts(segments, duration)
    counts = collections.Counter()
    rows = []
    cursor = 0.0
    while cursor < duration - 1e-8:
        def preference(nominal):
            c = counts.copy()
            c[{600: 'short', 1800: 'medium', 3600: 'long'}[nominal]] += nominal
            total = sum(c.values())
            return sum((c[k] / total - v) ** 2 / v for k, v in policy['length_audio_fractions'].items())
        options = sorted(policy['nominal_window_seconds'], key=preference)
        if cursor == 0:
            first = (600, 1800, 1800, 3600)[int(source['record_id'][:8], 16) % 4]
            options = [first] + [n for n in options if n != first]
        # Back off at safe boundaries for the model's context limit, never drop
        # a window because it has music, one speaker, or uncertain ASR tokens.
        tried = set()
        selected = None
        for nominal in options + [300, 120, 30]:
            stop = min(duration, cursor + nominal)
            end = cuts[bisect.bisect_right(cuts, stop) - 1]
            if end <= cursor or end in tried:
                continue
            tried.add(end)
            target = window_segments(segments, cursor, end, duration)
            sample = {'segments': [dict(start=s['Start'], end=s['End'], speaker=s['Speaker'], text=s['Content']) for s in target]}
            body, _ = target_with_spans(sample)
            ntokens = len(tokenizer.apply_chat_template([{'role': 'assistant', 'content': body}], tokenize=True, add_generation_prompt=False))
            bound = ntokens + math.ceil((end - cursor) * processor['sample_rate'] / processor['compression_ratio']) + 128
            if bound > policy['maximum_sequence_tokens']:
                continue
            wid = hashlib.sha256(f"{source['record_id']}|{cursor:.6f}|{end:.6f}|{source['label_sha256']}".encode()).hexdigest()[:24]
            kind = 'short' if end - cursor <= 660 else 'medium' if end - cursor <= 1860 else 'long'
            selected = {**source, 'window_id': wid, 'source_start': cursor, 'source_end': end,
                        'duration': end - cursor, 'duration_class': kind, 'target': target,
                        'target_tokens': ntokens, 'sequence_token_upper_bound': bound,
                        'speaker_count': len({s['Speaker'] for s in target}),
                        'quality': {'reasons': [], 'heuristics_are_diagnostics': True}}
            break
        if selected is None:
            raise ValueError(f'Unrepresentable safe crop at {cursor}; requires explicit handling, not silent exclusion')
        rows.append(selected)
        counts[selected['duration_class']] += selected['duration']
        cursor = selected['source_end']
    if abs(sum(r['duration'] for r in rows) - duration) > 1e-6:
        raise ValueError('Coverage loss')
    # Text is identical before/after cropping, including low-confidence words.
    if ''.join(t['Content'] for r in rows for t in r['target']) != ''.join(s['text'] for s in segments):
        raise ValueError('Crop token stream changed: ' + source['record_id'])
    return rows


def work(job):
    source, raw_path, audio_path = job
    rid = source['record_id']
    raw_path, audio_path = Path(raw_path), Path(audio_path)
    rawhash = sha(raw_path)
    if rawhash != source['raw_transcript_sha256']:
        raise ValueError('Raw identity mismatch: ' + rid)
    receipt_path = raw_path.parent / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    raw = json.loads(raw_path.read_text())
    if not (receipt['record_id'] == rid and receipt['transcription_id'] == raw['id']
            and receipt['audio_sha256'] == source.get('original_audio_sha256', source['audio_sha256'])
            and receipt['artifacts_sha256']['soniox.raw.json'] == rawhash):
        raise ValueError('Paid pairing mismatch: ' + rid)
    audiohash = sha(audio_path)
    if audiohash != source['audio_sha256']:
        raise ValueError('Audio identity mismatch: ' + rid)
    inventory = {'record_id': rid, 'audio_sha256': audiohash, 'raw_sha256': rawhash,
                 'receipt_sha256': sha(receipt_path), 'transcription_id': raw['id'],
                 'local_audio': str(audio_path), 'local_raw': str(raw_path)}
    if source.get('unresolved_source'):
        return {'source': source, 'inventory': inventory, 'rows': [], 'unresolved': True}
    label_path = OLD / source['label_path']
    if sha(label_path) != source['label_sha256']:
        raise ValueError('Derived label identity mismatch: ' + rid)
    label = json.loads(label_path.read_text())
    independent_verify(raw['tokens'], label['segments'], source['audio_duration'])
    acoustic_path = OLD / 'acoustic' / (rid + '.json')
    vad_path = ASR / 'datasets/releases' / source['vad_path'].split('/data/work/asr-datasets/')[-1]
    decode_evidence = None
    for path in (acoustic_path, vad_path):
        if not path.exists():
            continue
        evidence = json.loads(path.read_text())
        if evidence.get('audio_sha256') == audiohash and (evidence.get('status') == 'passed_decode' or evidence.get('full_decode_verified')):
            decode_evidence = {'path': str(path), 'sha256': sha(path)}
            break
    if source['split'] == 'train' and decode_evidence is None:
        raise ValueError('Missing matching full-decode evidence: ' + rid)
    inventory['decode_evidence'] = decode_evidence
    dest = OUT / source['label_path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(label_path, dest)
    if sha(dest) != source['label_sha256']:
        raise ValueError('Copy mismatch')
    rows = select_windows(source, label) if source['split'] == 'train' else []
    diagnostics = {'record_id': rid, 'speaker_count': source['speaker_count'],
                   'low_confidence_fraction': source['low_confidence_fraction'],
                   'target_language_fraction': source['target_language_fraction'],
                   'intraword_gap_regions': label.get('quarantined_regions', []),
                   'acoustic_evidence': str(acoustic_path), 'effect_on_selection': 'none'}
    return {'source': source, 'inventory': inventory, 'rows': rows, 'diagnostics': diagnostics}


def write_lines(path, rows):
    tmp = path.with_suffix(path.suffix + '.building')
    with tmp.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / '.building').write_text('Recovery in progress\n')
    sources = json.loads((OLD / 'sources.json').read_text())
    raw_inventory = {p.parent.name: p for root in ('soniox-500h-v1', 'soniox-50h-v2')
                     for p in (ASR / 'datasets' / root).glob('records/*/soniox.raw.json')}
    original = {}
    for origin in ('qda-496h-v1', 'qda-50h-v2'):
        for line in (ASR / 'datasets/releases' / origin / 'training/sources.jsonl').read_text().splitlines():
            row = json.loads(line)
            original.setdefault(row['record_id'], dict(row, origin=origin))
    unresolved = json.loads((OLD / 'structural-quarantine.json').read_text())
    jobs = []
    all_sources = sources + [{**original[r['record_id']], 'unresolved_source': r} for r in unresolved]
    for source in all_sources:
        paths = [ASR / 'datasets/releases' / source['origin'] / source['audio_path'], ASR / source['audio_path']]
        audio = next((p for p in paths if p.exists()), None)
        if audio is None or source['record_id'] not in raw_inventory:
            raise FileNotFoundError(source['record_id'])
        jobs.append((source, str(raw_inventory[source['record_id']]), str(audio)))
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=initialize,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(work, job): job[0]['record_id'] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            rid = futures[future]
            try:
                results[rid] = future.result()
            except Exception as error:
                atomic_json(HERE / 'progress.json', {'stage': 'error', 'record_id': rid, 'error': str(error), 'updated_at': time.time()})
                raise RuntimeError(f'Recovery failed for {rid}: {error}') from error
            if len(results) % 20 == 0 or len(results) == len(jobs):
                progress = {'checked': len(results), 'total': len(jobs), 'updated_at': time.time(), 'stage': 'verifying_and_recovering'}
                atomic_json(HERE / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    train = [r for source in sources for r in results[source['record_id']]['rows']]
    dev = json.loads((OLD / 'screening-selection.json').read_text())['development']
    heldout = [s for s in sources if s['split'] != 'train']
    if {s['audio_sha256'] for s in train} & {s['audio_sha256'] for s in heldout}:
        raise ValueError('Held-out audio leakage')
    if {s['show'] for s in train} & {s['show'] for s in heldout}:
        raise ValueError('Held-out show leakage')
    if len({r['window_id'] for r in train}) != len(train):
        raise ValueError('Duplicate window ID')
    write_lines(OUT / 'windows.jsonl', train)
    write_lines(OUT / 'pairing-inventory.jsonl', [results[k]['inventory'] for k in sorted(results)])
    write_lines(OUT / 'diagnostics.jsonl', [r['diagnostics'] for r in results.values() if 'diagnostics' in r])
    atomic_json(OUT / 'sources.json', sources)
    atomic_json(OUT / 'development.json', {'source_release': 'qda-full-r64-v4', 'development': dev})
    atomic_json(OUT / 'unresolved-sources.json', unresolved)
    previous = json.loads((OLD / 'selection-summary.json').read_text())
    total = sum(r['duration'] for r in train) / 3600
    report = {'status': 'verified_selection', 'created_at': time.time(), 'train_hours': total,
              'previous_train_hours': previous['train_hours'], 'restored_hours': total - previous['train_hours'],
              'source_release_hours': sum(r['audio_duration'] for r in original.values()) / 3600,
              'original_paid_release_hours': 496.3393171701389,
              'heldout_hours': sum(r['audio_duration'] for r in heldout) / 3600,
              'unresolved_source_hours': sum(r['hours'] for r in unresolved),
              'unresolved_sources': len(unresolved), 'paid_pairings_verified': len(results),
              'recordings': len({r['record_id'] for r in train}), 'windows': len(train),
              'length_hours': {k: sum(r['duration'] for r in train if r['duration_class'] == k) / 3600 for k in ('short', 'medium', 'long')},
              'speaker_hours': {str(k): sum(r['duration'] for r in train if r['speaker_count'] == k) / 3600 for k in sorted({r['speaker_count'] for r in train})},
              'heuristic_exclusions': False, 'speaker_count_limits': None,
              'pairing_inventory_sha256': sha(OUT / 'pairing-inventory.jsonl'), 'windows_sha256': sha(OUT / 'windows.jsonl'),
              'policy_sha256': sha(HERE / 'policy.json'), 'builder_sha256': sha(__file__),
              'targets_sha256': sha(V4 / 'targets.py'), 'previous_selection_sha256': sha(OLD / 'selection-summary.json'),
              'active_run_modified': False, 'training_started_on_this_selection': False,
              'note': 'Reuses existing Soniox results. Diagnostics do not reject audio. Unresolved source speaker/word conflicts remain explicit; they are not noise rejections.'}
    atomic_json(OUT / 'selection-summary.json', report)
    atomic_json(HERE / 'result.json', report)
    (OUT / '.building').unlink()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    main(args.workers)
