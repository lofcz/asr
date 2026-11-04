"""Reproducible word-boundary agreement benchmark against saved Soniox tokens."""
import argparse
from collections import Counter
import difflib
import json
from pathlib import Path
import re
import random
import statistics
import unicodedata

from align_existing import atomic_json, sha

HERE = Path(__file__).resolve().parent
ASR = HERE.parents[1]
OUT = HERE / 'benchmarks/soniox-v1'
SOURCES = {
    'cs': '7aee73b94a38df92305839c55b6c00e9b1a1ba9e6d21d84851bb0973d5f57047',
    'sk': 'ea1c64a9aaf53946a97ca2a0d648ad52b7996235ae1c3749041605dfeb83dc40',
}


def norm(word):
    # Keep diacritics: Slovak 'tri' must never become Czech 'tři'.
    return ''.join(c for c in unicodedata.normalize('NFC', word).casefold() if c.isalnum())


def soniox_words(raw):
    tokens = raw['tokens']
    text = ''.join(t['text'] for t in tokens)
    assert text == raw['text'], 'Raw token text does not reconstruct transcript'
    spans, pos = [], 0
    for ti, t in enumerate(tokens):
        spans.append((pos, pos + len(t['text']), ti, t))
        pos += len(t['text'])
    words, cursor = [], 0
    for wi, m in enumerate(re.finditer(r'\S+', text)):
        lexical = {i for i in range(m.start(), m.end()) if text[i].isalnum()}
        if not lexical:
            continue
        while cursor < len(spans) and spans[cursor][1] <= m.start():
            cursor += 1
        contributors, ti_span = [], cursor
        while ti_span < len(spans) and spans[ti_span][0] < m.end():
            a, b, ti, t = spans[ti_span]
            if any(a <= i < b for i in lexical):
                contributors.append((ti, t))
            ti_span += 1
        problems = []
        for ti, t in contributors:
            if t.get('is_audio_event') or t.get('translation_status') not in (None, 'original'):
                problems.append('non_original_speech_token')
            if len([w for w in t['text'].split() if norm(w)]) > 1:
                problems.append('token_spans_multiple_words')
        starts = [t.get('start_ms') for _, t in contributors]
        ends = [t.get('end_ms') for _, t in contributors]
        if not contributors or None in starts + ends:
            a = b = None
            problems.append('missing_timestamp')
        else:
            a, b = min(starts) / 1000, max(ends) / 1000
            if not 0 <= a < b:
                problems.append('nonpositive_duration')
        words.append({'word': m.group(), 'norm': norm(m.group()), 'index': wi,
                      'start': a, 'end': b, 'token_indices': [ti for ti, _ in contributors],
                      'issues': sorted(set(problems)),
                      'confidence': min(t.get('confidence', 0) for _, t in contributors)})
    return words


def prepare():
    manifest = []
    for lang, digest in SOURCES.items():
        source = ASR / 'experiments/production-reference-audit-v2'
        raw_path = source / 'records' / digest / 'soniox.raw.json'
        transcript = ASR / 'experiments/qda-full-r64-v4/markers/r64-full-final/finetuned' / lang / 'hypothesis.json'
        reference = soniox_words(json.loads(raw_path.read_text()))
        segments = json.loads(transcript.read_text())
        atomic_json(OUT / f'{lang}.transcript.json', segments)
        atomic_json(OUT / f'{lang}.reference.json', reference)
        manifest.append({'id': lang, 'language': lang, 'kind': 'synthetic_production_marker',
                         'audio': str(source / lang / 'audio.playback.wav'),
                         'remote_audio': f'/data/work/asr-evaluation/production-reference-audit-v2/{lang}/audio.playback.wav',
                         'audio_sha256': sha(source / lang / 'audio.playback.wav'),
                         'source_transcript': str(transcript), 'transcript_sha256': sha(transcript),
                         'soniox_raw': str(raw_path), 'soniox_raw_sha256': sha(raw_path),
                         'reference_words': len(reference),
                         'reference_issue_counts': dict(Counter(x for w in reference for x in w['issues']))})
    panel_root = ASR / 'experiments/unseen-250h-v1'
    panel = json.loads((panel_root/'panel.json').read_text())
    for sid in ('cs-01', 'sk-01'):
        entry = next(s for s in panel['samples'] if s['sample_id'] == sid)
        lang = entry['language']
        name = lang + '-natural'
        raw_path = ASR / 'datasets/soniox-50h-v2' / entry['raw_transcript_path']
        assert sha(raw_path) == entry['raw_transcript_sha256']
        words = soniox_words(json.loads(raw_path.read_text()))
        lo, hi = entry['source_start'], entry['source_end']
        reference = [{**w, 'start': w['start']-lo, 'end': w['end']-lo} for w in words
                     if w['start'] is not None and lo <= w['start'] and w['end'] <= hi]
        transcript = panel_root / 'vanilla' / sid / 'hypothesis.json'
        original = json.loads(transcript.read_text())
        segments = [s for s in original if s.get('Speaker') is not None]
        atomic_json(OUT / f'{name}.transcript.json', segments)
        atomic_json(OUT / f'{name}.reference.json', reference)
        audio = panel_root/'inputs'/sid/'audio.wav'
        receipt = json.loads((audio.parent/'receipt.json').read_text())
        assert sha(audio) == receipt['crop_sha256']
        manifest.append({'id': name, 'language': lang, 'kind': 'natural_podcast', 'title': entry['title'],
                         'audio': str(audio),
                         'remote_audio': f'/data/work/asr-evaluation/unseen-250h-v1/inputs/{sid}/audio.wav',
                         'audio_sha256': sha(audio), 'source_start': lo, 'source_end': hi,
                         'source_transcript': str(transcript), 'transcript_sha256': sha(transcript),
                         'soniox_raw': str(raw_path), 'soniox_raw_sha256': sha(raw_path),
                         'reference_words': len(reference), 'excluded_non_speech_segments': len(original)-len(segments),
                         'reference_issue_counts': dict(Counter(x for w in reference for x in w['issues']))})
    atomic_json(OUT / 'manifest.json', manifest)
    atomic_json(OUT / 'protocol.json', {
        'reference': 'Saved Soniox token timestamps, aggregated into whitespace words',
        'boundary_convention': 'First/last alphanumeric token; punctuation-only tokens do not extend word bounds',
        'metric': 'Absolute start and end deviations in milliseconds; pooled boundaries and separate starts/ends',
        'same_input': 'Both models receive identical saved VibeVoice transcripts (v4-final markers; existing vanilla natural speech) and identical segment crops with 200 ms padding. No Soniox word boundaries supplied to aligners.',
        'text_matching': 'Monotonic exact word sequence matching after case/punctuation normalization; retain diacritics; no number equivalence',
        'exclusions': 'Unmatched text, ambiguous/invalid Soniox reference boundaries, and unsupported character wildcard words; report counts. No exclusion based on measured alignment error.',
        'pairing': 'Headline scores use exactly the same matched valid words for both models; missing alignments counted separately',
        'precision': 'FP32, official model weights; WhisperX and NVIDIA NeMo Forced Aligner implementations',
        'limits': 'Agreement with Soniox is not independently annotated timing accuracy. Four recordings (two markers, two natural podcasts) are a pilot, not a broad population benchmark.',
        'calibration': 'No timing offset fitted to this reference',
    })
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def percentile(values, p):
    v = sorted(values)
    if not v:
        return None
    k = (len(v)-1) * p
    lo = int(k)
    return v[lo] + (v[min(lo+1, len(v)-1)]-v[lo])*(k-lo)


def stats(signed):
    absolute = [abs(x) for x in signed]
    return {'n': len(signed), 'mae_ms': statistics.mean(absolute),
            'median_ms': statistics.median(absolute), 'p90_ms': percentile(absolute, .9),
            'p95_ms': percentile(absolute, .95), 'max_ms': max(absolute),
            'signed_bias_ms': statistics.mean(signed),
            **{f'within_{t}ms_pct': 100*sum(x <= t+1e-7 for x in absolute)/len(absolute) for t in (50, 100, 200)}}


def score():
    summaries = []
    for sample in json.loads((OUT/'manifest.json').read_text()):
        lang = sample['id']
        ref = json.loads((OUT/f'{lang}.reference.json').read_text())
        runs = {name: json.loads((OUT/f'{lang}.{name}.json').read_text()) for name in ('wav2vec2', 'canary')}
        assert all(r['audio_sha256'] == sample['audio_sha256'] for r in runs.values())
        assert all(r['transcript_sha256'] == sha(OUT/f'{lang}.transcript.json') for r in runs.values())
        words = {name: [w for s in run['segments'] for w in s['words']] for name, run in runs.items()}
        assert [w['word'] for w in words['wav2vec2']] == [w['word'] for w in words['canary']]
        a, b = [norm(w['word']) for w in words['wav2vec2']], [w['norm'] for w in ref]
        mapping = {}
        for op, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == 'equal':
                mapping.update(zip(range(i,j), range(k,l)))
        rows, exclusions = [], Counter()
        for hi, word in enumerate(words['wav2vec2']):
            if hi not in mapping:
                exclusions['unmatched_transcript_word'] += 1
                continue
            rw = ref[mapping[hi]]
            reason = []
            if rw['issues']:
                reason.append('reference_issue')
            for name in runs:
                w = words[name][hi]
                if w.get('start') is None or w.get('end') is None:
                    reason.append(name+'_missing_alignment')
                if 'unsupported_characters' in w.get('issues', []):
                    reason.append(name+'_unsupported_characters')
            if reason:
                exclusions.update(set(reason))
                continue
            row = {'index': hi, 'word': word['word'], 'source_segment': word['source_segment'], 'reference': rw}
            for name in runs:
                w = words[name][hi]
                row[name] = {**w, 'start_error_ms': 1000*(w['start']-rw['start']),
                             'end_error_ms': 1000*(w['end']-rw['end'])}
            rows.append(row)
        result = {'sample': lang, 'language': sample['language'], 'kind': sample['kind'],
                  'reference_words': len(ref), 'supplied_words': len(a),
                  'matched_words': len(mapping), 'paired_words': len(rows), 'exclusions': dict(exclusions), 'models': {}}
        for name, run in runs.items():
            starts = [r[name]['start_error_ms'] for r in rows]
            ends = [r[name]['end_error_ms'] for r in rows]
            result['models'][name] = {'boundaries': stats(starts+ends), 'starts': stats(starts), 'ends': stats(ends),
                                      'runtime': run['summary'],
                                      'both_boundaries_within_100ms_pct': 100*sum(max(abs(a),abs(b)) <=100+1e-7 for a,b in zip(starts,ends))/len(rows)}
            if name == 'canary':
                raw_starts = [1000*(r[name]['nfa_raw_start']-r['reference']['start']) for r in rows]
                raw_ends = [1000*(r[name]['nfa_raw_end']-r['reference']['end']) for r in rows]
                result['models'][name]['raw_nfa_including_punctuation'] = stats(raw_starts+raw_ends)
        groups = {}
        for row in rows:
            delta = sum(abs(row['canary'][k])-abs(row['wav2vec2'][k]) for k in ('start_error_ms', 'end_error_ms'))
            group = groups.setdefault(row['source_segment'], [0.0, 0])
            group[0] += delta
            group[1] += 2
        clusters = list(groups.values())
        rng = random.Random(20261024)
        draws = []
        for _ in range(2000):
            selected = rng.choices(clusters, k=len(clusters))
            draws.append(sum(x[0] for x in selected)/sum(x[1] for x in selected))
        result['paired_difference'] = {
            'canary_minus_wav2vec2_mae_ms': sum(x[0] for x in clusters)/sum(x[1] for x in clusters),
            'segment_bootstrap_95pct_ms': [percentile(draws, .025), percentile(draws, .975)],
            'segments': len(clusters), 'resamples': 2000,
            'interpretation': 'Conditional on this recording and Soniox reference; does not estimate generalization across recordings.',
        }
        atomic_json(OUT/f'{lang}.comparison.json', {'summary': result, 'words': rows})
        summaries.append(result)
    atomic_json(OUT/'summary.json', {'reference': 'Soniox word timing agreement', 'samples': summaries})
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'score'])
    args = p.parse_args()
    prepare() if args.action == 'prepare' else score()
