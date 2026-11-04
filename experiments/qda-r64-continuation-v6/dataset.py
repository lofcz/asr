"""Exact interval complement and Soniox crop contracts, without quality filters."""
import bisect
import collections


def complement(duration, consumed):
    cursor = 0.0
    result = []
    for start, end in sorted(consumed):
        if not 0 <= start < end <= duration + 1e-7 or start < cursor - 1e-7:
            raise ValueError('Invalid or repeated consumed interval')
        if start > cursor + 1e-7:
            result.append((cursor, start))
        cursor = end
    if cursor < duration - 1e-7:
        result.append((cursor, duration))
    return result


def safe_cuts(segments, duration):
    cuts = [0.0]
    extent = 0.0
    for a, b in zip(segments, segments[1:]):
        extent = max(extent, a['end'])
        if b['start'] >= extent:
            cuts.append((extent + b['start']) / 2)
    return sorted(set(cuts + [duration]))


def window_segments(segments, start, end, source_duration):
    chosen = []
    for segment in segments:
        a, b = segment['start'], segment['end']
        intersects = a < end and b > start
        if a == b:
            intersects = start <= a < end or (end == source_duration and a == end)
        if intersects:
            if a < start or b > end:
                raise ValueError('Window boundary cuts a Soniox segment')
            chosen.append(segment)
    if not chosen:
        raise ValueError('Empty target crop')
    speakers = {}
    return [{'Start': round(s['start'] - start, 2), 'End': round(s['end'] - start, 2),
             'Speaker': speakers.setdefault(s['speaker'], len(speakers)), 'Content': s['text']}
            for s in chosen]


def verify_partition(sources, consumed, remaining):
    """Require every eligible train second exactly once across parent + continuation."""
    by_audio = collections.defaultdict(list)
    train = {s['audio_sha256']: s for s in sources if s['split'] == 'train'}
    heldout = {s['audio_sha256'] for s in sources if s['split'] != 'train'}
    shows = {s['show'] for s in sources if s['split'] != 'train'}
    if len({r['window_id'] for r in remaining}) != len(remaining):
        raise ValueError('Duplicate continuation windows')
    for r in consumed + remaining:
        if r['audio_sha256'] in heldout or r['show'] in shows or r['split'] != 'train':
            raise ValueError('Held-out leakage')
        s = train.get(r['audio_sha256'])
        if not s or r['record_id'] != s['record_id'] or r['label_sha256'] != s['label_sha256']:
            raise ValueError('Source or label identity mismatch')
        if abs(r['source_end'] - r['source_start'] - r['duration']) > 1e-6:
            raise ValueError('Duration mismatch')
        by_audio[r['audio_sha256']].append(r)
    for key, source in train.items():
        cursor = 0.0
        for r in sorted(by_audio[key], key=lambda x: x['source_start']):
            if abs(r['source_start'] - cursor) > 1e-7 or r['source_end'] <= cursor:
                raise ValueError('Coverage gap or repeat: ' + source['record_id'])
            cursor = r['source_end']
        if abs(cursor - source['audio_duration']) > 1e-7:
            raise ValueError('Incomplete source coverage')
    return {'parent_hours': sum(r['duration'] for r in consumed) / 3600,
            'remaining_hours': sum(r['duration'] for r in remaining) / 3600,
            'total_unique_train_hours': sum(s['audio_duration'] for s in train.values()) / 3600,
            'heldout_hours': sum(s['audio_duration'] for s in sources if s['split'] != 'train') / 3600,
            'overlap_seconds': 0, 'uncovered_train_seconds': 0,
            'speaker_count_filter': False, 'acoustic_quality_filter': False}
