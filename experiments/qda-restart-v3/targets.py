"""Rebuild speech targets from immutable Soniox tokens, before applying segmentation.

Whitespace defines lexical boundaries. A timestamp gap is NEVER a word boundary.
Ambiguous speaker changes inside words fail closed; no speaker is guessed.
"""
import hashlib
import json
import math
import re

VERSION = 'whole-word-targets-v3.1'
CONTROLS = {'', '<end>', '<fin>'}


def speech_tokens(tokens, duration):
    accepted = []
    for i, token in enumerate(tokens):
        text = token.get('text', '')
        if not isinstance(text, str):
            raise ValueError(f'nonstring token {i}')
        if text == '' or text.strip() in ('<end>', '<fin>'):
            continue
        if token.get('is_audio_event'):
            raise ValueError(f'unverified audio event at token {i}')
        a, b = token.get('start_ms'), token.get('end_ms')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (a, b)):
            raise ValueError(f'invalid timestamp at token {i}')
        a, b = a / 1000, b / 1000
        if not 0 <= a <= b <= duration + .02:
            raise ValueError(f'out of bounds timestamp at token {i}')
        speaker = token.get('speaker')
        if speaker is None or isinstance(speaker, (dict, list, bool)):
            raise ValueError(f'missing or invalid speaker at token {i}')
        confidence = token.get('confidence')
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f'invalid confidence at token {i}')
        accepted.append({'text': text, 'start': a, 'end': b, 'speaker': str(speaker),
                         'token_indices': [i], 'confidence': confidence, 'language': token.get('language')})
    if not accepted:
        raise ValueError('empty transcript')
    return accepted


def reconstruct(tokens, duration):
    accepted = speech_tokens(tokens, duration)
    words, repairs = [], []
    for token in accepted:
        prev = words[-1] if words else None
        boundary = prev is None or token['text'][0].isspace() or prev['text'][-1].isspace()
        if not boundary and token['speaker'] != prev['speaker']:
            if prev['text'][-1].isalnum() and token['text'][0].isalnum():
                raise ValueError(f"speaker change inside word at token {token['token_indices'][0]}")
            boundary = True  # punctuation may precede a genuine speaker change
        if not boundary:
            gap = token['start'] - prev['end']
            if gap > .8:
                repairs.append({'token_index': token['token_indices'][0], 'gap_seconds': gap,
                                'left': prev['text'], 'right': token['text']})
            if gap > 2:
                raise ValueError(f"ambiguous intraword timestamp gap at token {token['token_indices'][0]}")
            prev['text'] += token['text']
            prev['token_indices'] += token['token_indices']
            prev['start'] = min(prev['start'], token['start'])
            prev['end'] = max(prev['end'], token['end'])
            prev['confidence'] = min(prev['confidence'], token['confidence'])
        else:
            words.append(dict(token, token_indices=list(token['token_indices'])))
    segments = []
    for word in words:
        prev = segments[-1] if segments else None
        merge = (prev is not None and word['speaker'] == prev['speaker']
                 and word['start'] >= prev['start'] and word['start'] - prev['end'] <= .8
                 and word['end'] - prev['start'] < 30
                 and not (word['end'] - prev['start'] >= 8 and prev['text'].rstrip().endswith(('.', '?', '!'))))
        if merge:
            prev['text'] += word['text']
            prev['end'] = max(prev['end'], word['end'])
            prev['token_indices'] += word['token_indices']
        else:
            segments.append({**{k: word[k] for k in ('text', 'start', 'end', 'speaker')},
                             'token_indices': list(word['token_indices'])})
    assert ''.join(s['text'] for s in segments) == ''.join(t['text'] for t in accepted)
    return segments, words, repairs


def independent_verify(tokens, segments, duration):
    """Independent checks on serialized supervision; deliberately does not call reconstruct."""
    expected = [i for i, t in enumerate(tokens) if isinstance(t.get('text'), str) and t['text'] != '' and t['text'].strip() not in ('<end>', '<fin>')]
    actual = [i for s in segments for i in s['token_indices']]
    if actual != expected:
        raise ValueError('token loss, duplication, reordering, or unverified event')
    for s in segments:
        rows = [tokens[i] for i in s['token_indices']]
        if s['text'] != ''.join(t['text'] for t in rows):
            raise ValueError('text differs from raw token stream')
        if {str(t['speaker']) for t in rows} != {s['speaker']}:
            raise ValueError('speaker reassignment')
        if s['start'] != min(t['start_ms'] for t in rows) / 1000 or s['end'] != max(t['end_ms'] for t in rows) / 1000:
            raise ValueError('timestamp extent changed')
        if not 0 <= s['start'] <= s['end'] <= duration + .02:
            raise ValueError('invalid segment extent')
    for a, b in zip(segments, segments[1:]):
        if a['text'][-1].isalnum() and b['text'][0].isalnum():
            raise ValueError('lexical word split between serialized segments')
        if b['start'] < a['start']:
            raise ValueError('nonmonotonic segment start; needs overlap review')
    raw_words = re.findall(r'\w+', ''.join(tokens[i]['text'] for i in expected).casefold())
    output_words = re.findall(r'\w+', ' '.join(s['text'] for s in segments).casefold())
    if raw_words != output_words:
        raise ValueError('word sequence changes when the model segments are joined')
    return {'tokens': len(actual), 'words': len(raw_words), 'segments': len(segments),
            'raw_stream_sha256': hashlib.sha256(''.join(tokens[i]['text'] for i in expected).encode()).hexdigest()}


def window_segments(segments, start, end):
    rows = [s for s in segments if s['start'] < end and s['end'] > start]
    if not rows or any(s['start'] < start or s['end'] > end for s in rows):
        raise ValueError('window boundary cuts speech')
    speakers = {}
    result = []
    for s in rows:
        sid = speakers.setdefault(s['speaker'], len(speakers))
        result.append({'Start': round(s['start'] - start, 2), 'End': round(s['end'] - start, 2),
                       'Speaker': sid, 'Content': s['text']})
    return result


def safe_windows(segments, duration, size):
    """Nonoverlapping complete coverage at segment boundaries, never inside overlap."""
    cuts = [0.]
    max_end = 0.
    for a, b in zip(segments, segments[1:]):
        max_end = max(max_end, a['end'])
        if b['start'] >= max_end:
            cuts.append((max_end + b['start']) / 2)
    cuts.append(duration)
    start = 0.
    result = []
    while start < duration - .01:
        if duration - start <= size * 1.10:
            end = duration
        else:
            candidates = [x for x in cuts if start + size * .7 <= x <= min(duration, start + size * 1.05)]
            if not candidates:
                raise ValueError('no safe window cut')
            end = min(candidates, key=lambda x: abs(x - start - size))
        result.append((start, end))
        start = end
    return result
