"""Durable content-addressed TTS; uncertain submissions require explicit recovery."""
import base64
import fcntl
import io
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import wave
from core import DATA, REPLICA, digest, encode, env, read, save, put, verified

VERSION = '2026-08-14'
MODEL = 'sonic-3.6-2026-08-27'
RATE = 24000


def request_for(text, language, voice, speed=1.0, original_timestamps=False):
    return {'model_id': MODEL, 'transcript': text, 'voice': {'id': voice},
            'output_format': {'container': 'raw', 'encoding': 'pcm_s16le', 'sample_rate': RATE},
            'language': language, 'normalization': 'auto', 'add_timestamps': True,
            'use_normalized_timestamps': not original_timestamps,
            'generation_config': {'speed': speed, 'volume': 1}}


def recover(relative, request):
    # Either disk may have committed the terminal event just before a crash.
    events, raw_journal = None, None
    for base in (DATA, REPLICA):
        journal = base / relative / 'events.jsonl'
        if not journal.exists():
            continue
        try:
            raw = journal.read_bytes()
            candidate = [json.loads(line) for line in raw.splitlines()]
        except json.JSONDecodeError:
            continue
        if any(e.get('type') == 'done' for e in candidate):
            events, raw_journal = candidate, raw
            break
    if events is None:
        return None
    pcm = bytearray()
    words = []
    done = False
    for e in events:
        if done:
            raise ValueError('TTS event after completion')
        kind = e.get('type')
        if kind == 'error':
            raise ValueError('Provider returned a stream error')
        if kind == 'chunk':
            pcm.extend(base64.b64decode(e['data'], validate=True))
        elif kind == 'timestamps':
            timing = e.get('word_timestamps') or e.get('timestamps')
            if not isinstance(timing, dict):
                raise ValueError('Unrecognized timestamp event; retain raw response')
            w, s, t = timing['words'], timing['start'], timing['end']
            if not len(w) == len(s) == len(t):
                raise ValueError('Mismatched timestamp arrays')
            words.extend({'text': a, 'start': b, 'end': c} for a, b, c in zip(w, s, t))
        elif kind == 'done':
            done = True
    if not done or not pcm or len(pcm) % 2 or not words:
        raise ValueError('Incomplete audio or missing word timestamps')
    duration = len(pcm) / (2 * RATE)
    previous = -1.0
    for word in words:
        a, b = word['start'], word['end']
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (a, b)):
            raise ValueError('Nonfinite timestamp')
        if a < 0 or b < a or a < previous - .05 or b > duration + .05:
            raise ValueError('Word timestamps outside the generated audio')
        previous = a
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        wav.writeframes(pcm)
    audio = buf.getvalue()
    put(relative / 'audio.wav', audio)
    save(relative / 'words.json', words)
    receipt = {'status': 'complete', 'request_sha256': digest(encode(request)),
               'audio_sha256': digest(audio), 'words_sha256': digest(encode(words)),
               'events_sha256': digest(raw_journal), 'duration_seconds': duration,
               'sample_rate': RATE, 'samples': len(pcm) // 2, 'characters': len(request['transcript']),
               'model': MODEL, 'api_version': VERSION, 'completed_at': time.time(),
               'label_status': 'tts_alignment_requires_independent_asr_check'}
    # A process crash can occur after the SSD event commit but before its replica.
    put(relative / 'events.jsonl', raw_journal)
    save(relative / 'receipt.json', receipt)
    return receipt


def reserve(identity, request):
    """Serialize spend reservations across workers, before a paid POST."""
    (REPLICA / 'budget').mkdir(parents=True, exist_ok=True)
    with (REPLICA / 'budget' / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget = read('budget/reservations.json', {'characters': 0, 'requests': {}})
        if identity in budget['requests']:
            return
        ceiling = read('plan.json', {}).get('character_ceiling', 5000000)
        characters = len(request['transcript'])
        if budget['characters'] + characters > ceiling:
            raise RuntimeError('Configured TTS character budget reached')
        budget['requests'][identity] = characters
        budget['characters'] += characters
        budget['credit_estimate_basis'] = 'Approximately one credit per character; conservative reservation, includes unresolved requests'
        save('budget/reservations.json', budget)


def synthesize(request):
    identity = digest(encode({'api_version': VERSION, 'body': request}))
    relative = Path('utterances') / identity
    (DATA / relative).mkdir(parents=True, exist_ok=True)
    with (DATA / relative / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = read(relative / 'receipt.json')
        if receipt:
            verified(relative / 'audio.wav', receipt['audio_sha256'])
            verified(relative / 'words.json', receipt['words_sha256'])
            verified(relative / 'events.jsonl', receipt['events_sha256'])
            return identity, receipt
        intent = read(relative / 'intent.json')
        if intent:
            receipt = recover(relative, request)
            if receipt:
                return identity, receipt
            if intent.get('status') not in ('rejected_retryable',):
                recovery=read(relative/'network-recovery-001.json',{})
                used=read(relative/'network-recovery-001.used.json')
                empty=all(not (base/relative/name).exists() for base in (DATA,REPLICA)
                          for name in ('response.json','events.jsonl','audio.wav','receipt.json'))
                permitted=(not used and empty and intent.get('status')=='submitted'
                    and time.time()-intent.get('submitted_at',time.time())>300
                    and recovery.get('request_sha256')==digest(encode(request))
                    and recovery.get('max_additional_characters')==len(request['transcript'])
                    and recovery.get('max_retries')==1)
                if not permitted:
                    raise RuntimeError('Ambiguous previous TTS submission retained; no duplicate charge: ' + identity)
                # An explicitly recorded, one-time repair stays within the overall
                # character ceiling. Reserve the possible duplicate cost separately.
                reserve(identity+':network-recovery-001',request)
                save(relative/'network-recovery-001.used.json',{'at':time.time(),
                    'previous_intent':intent,'possible_additional_characters':len(request['transcript']),
                    'reason':recovery.get('reason'),'original_request_may_have_been_billed':True})
        save(relative / 'request.json', request)
        reserve(identity, request)
        started = time.time()
        save(relative / 'intent.json', {'status': 'submitted', 'submitted_at': started,
             'request_sha256': digest(encode(request)), 'characters': len(request['transcript'])})
        req = urllib.request.Request('https://api.cartesia.ai/tts/sse', data=encode(request),
              headers={'Authorization': 'Bearer ' + env()['CARTESIA_API_KEY'],
                       'Cartesia-Version': VERSION, 'Content-Type': 'application/json'})
        handles = []
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                save(relative / 'response.json', {'http_status': response.status,
                    'headers': {k: v for k, v in response.headers.items()
                                if k.lower() in ('request-id', 'x-request-id', 'content-type', 'date')}})
                for base in (REPLICA, DATA):
                    (base / relative).mkdir(parents=True, exist_ok=True)
                    handles.append((base / relative / 'events.jsonl').open('wb'))
                lines = []
                for raw in response:
                    line = raw.decode('utf-8').rstrip('\r\n')
                    if line.startswith('data:'):
                        lines.append(line[5:].lstrip())
                    if not line and lines:
                        event = json.loads('\n'.join(lines))
                        data = (json.dumps(event, ensure_ascii=False) + '\n').encode()
                        for f in handles:
                            f.write(data)
                            f.flush()
                            os.fsync(f.fileno())
                        lines = []
                        if event.get('type')=='done':
                            break
                if lines:
                    raise ValueError('Unterminated SSE frame')
        except urllib.error.HTTPError as e:
            save(relative / 'error.json', {'status': e.code, 'at': time.time(),
                'message': e.read(4000).decode(errors='replace')})
            if e.code in (400, 401, 402, 403, 404, 422, 429):
                save(relative / 'intent.json', {'status': 'rejected_retryable' if e.code == 429 else 'rejected',
                    'http_status': e.code, 'submitted_at': started})
            raise RuntimeError('Cartesia HTTP ' + str(e.code)) from None
        finally:
            for f in handles:
                f.close()
        receipt = recover(relative, request)
        if not receipt:
            raise ValueError('Stream ended without a completion event; retained for recovery')
        return identity, receipt
