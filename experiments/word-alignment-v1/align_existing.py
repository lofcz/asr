"""WhisperX alignment only: preserve every supplied segment, word and speaker."""
import argparse
from collections import Counter
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor/whisperx'))
os.environ.setdefault('HF_HOME', str(HERE / 'hf'))
os.environ.setdefault('NLTK_DATA', str(HERE / 'nltk_data'))
WHISPERX_COMMIT = '5d3678b8535698a2826a5db9eb3e4105aef85dd9'
MODELS = {'cs': 'comodoro/wav2vec2-xls-r-300m-cs-250',
          'sk': 'comodoro/wav2vec2-xls-r-300m-sk-cv8'}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.building')
    with tmp.open('w') as f:
        json.dump(clean_json(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(4 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def decode(path):
    import numpy as np
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-threads', '1', '-i', str(path),
               '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', '-']
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not np.isfinite(audio).all() or not audio.size:
        raise ValueError('Invalid decoded audio')
    return audio


def load_aligner(language, device):
    from huggingface_hub import HfApi, snapshot_download
    import whisperx
    lock = HERE / ('model-' + language + '.json')
    if lock.exists():
        pin = json.loads(lock.read_text())
    else:
        info = HfApi().model_info(MODELS[language])
        pin = {'model_id': MODELS[language], 'revision': info.sha}
        atomic_json(lock, pin)
    snapshot = snapshot_download(pin['model_id'], revision=pin['revision'],
        allow_patterns=['*.json', '*.bin', '*.safetensors', 'README.md'],
        cache_dir=str(HERE / 'hf/hub'))
    model, metadata = whisperx.load_align_model(language_code=language, device=device,
                                               model_name=snapshot, model_cache_only=True)
    model.eval()
    return model, metadata, pin


def align_segments(segments, audio, model, metadata, device, padding=.2):
    import torch
    import whisperx
    duration = len(audio) / 16000
    output, raw = [], []
    torch.cuda.synchronize() if device.startswith('cuda') else None
    started = time.perf_counter()
    for index, segment in enumerate(segments):
        text = segment.get('Content', segment.get('text', ''))
        speaker = segment.get('Speaker', segment.get('speaker'))
        start = float(segment.get('Start', segment.get('start', 0)))
        end = float(segment.get('End', segment.get('end', 0)))
        if not math.isfinite(start + end) or end <= start or start >= duration or end <= 0:
            raise ValueError('Invalid source segment bounds at ' + str(index))
        spans = list(re.finditer(r'\S+', text))
        normalized = ' '.join(s.group() for s in spans)
        request = {'text': normalized, 'start': max(0., start - padding),
                   'end': min(duration, end + padding)}
        aligned = whisperx.align([request], model, metadata, audio, device,
                                  interpolate_method='ignore', return_char_alignments=False)
        raw.append({'source_segment': index, 'alignment_input': request, 'result': aligned})
        words = aligned['word_segments']
        expected, actual = [m.group() for m in spans], [w['word'] for w in words]
        mapping = {}
        for tag, a, b, c, d in difflib.SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes():
            if tag == 'equal':
                mapping.update(zip(range(a, b), range(c, d)))
        aligned_words = []
        for wi, span in enumerate(spans):
            item = {'word': span.group(), 'speaker': speaker, 'source_segment': index,
                    'word_index': wi, 'char_start': span.start(), 'char_end': span.end(),
                    'start': None, 'end': None, 'score': None, 'issues': []}
            if wi in mapping:
                w = words[mapping[wi]]
                item.update({k: w.get(k) for k in ('start', 'end', 'score')})
            a, b = item['start'], item['end']
            if a is None or b is None or not math.isfinite(a + b):
                item['start'] = item['end'] = None
                item['issues'].append('unaligned')
            elif not 0 <= a < b <= duration:
                item['issues'].append('invalid_or_zero_duration')
                item['raw_start'], item['raw_end'] = a, b
                item['start'] = item['end'] = None
            else:
                if (item['score'] or 0) < .5:
                    item['issues'].append('low_alignment_score')
            unknown = [c for c in span.group().lower() if c.isalnum() and c not in metadata['dictionary']]
            if unknown:
                item['issues'].append('unsupported_characters')
                item['unsupported_characters'] = sorted(set(unknown))
            item['status'] = 'review' if item['issues'] else 'aligned'
            aligned_words.append(item)
        output.append({**segment, 'words': aligned_words})
    torch.cuda.synchronize() if device.startswith('cuda') else None
    wall = time.perf_counter() - started
    flat = [w for s in output for w in s['words']]
    for original, enriched in zip(segments, output):
        assert {k: enriched[k] for k in original} == original, 'Source segment was modified'
        text = original.get('Content', original.get('text', ''))
        assert [m.group() for m in re.finditer(r'\S+', text)] == [w['word'] for w in enriched['words']]
    stats = {'words': len(flat), 'timed_words': sum(w['start'] is not None for w in flat),
             'review_words': sum(w['status'] == 'review' for w in flat),
             'issues': dict(Counter(issue for w in flat for issue in w['issues'])),
             'alignment_seconds': wall, 'audio_seconds': duration, 'realtime_factor': wall / duration,
             'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30 if device.startswith('cuda') else 0,
             'source_segments_preserved': True, 'source_words_preserved': True,
             'speaker_ids_preserved': True, 'timestamp_interpolation': False,
             'score_note': 'CTC path scores are not calibrated word-boundary probabilities; 0.5 is a review heuristic.'}
    return {'segments': output, 'summary': stats, 'raw_alignment': raw}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['cs', 'sk'], required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--transcript', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    start = time.perf_counter()
    model, metadata, pin = load_aligner(args.language, args.device)
    loading = time.perf_counter() - start
    audio = decode(args.audio)
    segments = json.loads(args.transcript.read_text())
    if isinstance(segments, dict):
        segments = segments['segments']
    result = align_segments(segments, audio, model, metadata, args.device)
    result.update({'language': args.language, 'model': pin, 'whisperx_commit': WHISPERX_COMMIT,
                   'transcript_sha256': sha(args.transcript), 'audio_sha256': sha(args.audio),
                   'model_load_seconds': loading, 'method': 'WhisperX forced alignment only'})
    atomic_json(args.out, result)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
