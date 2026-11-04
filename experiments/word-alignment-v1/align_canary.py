"""Align supplied transcript with Canary's auxiliary CTC model and official NFA.

The main Canary encoder/decoder is never loaded. No replacement ASR text is used.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
import statistics
import time
import unicodedata

from align_existing import atomic_json, decode, sha

HERE = Path(__file__).resolve().parent


def quantize_linear_layers(module, mode):
    """Quantize Linear weights only; leave convolutions and norms in FP16."""
    import torch
    import bitsandbytes as bnb
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            if mode == 'int8':
                replacement = bnb.nn.Linear8bitLt(child.in_features, child.out_features,
                    bias=child.bias is not None, has_fp16_weights=False, threshold=6.0)
                replacement.weight = bnb.nn.Int8Params(child.weight.detach(), requires_grad=False,
                                                       has_fp16_weights=False)
            else:
                replacement = bnb.nn.Linear4bit(child.in_features, child.out_features,
                    bias=child.bias is not None, compute_dtype=torch.float16,
                    compress_statistics=True, quant_type='nf4')
                replacement.weight = bnb.nn.Params4bit(child.weight.detach(), requires_grad=False,
                    compress_statistics=True, quant_type='nf4', module=replacement)
            if child.bias is not None:
                replacement.bias = torch.nn.Parameter(child.bias.detach(), requires_grad=False)
            setattr(module, name, replacement)
            count += 1
        else:
            count += quantize_linear_layers(child, mode)
    return count


def load_model(device, precision='float32', quantization='none'):
    import torch
    from omegaconf import OmegaConf, open_dict
    from nemo.collections.asr.models import EncDecCTCModelBPE
    directory = HERE / 'canary-aux'
    cfg = OmegaConf.load(directory / 'timestamps_asr_model_config.yaml')
    with open_dict(cfg):
        for key in ('train_ds', 'validation_ds', 'test_ds'):
            cfg[key] = None
        for key in ('model_path', 'vocab_path', 'spe_tokenizer_vocab'):
            cfg.tokenizer[key] = str(directory / str(cfg.tokenizer[key]).removeprefix('nemo:'))
        cfg.preprocessor.dither = 0.0
    model = EncDecCTCModelBPE(cfg=cfg)
    state = torch.load(directory / 'timestamps_asr_model_weights.ckpt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    logical_parameters = sum(p.numel() for p in model.parameters())
    dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[precision]
    if quantization != 'none' and precision != 'float16':
        raise ValueError('Quantized encoder requires float16 compute in this tested path')
    # Cast while still on CPU so loading the reduced model never requires the
    # original FP32 weight footprint on GPU. Keep audio preprocessing FP32.
    model.encoder.to(dtype=dtype)
    model.decoder.to(dtype=dtype)
    replaced = quantize_linear_layers(model.encoder, quantization) if quantization != 'none' else 0
    model.eval().requires_grad_(False).to(device)
    model._alignment_compute_dtype = dtype
    model._alignment_precision = precision
    model._alignment_quantization = quantization
    model._alignment_quantized_linear_layers = replaced
    model._alignment_logical_parameters = logical_parameters
    return model


def align_segments(segments, audio, model, device, padding=.2):
    import torch
    from nemo.collections.asr.parts.utils.aligner_utils import (
        get_utt_obj, viterbi_decoding, add_t_start_end_to_utt_obj, Segment, Word,
    )
    timestep = float(model.cfg.preprocessor.window_stride) * model.encoder.subsampling_factor
    blank = model.tokenizer.vocab_size
    duration = len(audio)/16000
    output, raw = [], []
    gpu = torch.device(device).type == 'cuda'
    if gpu:
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for index, segment in enumerate(segments):
            spans = list(re.finditer(r'\S+', segment['Content']))
            # SentencePiece applies NFKC internally (e.g. … -> ...). NFA's
            # capitalization recovery must see the same spelling. Output words
            # and original character offsets below still come from `spans`.
            nfa_words = [unicodedata.normalize('NFKC',m.group()) for m in spans]
            text = ' '.join(nfa_words)
            lo = max(0, int((float(segment['Start'])-padding)*16000))
            hi = min(len(audio), int((float(segment['End'])+padding)*16000))
            offset = lo/16000
            signal = torch.from_numpy(audio[lo:hi].copy()).unsqueeze(0).to(device)
            length = torch.tensor([signal.shape[-1]], device=device)
            dtype = getattr(model, '_alignment_compute_dtype', torch.float32)
            with torch.amp.autocast(torch.device(device).type, dtype=dtype,
                                    enabled=dtype != torch.float32):
                log_probs, encoded_len, _ = model(input_signal=signal, input_signal_length=length)
            frames = int(encoded_len[0])
            log_probs = log_probs[:, :frames].float()
            if not torch.isfinite(log_probs).all():
                raise ValueError('Non-finite CTC probabilities; refuse to manufacture alignment')
            utt = get_utt_obj(text=text, T=frames, model=model, segment_separators=None, utt_id=str(index))
            ids = torch.tensor([utt.token_ids_with_blanks], dtype=torch.long)
            path = viterbi_decoding(log_probs, ids, torch.tensor([frames]), torch.tensor([ids.shape[1]]), device)[0]
            utt = add_t_start_end_to_utt_obj(utt, path, timestep)
            aligned = [w for s in utt.segments_and_tokens if isinstance(s, Segment)
                       for w in s.words_and_tokens if isinstance(w, Word)]
            assert [w.text for w in aligned] == nfa_words, 'NFA changed input words'
            raw.append({'source_segment': index, 'crop_start': offset, 'crop_end': hi/16000,
                        'utterance': asdict(utt)})
            words = []
            for wi, (span, w) in enumerate(zip(spans, aligned)):
                tokens = [t for t in w.tokens if t.token_id != blank and any(c.isalnum() for c in t.text_cased)]
                issues = []
                if not tokens or any(t.t_start is None or t.t_start < 0 or t.t_end <= t.t_start for t in tokens):
                    a = b = None
                    issues.append('unaligned')
                else:
                    a, b = offset + tokens[0].t_start, offset + tokens[-1].t_end
                    if not 0 <= a < b <= duration:
                        a = b = None
                        issues.append('invalid_or_zero_duration')
                if any(t.token_id == 0 for t in tokens):
                    issues.append('unsupported_characters')
                words.append({'word': span.group(), 'speaker': segment.get('Speaker'),
                              'source_segment': index, 'word_index': wi,
                              'char_start': span.start(), 'char_end': span.end(),
                              'start': a, 'end': b, 'issues': issues,
                              'status': 'review' if issues else 'aligned',
                              'nfa_raw_start': offset+w.t_start if w.t_start is not None else None,
                              'nfa_raw_end': offset+w.t_end if w.t_end is not None else None})
            output.append({**segment, 'words': words})
    if gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    flat = [w for s in output for w in s['words']]
    return {'segments': output, 'raw_alignment': raw, 'summary': {
        'words': len(flat), 'timed_words': sum(w['start'] is not None for w in flat),
        'review_words': sum(bool(w['issues']) for w in flat),
        'issues': dict(Counter(x for w in flat for x in w['issues'])),
        'alignment_seconds': elapsed, 'audio_seconds': duration,
        'realtime_factor': elapsed/duration, 'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30 if gpu else 0,
        'precision': getattr(model, '_alignment_precision', 'float32'),
        'quantization': getattr(model, '_alignment_quantization', 'none'),
        'quantized_linear_layers': getattr(model, '_alignment_quantized_linear_layers', 0),
        'source_segments_preserved': True, 'source_words_preserved': True, 'speaker_ids_preserved': True,
        'timestamp_interpolation': False, 'output_frame_seconds': timestep,
        'ctc_input_unicode_normalization': 'NFKC; exact original output words preserved',
        'word_bounds': 'First/last lexical token; standalone punctuation excluded. Original NFA bounds also retained.',
    }}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--language', nargs='+', choices=['cs', 'sk'],
                   help='Optional language array for output metadata only; never passed to the model or used to route audio.')
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--transcript', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--precision', choices=['float32', 'float16', 'bfloat16'], default='float32')
    p.add_argument('--quantization', choices=['none', 'int8', 'nf4'], default='none')
    args = p.parse_args()
    import torch
    import nemo
    torch.set_num_threads(2)
    torch.manual_seed(0)
    t = time.perf_counter()
    model = load_model(args.device, args.precision, args.quantization)
    loading = time.perf_counter()-t
    audio = decode(args.audio)
    segments = json.loads(args.transcript.read_text())
    align_segments(segments[:1], audio, model, args.device)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    results = []
    for repeat in range(args.repeats):
        r = align_segments(segments, audio, model, args.device)
        print(json.dumps({'repeat': repeat, **r['summary']}), flush=True)
        results.append(r)
    # Verify repeats are identical, not just fast.
    assert all(r['segments'] == results[0]['segments'] for r in results)
    result = results[0]
    times = [r['summary']['alignment_seconds'] for r in results]
    result['summary'].update({'alignment_seconds': statistics.median(times), 'repeat_seconds': times,
                              'realtime_factor': statistics.median(times)/(len(audio)/16000), 'warmed_up': True})
    result.update({'model': json.loads((HERE/'canary-aux/provenance.json').read_text()),
                   'method': 'Canary auxiliary CTC only; official NeMo Forced Aligner Viterbi',
                   'nemo_version': nemo.__version__, 'precision': args.precision,
                   'quantization': args.quantization,
                   'transcript_sha256': sha(args.transcript), 'audio_sha256': sha(args.audio),
                   'model_load_seconds': loading, 'parameters': model._alignment_logical_parameters})
    if args.language is not None:
        result['language'] = list(dict.fromkeys(args.language))
    atomic_json(args.out, result)


if __name__ == '__main__':
    main()
