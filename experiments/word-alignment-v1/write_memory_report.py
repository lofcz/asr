"""Write the measured memory/precision tradeoff, including implementation limits."""
import json
from pathlib import Path

P=Path(__file__).resolve().parent/'benchmarks/memory-v1'
data=json.loads((P/'summary.json').read_text())
names={'float32':'FP32 reference','float16':'FP16','int8':'INT8 linear weights + FP16','nf4':'NF4 linear weights + FP16'}
agg=data['aggregate']
assert set(names)<=set(agg), 'Wait for every planned comparison before publishing'
lines=[
    '# Canary auxiliary CTC: lower-memory alignment', '', 'Measured 2026-10-24 on mlserver RTX A5000 GPUs.', '',
    '**Recommendation: INT8 encoder Linear weights with FP16 compute and the existing NVIDIA tokenizer/NFA alignment.** '
    'This retains bilingual Czech/Slovak support without language flags. FP16 is the simpler fallback when speed is more important than the additional memory saving. '
    'NF4 is available when memory is exceptionally constrained, but it changes more individual boundaries.', '',
    'The comparison includes the two markers, two natural podcast excerpts, and a six-minute natural conversation with one Czech and one Slovak speaker. '
    'The fixed paired scoring set contains 5,565 words / 11,130 boundaries over 37.48 minutes. Soniox timestamps are the agreement reference, not independent human boundary ground truth. '
    'All variants receive the same audio, text, cropping and tokenizer. No new transcription or training was run.', '',
    '## Memory and boundary agreement', '',
    '| Variant | Peak allocated, short inputs | Peak allocated, continuous 6 min | Mean boundary error vs Soniox | Boundaries identical to FP32 | Mean change from FP32 |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
]
for k,name in names.items():
    r=agg[k]
    lines.append(f"| {name} | {r['segmented_peak_allocated_gpu_gib']:.3f} GiB | {r['continuous_6min_peak_allocated_gpu_gib']:.3f} GiB | {r['reference_agreement']['mae_ms']:.3f} ms | {r['identical_to_fp32_pct']:.3f}% | {r['fp32_deviation']['mae_ms']:.3f} ms |")
lines += ['', 'The large six-minute peak includes length-dependent intermediate tensors and the alignment computation. It is not the size of the resident model. '
          'The same multilingual weights can process shorter audio inputs containing either or both speakers; time-based chunking does not require language identification.', '',
          '## Resident model and actual process footprint', '',
          '| Variant | Resident PyTorch allocation | Resident process, NVML | Peak process, short inputs | Peak process, continuous 6 min |',
          '| --- | ---: | ---: | ---: | ---: |']
for k in ('float16','int8','nf4'):
    r=agg[k]
    lines.append(f"| {names[k]} | {r['resident_allocated_gpu_gib']:.3f} GiB | {r['resident_process_gpu_gib']:.3f} GiB | {r['segmented_peak_process_gpu_gib']:.3f} GiB | {r['continuous_6min_peak_process_gpu_gib']:.3f} GiB |")
lines += ['', 'NVML samples the benchmark process every 20 ms and includes CUDA context, libraries and allocator reservation. PyTorch allocation excludes that overhead. '
          'These are observed peaks, not universal upper bounds. The preceding FP32 benchmark did not collect NVML process peaks, so none are invented here.', '',
          '## Precision details', '',
          '| Variant | P90 error vs Soniox | Within 100 ms of Soniox | Largest boundary change from FP32 |',
          '| --- | ---: | ---: | ---: |']
for k,name in names.items():
    r=agg[k]
    lines.append(f"| {name} | {r['reference_agreement']['p90_ms']:.2f} ms | {r['reference_agreement']['within_100ms_pct']:.3f}% | {r['fp32_deviation']['max_ms']:.2f} ms |")
lines += ['', 'Very small improvements or regressions in mean Soniox agreement are numerical perturbations on a small panel, not evidence that quantization improves the model. '
          'Compare individual-boundary changes as well as mean error. All supplied words remain intact, all previously scored words retain timestamps, and repeated runs produce identical word timings.', '',
          '## Runtime caveat', '',
          '| Variant | Sum of per-sample median passes | Sum of first full passes |',
          '| --- | ---: | ---: |']
for k,name in names.items():
    r=agg[k]
    lines.append(f"| {name} | {r['warm_total_seconds']:.2f} s | {r['first_full_pass_seconds']:.2f} s |")
lines += ['', 'Three full passes follow a warmup on the first segment. The reduced-precision path has substantial additional overhead on the first pass through new shapes, visible above. '
          'The median is useful for repeated shapes but should not be presented as cold-request latency. Model loading and decoding are excluded. '
          'INT8 and NF4 were tested on separate A5000s concurrently and shared the host CPU, so their timing comparison is approximate; memory and timestamp comparisons remain per-process. '
          'Do not claim lower precision is necessarily faster.', '',
          '## Implementation', '',
          '- Source model is the original NVIDIA Canary auxiliary CTC, pinned as in the preceding benchmarks. No main Canary decoder is loaded.',
          '- FP16 casts encoder and decoder on CPU before moving them to GPU, avoiding a temporary full FP32 weight allocation on GPU.',
          '- INT8 uses bitsandbytes 0.46.0 `Linear8bitLt`, with `has_fp16_weights=False` and threshold 6.0. NF4 uses `Linear4bit` with FP16 compute and compressed quantization statistics.',
          '- Both quantize 217 encoder Linear modules. Convolutions, norms and the CTC decoder remain FP16. Mel preprocessing and CTC Viterbi probabilities remain FP32. This is deliberately described as mixed precision, not all-layer INT8/INT4.',
          '- The official SentencePiece tokenizer and NeMo Viterbi algorithm are unchanged. Non-finite CTC output is rejected. Source text, original segment metadata and word order are asserted unchanged.',
          '- Optional `language` metadata remains an array and is never passed to the aligner or used for routing.', '',
          'Example:', '', '```bash',
          '/data/work/asr-word-alignment/nemo-venv/bin/python /data/work/asr-word-alignment/align_canary.py \\\n+  --audio input.wav --transcript transcript.json --out aligned.json \\\n+  --precision float16 --quantization int8', '```', '',
          '## Existing quantized releases and smaller models', '',
          'The [third-party Canary auxiliary GGUF release](https://huggingface.co/cstr/canary-ctc-aligner-GGUF/tree/2b22dc9aff585bc8368a5228173c8afe22c155b7) provides the same model family in F16, Q8_0, Q5_0 and Q4_K. '
          'Actual file sizes at that pinned revision are 1,252,787,584 / 666,894,464 / 460,848,896 / 392,167,040 bytes respectively (1.253 GB / 667 MB / 461 MB / 392 MB). '
          'The model-card table contains older, larger sizes; file metadata is used here. File size is not runtime VRAM.', '',
          'That author reports identical Q4/F16 alignment on one short English JFK clip. It does not demonstrate Czech/Slovak equivalence. '
          'Its runtime also uses a different tokenizer implementation. **The measured INT8/NF4 results above apply to our bitsandbytes + NeMo path, not to those GGUF files.**', '',
          'I did not find a smaller distilled model with demonstrated suitability for this bilingual task in the checked sources:', '',
          '- [NVIDIA Canary-180M-Flash](https://huggingface.co/nvidia/canary-180m-flash) supports English, German, French and Spanish; it does not cover Czech/Slovak.',
          '- [NOESIS 112-language Qwen3 aligner](https://huggingface.co/AMAImedia/NOESIS-Qwen3-Forced-Aligner-0.6B-112LANG-BF16) stores about 918M parameters despite the 0.6B family name. Its card says mixed-language calls require pre-segmentation and were not tested. It is not a smaller, established replacement for this use case.',
          '- [LLM-ForcedAligner, ACL 2026](https://aclanthology.org/2026.acl-long.1174/) is modern research, but the examined primary publication says checkpoint/inference release is forthcoming. This does not establish a released compact Czech/Slovak replacement.', '',
          'This is a bounded search, not a claim that no suitable model can exist. Quantizing the already-tested multilingual CTC model gives a measured memory reduction without starting a distillation project.', '',
          '## Artifacts', '',
          '`summary.json` contains all per-recording and aggregate metrics. Each `sample.variant.json` preserves full word alignments and model/audio/transcript provenance. '
          '`benchmark_canary_memory.py`, `score_canary_memory.py` and `write_memory_report.py` reproduce the comparison. '
          'The baseline FP32 artifacts remain unchanged. Results are retained on the SSD, mlserver and the internal-drive backup.', '',
          'The existing hold on synthetic generation and fresh training remains in effect.', '',
]
(P/'REPORT.md').write_text('\n'.join(lines))
print(P/'REPORT.md')
