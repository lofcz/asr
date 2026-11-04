# Measured pilot results

Training uses the speech-only capacity sample described in README.md. Rich-label and curriculum requirements are tracked separately in DATASET_POLICY.md.

| Run | GPUs | Status | Peak allocated/GPU (GiB) | Audio realtime factor | Projected 500-hour pass |
|---|---:|---|---:|---:|---:|
| bf16-10m | 1 | completed | 21.96 | 57.3× | 8.7 h |
| bf16-30m-offload | 1 | completed | 21.00 | 36.2× | 13.8 h |
| bf16-30m-standard | 1 | failed | — | — | — |
| bf16-60m-offload | 1 | failed | — | — | — |
| qlora-30m | 1 | failed | — | — | — |
| qlora-30m-chunked | 1 | completed | 17.60 | 33.5× | 14.9 h |
| qlora-30m-ddp | 2 | completed | 17.84 | 66.7× | 7.5 h |
| qlora-60m-ddp | 2 | failed | — | — | — |
| qlora-60m-ddp-pageable | 2 | completed | 18.40 | 48.6× | 10.3 h |
| qlora-60m-offload | 1 | completed | 18.22 | 26.1× | 19.2 h |

Projections exclude labeling, feature extraction, initialization, checkpoint I/O, and evaluation. They extrapolate very few updates of one session. Different speaking rates/context lengths change cost. The full-hour DDP probe repeats the same source on both ranks for capacity measurement; its processed hours are not unique data. The successful pageable-offload run verified identical adapter weights across GPUs.

| Held-out run | WER | cpWER | DER | EOS / final timestamp |
|---|---:|---:|---:|---|
| baseline-context-heldout-10m | 13.22% | 13.30% | 11.60% | eos / 600.0 s |
| baseline-heldout-10m | 12.77% | 12.84% | 11.59% | eos / 600.0 s |
| bf16-30m-heldout-10m | 13.07% | 13.15% | 11.59% | eos / 600.0 s |
| qlora60-heldout-10m | 13.30% | 13.83% | 11.97% | eos / 600.0 s |

References are Soniox pseudo-labels on one separate 10-minute Slovak source. None of these small adapters improved its cpWER. The supplied-context baseline used only the guest name from episode metadata, and also did not improve this sample. These checks do not establish an optimal training schedule or the value of context on rare-term-rich sessions.

The full-hour training-source generation probe is still running at this report snapshot; consult `generation/qlora60-fullhour/progress.json` and eventually `result.json`. It is a coverage/capacity check, not held-out quality.

Recommendation: retain the base model for now, use the verified two-GPU QLoRA hardware configuration, and resolve rich-label quality plus representative evaluation before selecting an adapter for the full corpus.
