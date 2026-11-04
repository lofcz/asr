# QLoRA v1 release comparison

Completed: 136/136 model/recording evaluations.

32 main recordings, 16 per language; 319.1 minutes. Two additional long sessions total 85.1 minutes. Soniox references are unchanged. Synthetic markers are reported separately.

| Checkpoint | Main scored / expected | CS WER | SK WER | Balanced WER | Ordinary WER | DER | Long scored / 2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 31/32 | 10.079% | 10.487% | 10.283% | 10.323% | 7.806% | 1/2 |
| 300h | 28/32 | 8.134% | 7.850% | 7.992% | 8.066% | 10.788% | 0/2 |
| 450h | 32/32 | 7.834% | 7.247% | 7.540% | 7.624% | 10.563% | 0/2 |
| final | 32/32 | 7.676% | 7.201% | 7.439% | 7.510% | 11.381% | 0/2 |

WER/DER for incomplete panels cover only scored clips. Compare checkpoint values only on matched recordings; failures remain in the completion counts. Diacritics and nonnumeric inflections remain scored.

Failures recorded: 12.
- 300h / sk / bea38b20cafa4c3ace1c9015: eos — Invalid segment timestamps
- 300h / cs / 6be4a65cbb76cd96a26a041e: eos — Invalid segment timestamps
- 300h / cs / 5f8da4541111926d1cafb1b0: wall_budget — Incomplete generation: wall_budget
- 300h / sk / c69f1a93cc94d41237c49fd2: eos — Invalid segment timestamps
- 300h / sk / a730ef863bb30e51da363aad: eos — Invalid segment timestamps
- 300h / sk / a72935a2dfff6b5f4401b9b1: wall_budget — Incomplete generation: wall_budget
- 450h / cs / 5f8da4541111926d1cafb1b0: wall_budget — Incomplete generation: wall_budget
- 450h / sk / a72935a2dfff6b5f4401b9b1: wall_budget — Incomplete generation: wall_budget
- final / cs / 5f8da4541111926d1cafb1b0: wall_budget — Incomplete generation: wall_budget
- final / sk / a72935a2dfff6b5f4401b9b1: wall_budget — Incomplete generation: wall_budget
- vanilla / sk / 7c27c171d424846910930662: repetition_abort — Incomplete generation: repetition_abort
- vanilla / sk / a72935a2dfff6b5f4401b9b1: wall_budget — Incomplete generation: wall_budget

Recommended temporary v1 candidate: **450h**. Lowest text point estimate: final. Unresolved candidates under the paired show-level interval: final, 450h.

Complete per-language/cohort results, paired confidence intervals and marker scores are in report.json. Word-level errors are in word-errors.csv. No model is automatically deployed.

| Comparison: second minus first | WER delta 95% interval, recordings | WER delta 95% interval, shows |
|---|---:|---:|
| vanilla→300h | -3.315 to -1.790 pp | -3.250 to -1.764 pp |
| vanilla→450h | -3.562 to -2.171 pp | -3.417 to -2.184 pp |
| vanilla→final | -3.681 to -2.276 pp | -3.521 to -2.285 pp |
| 300h→450h | -0.900 to -0.195 pp | -0.991 to -0.236 pp |
| 300h→final | -1.008 to -0.256 pp | -1.066 to -0.282 pp |
| 450h→final | -0.285 to +0.064 pp | -0.286 to +0.092 pp |

Negative differences favor the second checkpoint. An interval crossing zero leaves that text difference unresolved. Intervals are descriptive after checkpoint selection.

| Checkpoint | Existing development WER | New development WER | Reserved confirmation WER |
|---|---:|---:|---:|
| vanilla | 8.304% | 12.534% | 7.524% |
| 300h | 5.854% | 9.640% | 6.512% |
| 450h | 5.771% | 9.008% | 6.042% |
| final | 5.738% | 8.846% | 6.007% |

**Long-session limitation:** the recommended checkpoint did not complete both unchunked long sessions. This evaluation does not establish reliable 30–60 minute direct decoding.
