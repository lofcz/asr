# Evaluation stopped at user request

Training remains paused. This is the six-recording Czech subtotal, approximately 60 minutes, using Soniox machine references. The planned ten-recording evaluation was cancelled. No conclusion about Slovak follows from this subtotal.

| Metric | Vanilla | 250h checkpoint |
|---|---:|---:|
| Number-normalized WER | 9.83% | 9.15% |
| Strict WER | 9.84% | 9.23% |
| DER, 250ms collar | 14.65% | 15.05% |

Relative word-error reduction: 6.96%.

The checkpoint produced one malformed timestamp in cs-04: `0.0.0`. Its original JSON remains invalid. Scores include a separately saved copy with only that timestamp repaired to `0.0`. Raw validity: vanilla 6/6; checkpoint 5/6.

- number_normalized_wer: checkpoint minus vanilla -0.68 pp; paired recording bootstrap 95% interval [-2.13, +1.12] pp; improved/worse/tied: 5/1/0.
- der_250ms: checkpoint minus vanilla +0.40 pp; paired recording bootstrap 95% interval [-0.29, +1.12] pp; improved/worse/tied: 3/3/0.

These intervals describe a small, selected development panel and do not account for Soniox label errors. The planned evaluation was stopped after inspecting interim results, which further limits confirmatory statistical interpretation.

| Sample | Show | Normalized WER vanilla → 250h | DER vanilla → 250h |
|---|---|---:|---:|
| cs-01 | Režim nerušit | 12.23% → 10.17% | 9.61% → 10.29% |
| cs-02 | Alex a host | 10.84% → 9.70% | 6.56% → 5.73% |
| cs-03 | Dobrý rozhovor | 12.36% → 11.59% | 1.38% → 2.68% |
| cs-04 | Kvér | 6.78% → 10.23% | 6.85% → 8.50% |
| cs-05 | Věda Plus | 9.52% → 6.28% | 23.45% → 23.13% |
| cs-06 | Kultura Plus | 6.77% → 6.05% | 38.02% → 37.99% |
