# ASR

VibeVoice fine-tuning for Czech and Slovak using QLoRA.

![Checkpoint word error rate](charts/01-checkpoint-progress.png)

**Figure 1.** Checkpoint comparison.

![Macro-averaged word error rate](charts/02-commercial-open-source.png)

**Figure 2.** Model comparison.

## Reproduce

```sh
uv run --no-project --python /usr/bin/python scripts/build_result_charts.py
```
