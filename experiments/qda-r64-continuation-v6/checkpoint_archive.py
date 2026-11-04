"""Bound resume state independently of backups; raw audio/labels are never pruned."""
import json, pathlib, shutil


def prune(run):
    from disk_retention import retain
    try:
        retain(run)
    except BlockingIOError:
        pass  # The independent retention/backup worker owns the lock briefly.
    # Frozen features are reproducible from immutable audio and the pinned model.
    # Keep the latest two batches of checkpoints for cheap rollback, and all dev.
    run=pathlib.Path(run);root=run.parents[1]
    checkpoints=sorted(p for p in run.glob('checkpoint-*') if (p/'state.pt').exists())
    if len(checkpoints)<3:return
    before=int(checkpoints[-2].name.rsplit('-',1)[1])
    plan=json.loads((root/'plan.json').read_text())
    for pair in plan['probe_pairs'][:before]:
        for row in pair:
            for suffix in ('.features.npz','.features.json'):
                (root/'cache'/(row['window_id']+suffix)).unlink(missing_ok=True)
