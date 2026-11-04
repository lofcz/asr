"""Collect measured experiments without treating failed or incomplete runs as successes."""
import json, pathlib

BASE = pathlib.Path(__file__).resolve().parent
summary = {'training': {}, 'generation': {}}
for path in sorted((BASE/'runs').glob('*/result.json')):
    r = json.loads(path.read_text())
    item = {k:r[k] for k in ('status','config','world_size','audio_realtime_factor',
            'hours_per_500h_epoch_one_gpu','hours_per_500h_epoch','peak_allocated_gib',
            'peak_reserved_gib','validation_loss_before','validation_loss_after',
            'max_adapter_difference','duplicate_sessions_in_global_batch','error_type') if k in r}
    item['steps'] = r.get('steps', [])
    summary['training'][path.parent.name] = item
for path in sorted((BASE/'generation').glob('*/result.json')):
    r = json.loads(path.read_text())
    summary['generation'][path.parent.name] = {k:v for k,v in r.items() if k != 'traceback'}
(BASE/'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
