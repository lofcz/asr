"""Read-only view of the active run. Probe/calibration data never enters its charts."""
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / 'experiments/qda-full-r64-v4'
DATA = ROOT / 'datasets/releases/qda-full-r64-v4'
RECOVERY = ROOT / 'experiments/qda-data-recovery-v5'


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def dashboard_data():
    active=read(ROOT/'experiments/active-training.json',{})
    FULL=ROOT/'experiments'/active.get('experiment','qda-full-r64-v4')
    DATA=ROOT/'datasets/releases'/active.get('dataset','qda-full-r64-v4')
    continuation=active.get('experiment')=='qda-r64-continuation-v6'
    run = FULL / 'runs/r64-full'
    events = []
    try:
        for line in (run / 'events.jsonl').read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a partial final append is not a completed step
    except FileNotFoundError:
        pass
    steps = {}
    validation = {}
    for event in events:
        if event.get('event') == 'step':
            steps[event['step']] = event  # resumed/replayed updates replace their previous observation
        if event.get('event') == 'validation':
            validation[event['step'], bool(event.get('full_panel'))] = event
    steps = [steps[k] for k in sorted(steps)]
    latest = steps[-1] if steps else {}
    config = read(run / 'config.json', {})
    if not config:
        selection=read(DATA/'selection-summary.json',{})
        spec=read(FULL/'run-spec.json',{})
        prepared=read(FULL/'prepared.json',{})
        config={'rank':64,'world_size':2,'data_hours':selection.get('train_hours',prepared.get('train_hours')),
                'steps':prepared.get('optimizer_steps'),'warmup_steps':prepared.get('warmup_steps'),
                'lr':spec.get('learning_rate'),'mode':'nf4-qlora-bf16-compute',
                'initialization':spec.get('initialization'),'fresh_optimizer':spec.get('fresh_optimizer')}
    recent = steps[-80:]
    active_seconds = sum(e.get('wall_seconds', 0) for e in recent)
    audio_seconds = sum(e.get('audio_hours', 0) * 3600 for e in recent)
    realtime = audio_seconds / active_seconds if active_seconds else None
    hours = latest.get('audio_hours_seen', 0)
    next_eval = (math.floor(hours / 50) + 1) * 50
    next_eval = min(next_eval, config.get('data_hours') or next_eval)
    evals = []
    paths = list(FULL.glob('r64-full-*h-result.json')) + list(FULL.glob('r64-full-result.json'))
    seen = set()
    for path in paths:
        item = read(path)
        if item and (item.get('step'), item.get('audio_hours_seen')) not in seen:
            seen.add((item.get('step'), item.get('audio_hours_seen')))
            evals.append({'file': path.name, **item})
    evals.sort(key=lambda x: x.get('audio_hours_seen', 0))
    parent_report=read(ROOT/'experiments/qda-full-r64-v4/r64-full-result.json',{}) if continuation else {}
    if continuation and parent_report:
        evals.insert(0,{**parent_report,'audio_hours_seen':0,'step':0,'from_parent':True,'parent_audio_hours':active['parent_audio_hours']})
    receipts = []
    for path in (run / 'backup-receipts').glob('checkpoint-*.json'):
        value = read(path, {})
        checkpoint = run / path.stem
        # A receipt is meaningful only while its actual local backup exists.
        if (checkpoint / 'adapter_model.safetensors').exists():
            receipts.append({'checkpoint': path.stem, 'step': int(path.stem.rsplit('-', 1)[1]),
                             'verified_at': value.get('verified_at'),
                             'full_resume_state': (checkpoint / 'state.pt').exists()})
    receipts.sort(key=lambda r: r['step'])
    health = read(FULL / 'health-status.json', {})
    live = read(FULL / 'live-sync-status.json', {})
    stage = read(FULL / 'experiment-status.json', {})
    awaiting_approval=stage.get('stage')=='awaiting-final-approval'
    final_report = read(FULL / 'r64-full-result.json', {})
    report_complete = final_report.get('report_complete', False)
    training_complete = read(run / 'completed.json') is not None
    now = time.time()
    warnings = []
    if now - live.get('mirrored_at', 0) > 90:
        warnings.append('Live metrics have not synced for more than 90 seconds.')
    calibrating=continuation and stage.get('stage') in ('preparing_remaining_audio','reusing_fixed_validation','continuation_lr_range','analyzing_continuation_lr','lr_range_complete')
    if not report_complete and not calibrating and not awaiting_approval and now - health.get('checked_at', 0) > 180:
        warnings.append('Training health heartbeat is stale.')
    if not report_complete and not awaiting_approval and (stage.get('stage', '').endswith('failed') or health.get('running') is False):
        warnings.append('Training is not reporting a healthy running process; inspect the incident status.')
    if report_complete and final_report.get('output_issues'):
        issues = final_report['output_issues']
        overrun = max(x['overrun_seconds'] for r in issues for x in r['issues'])
        warnings.append(f'Final report: {len(issues)} of {final_report["recordings"]} interview outputs exceed the clip boundary (maximum {overrun:.2f}s). All predicted words remain in WER; DER uses the original clip interval. Strict output validity failed for that clip.')
    selection = read(DATA / 'selection-summary.json', {})
    if not continuation and selection.get('train_hours', 500) < 450 and not training_complete:
        warnings.append('The active run uses 188.17 hours selected by the old restrictive filters. Recovery is tracked separately below.')
    return {'generated_at': now, 'run': active.get('run_id','qda-v6-r64-full' if continuation else 'qda-v4-r64-full'), 'continuation':continuation,'parent_audio_hours':active.get('parent_audio_hours',0),'parent_report':parent_report,'calibration':read(FULL/'result/history.json') if continuation else None, 'config': config,
            'awaiting_approval':awaiting_approval,'prepared_package':read(FULL/'prepared.json'),
            'steps': steps, 'validation': sorted(validation.values(), key=lambda e: (e['step'], e.get('full_panel', False))),
            'latest': latest, 'checkpoint': read(run / 'latest.json'),
            'wandb': read(run / 'wandb-status.json'), 'completed': read(run / 'completed.json'),
            'report_complete': report_complete, 'final_report_status': final_report.get('status'),
            'schedule': read(FULL / 'schedule-preview.json'), 'evaluations': evals,
            'baseline_markers': read(FULL / 'markers/vanilla/completed-v3.json') or read(ROOT/'experiments/qda-full-r64-v4/markers/vanilla/completed-v3.json'),
            'health': health, 'experiment': stage, 'live_sync': live,
            'backup_sync': read(FULL / 'sync-status.json'), 'backups': receipts,
            'telemetry': read(FULL / 'live-telemetry.json'),
            'recovery_status': read(FULL / 'recovery-status.json'), 'warnings': warnings,
            'progress': {'audio_hours': hours, 'total_audio_hours': config.get('data_hours'),
                         'step': latest.get('step', 0), 'total_steps': config.get('steps'),
                         'recent_realtime': realtime, 'next_evaluation_hours': next_eval,
                         'eta_seconds': (max(0, (config.get('data_hours') or 0) - hours) * 3600 / realtime) if realtime else None,
                         'next_evaluation_seconds': ((next_eval - hours) * 3600 / realtime) if realtime else None,
                         'eta_note': 'Projection from the last 80 unique updates; excludes future evaluations, checkpoint copying and interruptions. Length mix changes throughput.'},
            'dataset': {'selection': selection, 'structural': read(DATA / 'structural-audit.json'),
                        'recovered': read(RECOVERY / 'result.json'),
                        'remote_copy': read(ROOT / 'datasets/releases/qda-recovered-v5/copy-verification.json'),
                        'independent_verification': read(RECOVERY / 'independent-verification.json'),
                        'recovery_progress': read(RECOVERY / 'progress.json'),
                        'recovery_policy': read(RECOVERY / 'policy.json')}}
