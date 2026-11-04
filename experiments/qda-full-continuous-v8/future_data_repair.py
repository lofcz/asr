"""One audited correction to an unseen row; the approved base plan stays immutable."""
import copy
import hashlib
import json
from pathlib import Path

from contracts import sha

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError('Future data repair: ' + message)


def load_repair():
    path = HERE / 'runtime-repair.json'
    if not path.exists():
        return None
    repair = json.loads(path.read_text())
    from runtime_state import runtime_hashes
    current = runtime_hashes()
    require(current == repair['new_runtime_sha256'], 'executable hashes changed')
    old = repair['old_config']['runtime_sha256']
    changed = {k for k in set(old) | set(current) if old.get(k) != current.get(k)}
    require(changed == {str(HERE / name) for name in (
        'train_restart.py', 'runtime_state.py', 'disk_retention.py', 'future_data_repair.py')},
        'unexpected executable change')
    require(sha(HERE / 'plan.json') == repair['old_config']['plan_sha256'], 'base plan changed')
    require(sha(HERE / 'run-spec.json') == repair['old_config']['run_spec_sha256'], 'LR/run specification changed')
    require(sha(HERE / 'cache/contract.json') == repair['cache_contract_sha256'], 'cache contract changed')
    amendment = Path(repair['amendment_path'])
    require(sha(amendment) == repair['amendment_sha256'], 'amendment changed')
    value = json.loads(amendment.read_text())
    require(value['base_plan_sha256'] == repair['old_config']['plan_sha256'], 'wrong base plan')
    require(value['target_token_delta'] == repair['target_token_delta'], 'wrong token delta')
    require(sha(Path(value['corrected_label_file'])) == value['replacement']['label_sha256'], 'corrected label changed')
    for name, expected in value['evidence_sha256'].items():
        require(sha(amendment.parent / name) == expected, 'alignment evidence changed: ' + name)
    require(0 < repair['checkpoint_step'] < value['update'], 'correction touches consumed data')
    return repair


def validate_replacement(plan, value):
    """Accept exactly one appended, audio-aligned turn, never a reordered curriculum."""
    require(value['version'] == 1, 'unknown amendment version')
    index, rank = value['update'] - 1, value['rank']
    old = plan['probe_pairs'][index][rank]
    new = value['replacement']
    require(digest(old) == value['original_row_sha256'], 'original row differs')
    require(old['window_id'] == value['original_window_id'], 'wrong window')
    require(old['split'] == new['split'] == 'train', 'held-out row changed')
    allowed = {'target', 'target_tokens', 'total_tokens', 'content_tokens', 'target_json_sha256',
               'label_sha256', 'label_path', 'window_id', 'reuse_root', 'amended_from_window_id'}
    require({k: v for k, v in old.items() if k not in allowed} ==
            {k: v for k, v in new.items() if k not in allowed}, 'audio/crop/context identity changed')
    require('reuse_root' not in new, 'stale target cache reuse enabled')
    require(new['amended_from_window_id'] == old['window_id'], 'missing original identity')
    require(new['window_id'] not in {r['window_id'] for p in plan['probe_pairs'] for r in p}, 'cache identity collision')
    require(new['target'][:-1] == old['target'], 'existing transcript or speakers changed')
    turn = new['target'][-1]
    require(turn == value['restored_turn'], 'restored turn differs')
    require(turn['Content'] == ' Tak to pro mě bylo.' and len(turn['Content'].split()) == 5, 'wrong restored words')
    require(0 <= turn['Start'] < turn['End'] <= old['duration'], 'restored words outside audio')
    require(turn['Start'] >= old['target'][-1]['Start'], 'turn order changed')
    require(turn['Speaker'] == 1 and new['target'][-2]['Speaker'] == 0, 'speaker identity changed')
    delta = new['target_tokens'] - old['target_tokens']
    require(delta == value['target_token_delta'] > 0, 'incorrect supervised token count')
    require(new['total_tokens'] - old['total_tokens'] == delta and new['total_tokens'] <= 65536,
            'prefix or context budget changed')
    label = json.loads(Path(value['corrected_label_file']).read_text())
    from dataset import window_segments
    expected = window_segments(label['segments'], old['source_start'], old['source_end'], old['audio_duration'])
    require(new['target'] == expected, 'target does not match corrected source timestamps/speakers')
    return index, rank


def apply_repair(plan, repair):
    if repair is None:
        return plan
    value = json.loads(Path(repair['amendment_path']).read_text())
    index, rank = validate_replacement(plan, value)
    effective = copy.deepcopy(plan)
    effective['probe_pairs'][index][rank] = value['replacement']
    effective['probe_target_tokens'] += value['target_token_delta']
    return effective


def compatible_resume(saved, current, repair=None):
    if saved == current:
        return True
    if repair is None or saved != repair['old_config']:
        return False
    expected = dict(saved)
    expected.update(runtime_sha256=repair['new_runtime_sha256'],
                    code_sha256=repair['new_runtime_sha256'][str(HERE / 'train_restart.py')],
                    runtime_repair_sha256=sha(HERE / 'runtime-repair.json'),
                    target_tokens=saved['target_tokens'] + repair['target_token_delta'])
    return current == expected


def validate_resume_step(step, saved, repair):
    if repair is not None and saved == repair['old_config']:
        value = json.loads(Path(repair['amendment_path']).read_text())
        require(step < value['update'], 'old supervision already consumed corrected example')

