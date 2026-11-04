"""Audit new and supplemental paid pairing, split isolation, and the exact CPU plan."""
import argparse, collections, json, pathlib, sys
from contracts import sha, atomic_json

def audit(root, original, plan_path=None):
    read = lambda p: json.loads(p.read_text())
    selected = read(root / 'selection.json')
    supplement = read(root / 'evaluation-supplement.json')['records']
    sources = [json.loads(line) for line in (root / 'training/sources.jsonl').read_text().splitlines()]
    old = [json.loads(line) for line in (original / 'training/sources.jsonl').read_text().splitlines()]
    old_selection = read(original / 'selection.json')
    old_ids = {r['record_id'] for r in old_selection}
    old_hashes = {r['sha256'] for r in old_selection}
    assert not {r['record_id'] for r in selected} & old_ids
    assert not {r['sha256'] for r in selected} & old_hashes
    old_by_id = {r['record_id']: r for r in old}
    original_splits = {r['show']: r['split'] for r in old_selection}
    for r in selected:
        assert r['split'] == original_splits.get(r['show'], r['split'])
    for r in supplement:
        assert r['split'] == old_by_id[r['record_id']]['split'] == 'validation'
    all_rows = {r['record_id']: r for r in selected + supplement}
    assert len(all_rows) == len(selected) + len(supplement)
    for rid, row in all_rows.items():
        folder = root / 'records' / rid
        receipt, state = read(folder / 'receipt.json'), read(folder / 'state.json')
        assert receipt['record_id'] == rid and receipt['audio_sha256'] == row['sha256']
        assert receipt['transcription_id'] == state['transcription_id']
        assert state['phase'] == 'done' and state['remote_deleted']
        assert sha(folder / 'receipt.json') == state['receipt_sha256']
        for filename, digest in receipt['artifacts_sha256'].items():
            assert sha(folder / filename) == digest, (rid, filename)
        assert sha(root / row['path']) == row['sha256'], rid
        assert read(folder / 'soniox.raw.json')['id'] == state['transcription_id']
    hours, counts = collections.Counter(), collections.Counter()
    for r in sources:
        row = all_rows[r['record_id']]
        label = read(root / r['label_path'])
        assert r['split'] == row['split'] == label['split']
        assert r['audio_sha256'] == row['sha256'] == label['source_original_audio_sha256']
        assert r['raw_transcript_sha256'] == sha(root / r['raw_transcript_path']) == label['source_raw_sha256']
        assert label['source_transcription_id'] == read(root / 'records' / r['record_id'] / 'receipt.json')['transcription_id']
        assert sha(root / r['label_path']) == r['label_sha256']
        assert r['split'] != 'test'
        key = '/'.join((r['origin'], r['split'], r['language']))
        hours[key] += r['audio_duration'] / 3600
        counts[key] += 1
    train = {r['record_id'] for r in sources if r['split'] == 'train'}
    validation = {r['record_id'] for r in sources if r['split'] == 'validation'}
    assert not train & validation
    assert not train & old_ids
    assert not validation & {r['record_id'] for r in old if r['split'] == 'train'}
    excluded = {r['record_id'] for r in read(root / 'training-exclusions.json')}
    assert set(all_rows) == train | validation | excluded
    report = {'status': 'passed', 'paid_new_records': len(selected), 'supplemental_validation_records': len(supplement),
              'paired_records_checked': len(all_rows), 'eligible_records': len(sources), 'excluded_records': len(excluded),
              'hours': dict(hours), 'counts': dict(counts), 'manifest_sha256': sha(root / 'training/sources.jsonl')}
    if plan_path:
        import torch
        plan = read(plan_path)
        assert plan['dataset_sha256'] == report['manifest_sha256']
        assert plan['planning_sha256'] == sha(plan_path.parent / 'planning.py')
        cache = plan_path.parent / 'cache'
        assert sha(cache / 'contract.json') == plan['cache_contract_sha256']
        training_rows = [r for pair in plan['all_training_pairs'] for r in pair]
        assert len({r['window_id'] for r in training_rows}) == len(training_rows)
        assert {r['record_id'] for r in training_rows} == train
        by_source = collections.defaultdict(list)
        for r in training_rows: by_source[r['record_id']].append(r)
        for rid, windows in by_source.items():
            windows.sort(key=lambda r: r['source_start'])
            assert abs(windows[0]['source_start']) < .01
            assert all(abs(a['source_end'] - b['source_start']) < .01 for a,b in zip(windows, windows[1:]))
            source = next(r for r in sources if r['record_id'] == rid)
            assert abs(windows[-1]['source_end'] - source['audio_duration']) < .05
        probe_rows = [r for pair in plan['probe_pairs'] for r in pair]
        from planning import arrange
        _, expected_probe = arrange(training_rows, plan['seed'])
        assert expected_probe == plan['probe_pairs'], 'Plan does not reproduce deterministically'
        assert {r['window_id'] for r in probe_rows} <= {r['window_id'] for r in training_rows}
        assert {r['window_id'] for pair in plan['calibration_pairs'] for r in pair} <= {r['window_id'] for r in probe_rows}
        rows = {r['window_id']:r for r in training_rows + plan['validation'] + plan['long_validation']}
        for r in rows.values():
            assert r['total_tokens'] <= 65536 and r['target_tokens'] > 0 and r['content_tokens'] > 0
            if r['split'] == 'validation': assert r['record_id'] in validation
            ip = cache / (r['window_id'] + '.input.pt')
            assert sha(ip) == r['input_sha256']
            x = torch.load(ip, weights_only=True, map_location='cpu')
            assert x['ids'].shape == x['labels'].shape == x['mask'].shape == x['content_mask'].shape
            assert len(x['ids']) == r['total_tokens'] and int(x['mask'].sum()) == r['speech_features']
            assert int((x['labels'] != -100).sum()) == r['target_tokens']
            assert int(x['content_mask'].sum()) == r['content_tokens']
            assert not bool((x['content_mask'] & (x['labels'] == -100)).any())
        assert {r['language'] for r in plan['validation']} == {'cs','sk'}
        assert {r['language'] for r in plan['long_validation']} == {'cs','sk'}
        report['plan'] = {'sha256': sha(plan_path), 'verified_windows': len(rows), 'probe_audio_hours': plan['probe_audio_hours'],
                          'paired_updates': len(plan['probe_pairs']), 'validation_windows': len(plan['validation']),
                          'long_sessions': len(plan['long_validation']), 'nominal_length_hours':plan['probe_length_hours'],
                          'actual_duration_hours':plan['probe_duration_hours']}
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=pathlib.Path);p.add_argument('original',type=pathlib.Path)
    p.add_argument('--plan',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    result=audit(a.root,a.original,a.plan);atomic_json(a.out,result);print(json.dumps(result,indent=2))
