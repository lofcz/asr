"""Validate split provenance, timing, speaker mapping, and cached sequence limits."""
import hashlib, json, pathlib

BASE = pathlib.Path(__file__).resolve().parent
rows = [json.loads(line) for line in (BASE/'dataset/manifest.jsonl').read_text().splitlines()]
sources = json.loads((BASE/'sample/sources.json').read_text())
assert len({r['id'] for r in rows}) == len(rows)
for key in ('source_id', 'show_id'):
    assert not ({r[key] for r in rows if r['split'] == 'train'} &
                {r[key] for r in rows if r['split'] == 'validation'})
for row in rows:
    assert (BASE/'dataset'/row['audio_path']).is_file()
    assert row['segments']
    mapping = {}
    for segment in row['segments']:
        assert 0 <= segment['start'] <= segment['end'] <= row['audio_duration'] + .002
        assert segment['text'].strip()
        mapping.setdefault(segment['session_speaker'], segment['speaker'])
        assert mapping[segment['session_speaker']] == segment['speaker']
    assert len(set(mapping.values())) == len(mapping) == row['speaker_count']
result = {'rows': len(rows), 'unique_training_hours': sum(s['duration'] for s in sources if s['split']=='train')/3600,
          'unique_validation_minutes': sum(s['duration'] for s in sources if s['split']=='validation')/60,
          'source_and_show_split_disjoint': True, 'timestamps_and_speaker_mapping_valid': True}
if (BASE/'cache/index.jsonl').exists():
    import numpy as np
    assert not (BASE/'cache/.building').exists()
    meta = json.loads((BASE/'cache/metadata.json').read_text())
    assert meta['manifest_sha256'] == hashlib.sha256((BASE/'dataset/manifest.jsonl').read_bytes()).hexdigest()
    cached = [json.loads(line) for line in (BASE/'cache/index.jsonl').read_text().splitlines()]
    assert {r['id'] for r in cached} == {r['id'] for r in rows}
    arrays = {k: np.load(BASE/'cache'/f'{k}.npy', mmap_mode='r') for k in ('tokens','labels','mask','features')}
    for row in cached:
        start, length, prompt = row['token_offset'], row['token_length'], row['prompt_length']
        labels = arrays['labels'][start:start+length]
        mask = arrays['mask'][start:start+length]
        assert length <= 65536
        assert (labels[:prompt] == -100).all() and (labels[prompt:] >= 0).all()
        assert mask.sum() == row['feature_length'] and not mask[prompt:].any()
        assert (labels[prompt:] == arrays['tokens'][start+prompt:start+length]).all()
        features = arrays['features'][row['feature_offset']:row['feature_offset']+row['feature_length']]
        assert np.isfinite(features).all()
    result.update(cache_manifest_matches=True, maximum_training_tokens=max(r['token_length'] for r in cached),
                  cached_feature_bytes=arrays['features'].nbytes)
(BASE/'data-validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
