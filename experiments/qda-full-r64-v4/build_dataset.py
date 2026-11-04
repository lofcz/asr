"""Revisioned derived labels. Paid source files and previous releases are read-only."""
import argparse
import collections
import json
import pathlib
import time
from contracts import atomic_json, sha
from targets import reconstruct, independent_verify, VERSION

HERE = pathlib.Path(__file__).resolve().parent


def build(parent, output, reserved):
    output.mkdir(parents=True,exist_ok=True)
    (output/'.building').write_text('Derived dataset is rebuilding; admission forbidden.\n')
    policy = json.loads((HERE / 'policy.json').read_text())
    input_rows = {}
    manifests = {}
    for release in ('qda-496h-v1', 'qda-50h-v2'):
        root = parent / release
        manifest = root / 'training/sources.jsonl'
        manifests[release] = sha(manifest)
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            rid = row['record_id']
            if rid in input_rows:
                assert input_rows[rid]['audio_sha256'] == row['audio_sha256']
                assert input_rows[rid]['split'] == row['split']
                continue
            input_rows[rid] = dict(row, origin=release)
    reserved_ids = set(json.loads(reserved.read_text())['record_ids'])
    heldout_shows = {r['show'] for r in input_rows.values() if r['split'] != 'train' or r['record_id'] in reserved_ids}
    heldout_hashes = {r['audio_sha256'] for r in input_rows.values() if r['show'] in heldout_shows}
    rows, excluded = [], []
    old_defects, repaired_gaps = 0, 0
    for rid, source in sorted(input_rows.items()):
        root = parent / source['origin']
        try:
            audio = root / source['audio_path']
            rawpath = root / source['raw_transcript_path']
            record = root / 'records' / rid
            rawhash = sha(rawpath)
            assert rawhash == source['raw_transcript_sha256'], 'raw transcript identity mismatch'
            receipt = json.loads((record / 'receipt.json').read_text())
            assert receipt['record_id'] == rid
            assert receipt['audio_sha256'] == source.get('original_audio_sha256', source['audio_sha256'])
            assert receipt['artifacts_sha256']['soniox.raw.json'] == rawhash
            assert sha(audio) == source['audio_sha256'], 'audio identity mismatch'
            raw = json.loads(rawpath.read_text())
            assert raw['id'] == receipt['transcription_id'], 'Soniox/audio pairing mismatch'
            oldpath = root / source['label_path']
            assert sha(oldpath) == source['label_sha256']
            old = json.loads(oldpath.read_text())
            assert old['source_transcription_id'] == raw['id'] and old['source_raw_sha256'] == rawhash
            boundaries = sum(a['text'][-1:].isalnum() and b['text'][:1].isalnum() for a,b in zip(old['segments'],old['segments'][1:]))
            old_defects += boundaries
            segments, words, repairs = reconstruct(raw['tokens'], source['audio_duration'], local_quarantine=True)
            audit = independent_verify(raw['tokens'], segments, source['audio_duration'])
            split = 'train' if source['show'] not in heldout_shows else 'confirmation' if source['split'] == 'test' else 'development'
            if split == 'train':
                assert rid not in reserved_ids and source['audio_sha256'] not in heldout_hashes
            lang = collections.Counter(t.get('language') for t in raw['tokens'] if t.get('text','').strip() and t.get('text','') not in ('<end>', '<fin>'))
            confidences = [t['confidence'] for t in raw['tokens'] if t.get('text','').strip() and t.get('text','') not in ('<end>', '<fin>')]
            bad_indices = {i for w in words if w.get('quarantine') for i in w['token_indices']}
            label = {'version': VERSION, 'record_id': rid, 'split': split, 'duration': source['audio_duration'],
                     'audio_sha256': source['audio_sha256'], 'raw_sha256': rawhash, 'segments': segments,
                     'words': words, 'repairs': repairs, 'customized_context': [],
                     'quarantined_regions': [dict(start=s['start'],end=s['end'],reason='ambiguous_intraword_gap',
                         token_indices=s['token_indices']) for s in segments
                         if bad_indices.intersection(s['token_indices'])],
                     'token_confidences': [t.get('confidence') for t in raw['tokens']],
                     'quality': 'unreviewed Soniox pseudo-labels; structural reconstruction only',
                     'independent_word_audit': audit}
            path = output / 'labels' / (rid + '.json')
            atomic_json(path, label)
            row = dict(source, split=split, original_split=source['split'], label_path=str(path.relative_to(output)),
                       label_sha256=sha(path), raw_path=str(rawpath), audio_file=str(audio),
                       receipt_sha256=sha(record/'receipt.json'), vad_path=str(record/'vad.json'),
                       old_fragment_boundaries=boundaries, repaired_intraword_gaps=len(repairs),
                       target_language_fraction=lang[source['language']]/max(1,sum(lang.values())),
                       low_confidence_fraction=sum(c < policy['low_confidence_threshold'] for c in confidences)/len(confidences),
                       speaker_count=len({s['speaker'] for s in segments}))
            rows.append(row)
            repaired_gaps += len(repairs)
        except (AssertionError, ValueError, KeyError, OSError) as error:
            excluded.append({'record_id': rid, 'origin': source['origin'], 'reason': str(error), 'hours': source['audio_duration']/3600})
        if (len(rows)+len(excluded)) % 50 == 0:
            print(json.dumps({'checked':len(rows)+len(excluded),'accepted_structural':len(rows),'quarantined':len(excluded)}),flush=True)
    atomic_json(output/'sources.json', rows)
    atomic_json(output/'structural-quarantine.json', excluded)
    report = {'status':'complete', 'policy_sha256':sha(HERE/'policy.json'), 'target_code_sha256':sha(HERE/'targets.py'),
              'builder_sha256':sha(pathlib.Path(__file__)), 'source_manifests':manifests,
              'sources_sha256':sha(output/'sources.json'), 'source_records':len(input_rows),'structurally_valid_records':len(rows),
              'quarantined_records':len(excluded),'quarantine_reasons':dict(collections.Counter(r['reason'].split(' at token')[0] for r in excluded)),
              'old_fragment_boundaries':old_defects, 'repaired_intraword_gaps':repaired_gaps,
              'reserved_record_ids':sorted(reserved_ids), 'heldout_shows':sorted(heldout_shows),
              'hours_by_split':{s:sum(r['audio_duration'] for r in rows if r['split']==s)/3600 for s in ('train','development','confirmation')},
              'training_authorized':False, 'remaining':['acoustic audit','exact token/cache audit'],
              'completed_at':time.time()}
    atomic_json(output/'structural-audit.json',report)
    (output/'.building').unlink()
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--reserved',type=pathlib.Path,required=True);a=p.parse_args();build(a.parent,a.output,a.reserved)
