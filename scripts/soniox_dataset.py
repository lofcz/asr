#!/usr/bin/env python3
"""Read zero-copy window candidates or explicitly export an approved upstream sample."""
import argparse, json, pathlib, sqlite3, subprocess, tempfile, os, math
from soniox_durable import atomic, encode, file_hash, digest
from soniox_postprocess import candidate_label

def load_window(dataset,window_id,allow_unapproved=False):
    dataset=pathlib.Path(dataset);db=sqlite3.connect(f'file:{dataset}/postprocess/v1/catalog.sqlite3?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:row=db.execute('SELECT w.*,r.report_path,r.status FROM windows w JOIN recordings r USING(record_id) WHERE window_id=?',(window_id,)).fetchone()
    finally:db.close()
    if row is None:raise KeyError('Unknown window ID')
    if not row['training_ready'] and not allow_unapproved:raise ValueError('Window is not approved for training; explicit candidate inspection is required')
    rel=pathlib.Path(row['report_path']).parent;receipt=json.loads((dataset/rel/'receipt.json').read_bytes())
    if receipt['record_id']!=row['record_id'] or receipt['revision']!=row['revision']:raise ValueError('Window/record revision mismatch')
    for name,sha in receipt['artifacts_sha256'].items():
        if file_hash(dataset/rel/name)!=sha:raise ValueError('Derived artifact checksum mismatch: '+name)
    report=json.loads((dataset/rel/'report.json').read_bytes());segments=json.loads((dataset/rel/'segments.json').read_bytes())['segments'];w=json.loads(row['window_json'])
    if w not in json.loads((dataset/rel/'windows.json').read_bytes()):raise ValueError('Catalog window differs from checksummed source')
    if bool(row['training_ready'])!=report['training_ready']:raise ValueError('Catalog approval differs from source report')
    return {'window_id':window_id,'record_id':row['record_id'],'revision':row['revision'],'split':row['split'],'training_ready':bool(row['training_ready']),'audio_source':{'path':report['source_audio_path'],'sha256':report['audio_sha256'],'start':w['source_start'],'end':w['source_end']},'label':candidate_label(segments,w,report['context']['customized_context']),'curriculum_views':w['curriculum_views']}

def export_window(root,dataset,window_id,out):
    sample=load_window(dataset,window_id);source=pathlib.Path(root)/sample['audio_source']['path'];out=pathlib.Path(out)
    if file_hash(source)!=sample['audio_source']['sha256']:raise ValueError('Source audio identity mismatch')
    if out.exists() and any(out.iterdir()):raise ValueError('Export destination must be empty')
    out.mkdir(parents=True,exist_ok=True);start=sample['audio_source']['start'];duration=sample['label']['audio_duration']
    fd,temp=tempfile.mkstemp(dir=out,suffix='.flac');os.close(fd)
    try:
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-ss',str(start),'-i',str(source),'-t',str(duration),'-vn','-ar','24000','-ac','1','-sample_fmt','s16','-c:a','flac',temp],check=True)
        actual=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',temp],text=True))
        if abs(actual-duration)>.02:raise ValueError('Exported audio duration mismatch')
        with open(temp,'rb') as f:os.fsync(f.fileno())
        os.replace(temp,out/'audio.flac')
        label={**sample['label'],'audio_path':'audio.flac'};atomic(out/'sample.json',encode(label))
        atomic(out/'receipt.json',encode({'window_id':window_id,'record_id':sample['record_id'],'revision':sample['revision'],'split':sample['split'],'source':sample['audio_source'],'audio_sha256':file_hash(out/'audio.flac'),'label_sha256':file_hash(out/'sample.json'),'decoded_duration':actual}))
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return sample

def review_template(dataset,record_id):
    dataset=pathlib.Path(dataset);current=json.loads((dataset/'postprocess/v1/current'/f'{record_id}.json').read_bytes());report=json.loads((dataset/current['path']/'report.json').read_bytes())
    return {'record_id':record_id,'input_digest':report['base_input_digest'],'reviewed_by':'','reviewed_at':None,'speech_verified':False,'diarization_verified':False,'acoustic_events_verified':False,'token_overrides':[],'accepted_events':[],'customized_context':[],'context_provenance':None,'context_available_before_transcription':False,'notes':'Copy to dataset/reviews/<record_id>.json only after reviewing. Each token override needs token_index, original_sha256, reason, changes. Accepted speakerless events need start/end seconds, one of the six event tags as text, and evidence. Music/noise can overlap speech. No reference-derived context.'}

def iter_review_tasks(dataset,kind):
    dataset=pathlib.Path(dataset);db=sqlite3.connect(f'file:{dataset}/postprocess/v1/catalog.sqlite3?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:records=[dict(r) for r in db.execute("SELECT * FROM recordings WHERE status IN ('needs_review','approved')")]
    finally:db.close()
    for row in records:
        report_path=dataset/row['report_path'];receipt=json.loads((report_path.parent/'receipt.json').read_bytes())
        if file_hash(report_path)!=receipt['artifacts_sha256']['report.json']:raise ValueError('Report checksum mismatch')
        report=json.loads(report_path.read_bytes());duration=report['duration']
        if kind=='acoustic-full-timeline':
            intervals=((float(start),min(duration,start+10.)) for start in range(0,math.ceil(duration),10))
        else:
            intervals=((max(0,p['start']-2),min(duration,p['end']+2)) for p in report['issues'] if p['kind']=='possible_missing_speech')
        for a,b in intervals:
            yield {'task_id':digest(f"{row['record_id']}|{row['revision']}|{kind}|{a}|{b}".encode()),'record_id':row['record_id'],'revision':row['revision'],'base_input_digest':report['base_input_digest'],'split':row['split'],'kind':kind,'audio_path':report['source_audio_path'],'audio_sha256':report['audio_sha256'],'source_start':a,'source_end':b,'status':'not_run','output_policy':'Proposals only. No automatic transcript replacement or event acceptance. Full timeline includes speech, allowing background music/noise evidence.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('inspect');q.add_argument('window_id');q.add_argument('--allow-unapproved',action='store_true')
    q=sub.add_parser('export');q.add_argument('window_id');q.add_argument('--root',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('review-template');q.add_argument('record_id');q.add_argument('--out',required=True)
    q=sub.add_parser('review-tasks');q.add_argument('--kind',required=True,choices=['acoustic-full-timeline','missing-speech']);q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='inspect':
        sample=load_window(a.dataset,a.window_id,a.allow_unapproved);print(json.dumps({k:v for k,v in sample.items() if k!='label'},ensure_ascii=False,indent=2));print('segments:',len(sample['label']['segments']))
    elif a.command=='export':export_window(a.root,a.dataset,a.window_id,a.out);print('Exported paired approved sample')
    elif a.command=='review-tasks':
        rows=list(iter_review_tasks(a.dataset,a.kind));atomic(pathlib.Path(a.out),b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in rows));print('Exported',len(rows),'unexecuted review tasks')
    else:
        dest=pathlib.Path(a.out)
        if dest.exists():raise ValueError('Refusing to overwrite an existing review')
        atomic(dest,encode(review_template(a.dataset,a.record_id)));print('Wrote unapproved review template')
if __name__=='__main__':main()
