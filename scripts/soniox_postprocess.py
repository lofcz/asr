#!/usr/bin/env python3
"""Incremental, revisioned quality checks and zero-copy training window candidates.
Raw Soniox evidence is immutable. No paid API calls; no automatic event labels.
"""
import argparse, bisect, collections, copy, fcntl, hashlib, json, logging, math, os, pathlib, re, signal, sqlite3, tempfile, threading, time
from soniox_durable import Store, encode, digest, file_hash, verify_done
VERSION='qda-postprocess-v1.1'
EVENTS={'[Unintelligible Speech]','[Music]','[Human Sounds]','[Environmental Sounds]','[Noise]','[Silence]'}
CONTROLS={'<end>','<fin>'}
STOP=threading.Event()

def union(items):
    out=[]
    for a,b in sorted(items):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out

def difference(left,right,minimum=0):
    out=[];j=0
    for a,b in left:
        while j<len(right) and right[j][1]<=a:j+=1
        pos=a;k=j
        while k<len(right) and right[k][0]<b:
            c,d=right[k]
            if c-pos>0 and c-pos>=minimum:out.append([pos,min(c,b)])
            pos=max(pos,d)
            if pos>=b:break
            k+=1
        if b-pos>0 and b-pos>=minimum:out.append([pos,b])
    return out

def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def issue(kind,severity='review',start=None,end=None,**data):
    return {'kind':kind,'severity':severity,'start':start,'end':end,**data}

def apply_review(raw,review,base_digest,duration):
    tokens=copy.deepcopy(raw['tokens']);problems=[];events=[];context=[]
    if not review:return tokens,events,context,False,problems
    if review.get('input_digest')!=base_digest:
        return tokens,events,context,False,[issue('stale_review','blocking')]
    if not isinstance(review.get('reviewed_by'),str) or not review['reviewed_by'].strip() or not review.get('reviewed_at'):
        return tokens,events,context,False,[issue('review_provenance_missing','blocking')]
    seen=set()
    for patch in review.get('token_overrides',[]):
        i=patch.get('token_index');changes=patch.get('changes',{})
        if not isinstance(i,int) or isinstance(i,bool) or not 0<=i<len(tokens) or i in seen or patch.get('original_sha256')!=digest(encode(raw['tokens'][i])) or not patch.get('reason') or not isinstance(changes,dict) or set(changes)-{'text','speaker','start_ms','end_ms'}:
            problems.append(issue('invalid_token_override','blocking'));continue
        tokens[i].update(changes);seen.add(i)
    for e in review.get('accepted_events',[]):
        a,b=e.get('start'),e.get('end')
        if e.get('text') not in EVENTS or not finite(a) or not finite(b) or not 0<=a<b<=duration or not e.get('evidence') or e.get('speaker') is not None:
            problems.append(issue('invalid_event_annotation','blocking'));continue
        events.append({'start':a,'end':b,'text':e['text'],'speaker':None,'session_speaker':None,'is_event':True,'token_indices':[],'provenance':e['evidence']})
    supplied=review.get('customized_context')
    if supplied:
        if not isinstance(supplied,list) or not all(isinstance(t,str) and t.strip() for t in supplied) or not review.get('context_provenance') or review.get('context_available_before_transcription') is not True:
            problems.append(issue('invalid_context_provenance','blocking'))
        else:context=supplied
    approved=all(review.get(k) is True for k in ('speech_verified','diarization_verified','acoustic_events_verified')) and not problems
    return tokens,events,context,approved,problems

def normalize(tokens,duration):
    segments=[];problems=[];mapping={};discarded=[];valid=[];confidence=[];languages=collections.Counter();unknown=0;previous_start=-1;timing_reversals=0
    for i,t in enumerate(tokens):
        text=t.get('text')
        if not isinstance(text,str):problems.append(issue('invalid_token_text','blocking',token_index=i));continue
        if text.strip() in CONTROLS:
            discarded.append({'token_index':i,'reason':'endpoint_control'});continue
        if not text:
            discarded.append({'token_index':i,'reason':'empty_token'});continue
        a,b=t.get('start_ms'),t.get('end_ms')
        if not finite(a) or not finite(b) or not 0<=a<=b<=duration*1000+1:
            problems.append(issue('invalid_or_untimed_content','blocking',token_index=i));continue
        a/=1000;b/=1000
        if a<previous_start:timing_reversals+=1
        previous_start=a
        event=bool(t.get('is_audio_event'));sid=t.get('speaker')
        if sid is not None and (not isinstance(sid,(str,int)) or isinstance(sid,bool)):
            problems.append(issue('invalid_speaker_type','blocking',a,b,token_index=i));sid=None
        if sid is not None:sid=str(sid)
        if not event and text.strip() and sid is None:unknown+=1
        speaker=mapping.setdefault(sid,len(mapping)) if sid is not None else None
        if event:problems.append(issue('unreviewed_soniox_audio_event','review',a,b,token_index=i))
        if not event:
            if finite(t.get('confidence')) and 0<=t['confidence']<=1:confidence.append(t['confidence'])
            elif 'confidence' in t:problems.append(issue('invalid_confidence','review',a,b,token_index=i))
            if t.get('language'):languages[str(t['language'])]+=len(text.strip())
        prev=segments[-1] if segments else None
        new_word=text[0].isspace() or prev is None or prev['text'][-1:].isspace()
        can_merge=prev and prev['session_speaker']==sid and prev['is_event']==event and not event and a>=prev['start'] and a-prev['end']<=.8
        split_for_length=prev and new_word and (b-prev['start']>=30 or b-prev['start']>=8 and prev['text'].rstrip().endswith(('.','?','!')))
        if can_merge and not split_for_length:
            prev['text']+=text;prev['end']=max(prev['end'],b);prev['token_indices'].append(i)
        else:
            if prev and not new_word and not event and prev['session_speaker']!=sid and text[0].isalnum():problems.append(issue('speaker_change_inside_word','review',a,b,token_index=i))
            segments.append({'start':a,'end':b,'speaker':speaker,'session_speaker':sid,'text':text,'is_event':event,'token_indices':[i]})
        valid.append({'index':i,'start':a,'end':b,'text':text,'speaker':sid,'event':event,'confidence':t.get('confidence'),'language':t.get('language')})
    if unknown:problems.append(issue('speakerless_speech','blocking',count=unknown))
    if timing_reversals:problems.append(issue('non_monotonic_tokens','review',count=timing_reversals,note='May reflect legitimate overlapping speech; do not sort/rewrite raw tokens.'))
    # Every retained character is represented exactly once before display/target formatting.
    expected=''.join(t['text'] for t in valid)
    assert ''.join(s['text'] for s in segments)==expected
    metrics={'speech_token_count':sum(not t['event'] and bool(t['text'].strip()) for t in valid),'speaker_count':len(mapping),'session_speaker_map':mapping,'mean_token_confidence':sum(confidence)/len(confidence) if confidence else None,'low_confidence_token_fraction':sum(c<.65 for c in confidence)/len(confidence) if confidence else None,'language_character_counts':dict(languages),'speakerless_speech_tokens':unknown,'retained_token_count':len(valid),'explicitly_excluded_controls':discarded}
    return segments,valid,metrics,problems

def quality(tokens,segments,metrics,duration,vad,language):
    problems=[];speech=union([[max(0,t['start']-.1),min(duration,t['end']+.1)] for t in tokens if not t['event'] and t['text'].strip()]);coverage={}
    if not metrics['speech_token_count']:problems.append(issue('no_speech','review',0,duration))
    if metrics['speaker_count']==1:problems.append(issue('single_speaker_recording','info',note='Valid auxiliary material; inspect conversation balance, do not fabricate speakers.'))
    counts=metrics['language_character_counts']
    if counts:
        dominant=max(counts,key=counts.get);metrics['dominant_language']=dominant
        if dominant!=language:problems.append(issue('metadata_language_mismatch','review',detected=dominant,metadata=language))
        metrics['non_cs_sk_character_fraction']=sum(v for k,v in counts.items() if k not in ('cs','sk'))/sum(counts.values())
    low=union([[t['start'],t['end']] for t in tokens if finite(t['confidence']) and t['confidence']<.65 and not t['event']])
    for a,b in low:
        if b-a>=.3:problems.append(issue('low_confidence','review',a,b))
    switches=sum(a['session_speaker']!=b['session_speaker'] for a,b in zip(segments,segments[1:]) if not a['is_event'] and not b['is_event'])
    metrics['speaker_switches_per_minute']=switches/max(duration/60,.01)
    if metrics['speaker_switches_per_minute']>20:problems.append(issue('frequent_speaker_switches','review',rate=metrics['speaker_switches_per_minute']))
    byspeaker=collections.defaultdict(list)
    for t in tokens:
        if t['speaker'] is not None and not t['event']:byspeaker[t['speaker']].append([t['start'],t['end']])
    events=[]
    for rows in byspeaker.values():
        for a,b in union(rows):events.extend([(a,1),(b,-1)])
    active=0;last=0;overlap=0
    for t,delta in sorted(events):
        if active>=2:overlap+=t-last
        active+=delta;last=t
    metrics['multi_speaker_overlap_seconds']=overlap
    for s in segments:
        words=re.findall(r'\w+',s['text'].casefold())
        if any(words[i:i+8]==words[i+8:i+16]==words[i+16:i+24] for i in range(max(0,len(words)-23))):problems.append(issue('repeated_long_phrase','review',s['start'],s['end'],note='Review only; QDA repetitions are not automatically deleted.'))
    if not vad or not vad.get('full_decode_verified'):
        problems.append(issue('awaiting_verified_full_decode_and_vad','blocking'))
    else:
        vs=union(vad['speech_intervals_seconds']);missing=difference(vs,speech,1.0);outside=difference(speech,vs,1.0);gaps=difference([[0,duration]],union(vs+speech),2.0)
        coverage={'vad_speech_fraction':vad['speech_fraction'],'speech_without_transcript':missing,'transcript_without_vad':outside,'non_speech_review_candidates':gaps}
        for kind,items in [('possible_missing_speech',missing),('transcript_outside_vad',outside),('non_speech_needs_classification',gaps)]:
            for a,b in items:problems.append(issue(kind,'review',a,b))
        coverage['speech_without_transcript_seconds']=sum(b-a for a,b in missing)
        coverage['transcript_without_vad_seconds']=sum(b-a for a,b in outside)
    return coverage,problems

def windows(segments,duration,record_id):
    """Partition at segment/word boundaries where NO retained segment crosses the cut.
    No arbitrary fallback cut; oversized overlap components are explicitly flagged.
    """
    ordered=sorted(segments,key=lambda s:(s['start'],s['end']))
    if not ordered:return ordered,[]
    candidates=[0.];running=0.
    spoken=[s for s in ordered if not s.get('is_event')]
    for i,s in enumerate(spoken):
        if i and s['start']>=running:candidates.append((running+s['start'])/2)
        running=max(running,s['end'])
    if not spoken:candidates.extend(float(t) for t in range(120,math.ceil(duration),120))
    candidates.append(duration);candidates=sorted(set(candidates));starts=[s['start'] for s in ordered];result={}
    for bucket in (120,600,1800,3600):
        bounds=[0.]
        while duration-bounds[-1]>bucket+30:
            target=bounds[-1]+bucket;idx=bisect.bisect_left(candidates,target)
            possible=[c for c in candidates[max(0,idx-1):idx+1] if bounds[-1]<c<duration]
            if not possible:break
            end=min(possible,key=lambda c:abs(c-target))
            if end<=bounds[-1]:break
            bounds.append(end)
        bounds.append(duration)
        for a,b in zip(bounds,bounds[1:]):
            lo=bisect.bisect_left(starts,a);hi=bisect.bisect_left(starts,b) if b<duration else len(ordered)
            event_indices=[i for i,s in enumerate(ordered) if s.get('is_event') and s['start']<b and s['end']>a]
            if hi<=lo and not event_indices:continue
            assert all(a<=s['start']<=s['end']<=b+.001 for s in ordered[lo:hi] if not s.get('is_event'))
            key=(a,b)
            if key not in result:
                sid={}
                for s in ordered[lo:hi]:
                    if s['session_speaker'] is not None:sid.setdefault(s['session_speaker'],len(sid))
                result[key]={'window_id':digest(f'{record_id}|{a:.6f}|{b:.6f}'.encode()),'source_start':a,'source_end':b,'audio_duration':b-a,'segment_lo':lo,'segment_hi':hi,'event_indices':event_indices,'session_to_window_speaker':sid,'curriculum_views':[],'estimated_audio_tokens':math.ceil((b-a)*7.5),'token_budget_status':'requires actual processor/tokenizer count before batching'}
            result[key]['curriculum_views'].append({'nominal_seconds':bucket,'oversized':b-a>bucket+30})
    return ordered,list(result.values())

def candidate_label(segments,window,context=None):
    rows=[];a=window['source_start']
    chosen=[s for s in segments[window['segment_lo']:window['segment_hi']] if not s.get('is_event')]+[segments[i] for i in window.get('event_indices',[])]
    for s in sorted(chosen,key=lambda x:(x['start'],x['end'])):
        start=max(a,s['start']) if s.get('is_event') else s['start']
        end=min(window['source_end'],s['end']) if s.get('is_event') else s['end']
        item={'start':round(start-a,6),'end':round(end-a,6),'text':s['text']}
        if s['session_speaker'] is not None:item['speaker']=window['session_to_window_speaker'][s['session_speaker']]
        elif s['text'] not in EVENTS:raise ValueError('Unresolved speaker or event cannot become training target')
        rows.append(item)
    return {'audio_duration':window['audio_duration'],'segments':rows,'customized_context':context or []}

def connect(path):
    path.parent.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(path);db.row_factory=sqlite3.Row;db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('''CREATE TABLE IF NOT EXISTS recordings(record_id TEXT PRIMARY KEY,fingerprint TEXT NOT NULL,revision TEXT NOT NULL,split TEXT NOT NULL,language TEXT NOT NULL,duration REAL NOT NULL,status TEXT NOT NULL,report_path TEXT NOT NULL,metrics_json TEXT NOT NULL,updated REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS windows(window_id TEXT PRIMARY KEY,record_id TEXT NOT NULL,split TEXT NOT NULL,duration REAL NOT NULL,source_start REAL NOT NULL,source_end REAL NOT NULL,views_json TEXT NOT NULL,revision TEXT NOT NULL,training_ready INTEGER NOT NULL,window_json TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS windows_sampling ON windows(split,training_ready,duration);
    CREATE TABLE IF NOT EXISTS review_queue(issue_id TEXT PRIMARY KEY,record_id TEXT NOT NULL,kind TEXT NOT NULL,severity TEXT NOT NULL,start REAL,end REAL,payload_json TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS review_priority ON review_queue(severity,kind,record_id);
    CREATE TABLE IF NOT EXISTS failures(record_id TEXT PRIMARY KEY,error TEXT NOT NULL,updated REAL NOT NULL);''');db.commit();return db

def update_index(db,r,fingerprint,revision,rel,report,win):
    with db:
        db.execute('INSERT OR REPLACE INTO recordings VALUES(?,?,?,?,?,?,?,?,?,?)',(r['record_id'],fingerprint,revision,r['split'],r['language'],report['duration'],report['status'],str(rel/'report.json'),json.dumps(report['metrics']),time.time()))
        db.execute('DELETE FROM windows WHERE record_id=?',(r['record_id'],));db.execute('DELETE FROM review_queue WHERE record_id=?',(r['record_id'],))
        for w in win:
            db.execute('INSERT INTO windows VALUES(?,?,?,?,?,?,?,?,?,?)',(w['window_id'],r['record_id'],r['split'],w['audio_duration'],w['source_start'],w['source_end'],json.dumps(w['curriculum_views']),revision,int(report['training_ready']),json.dumps(w)))
        for i,problem in enumerate(report['issues']):
            db.execute('INSERT INTO review_queue VALUES(?,?,?,?,?,?,?)',(digest(f"{r['record_id']}|{revision}|{i}".encode()),r['record_id'],problem['kind'],problem['severity'],problem.get('start'),problem.get('end'),json.dumps(problem,ensure_ascii=False)))

def process(store,db,r,selection_sha):
    state=store.state(r)
    if state['phase']!='done':return False
    rid=r['record_id'];record=pathlib.Path('records')/rid;folder=store.backup/record
    vad_bytes=(folder/'vad.json').read_bytes() if (folder/'vad.json').exists() else None
    error_bytes=(folder/'vad-error.json').read_bytes() if not vad_bytes and (folder/'vad-error.json').exists() else None
    review_path=store.out/'reviews'/f'{rid}.json';review_bytes=review_path.read_bytes() if review_path.exists() else None
    source=store.root/r['path'];stat=source.stat()
    evidence={'version':VERSION,'selection_sha256':selection_sha,'receipt_sha256':state['receipt_sha256'],'vad_sha256':digest(vad_bytes) if vad_bytes else None,'vad_error_sha256':digest(error_bytes) if error_bytes else None}
    base_digest=digest(encode(evidence));fingerprint=digest(encode({**evidence,'review_sha256':digest(review_bytes) if review_bytes else None,'source_size':stat.st_size,'source_mtime_ns':stat.st_mtime_ns}))
    old=db.execute('SELECT fingerprint FROM recordings WHERE record_id=?',(rid,)).fetchone()
    if old and old['fingerprint']==fingerprint:return False
    verify_done(store,r,state);rawbytes=(folder/'soniox.raw.json').read_bytes();raw=json.loads(rawbytes);vad=json.loads(vad_bytes) if vad_bytes else None;review=json.loads(review_bytes) if review_bytes else None
    problems=[]
    if file_hash(source)!=r['sha256']:problems.append(issue('current_audio_hash_mismatch','blocking'))
    if raw.get('id')!=state['transcription_id']:problems.append(issue('raw_transcription_identity_mismatch','blocking'))
    if vad and vad.get('audio_sha256')!=r['sha256']:problems.append(issue('vad_audio_identity_mismatch','blocking'));vad=None
    if vad:
        store.verified(record/'vad.probabilities.npy.gz',vad['probabilities_sha256'])
    duration=vad['decoded_seconds'] if vad else r['seconds']
    tokens,accepted,context,approved,review_issues=apply_review(raw,review,base_digest,duration);problems+=review_issues
    segments,valid,metrics,problems2=normalize(tokens,duration);problems+=problems2
    coverage,more=quality(valid,segments,metrics,duration,vad,r['language']);problems+=more
    if error_bytes:problems.append(issue('full_decode_failed','blocking',details=json.loads(error_bytes)))
    segments+=accepted
    has_blocker=any(p['severity']=='blocking' for p in problems)
    # Unreviewed raw event tags stay evidence only; reviewed accepted_events supplies target events.
    target_segments=[s for s in segments if not s['is_event'] or s.get('provenance')]
    if not has_blocker:
        ordered,win=windows(target_segments,duration,rid)
        for w in win:candidate_label(ordered,w,context)
    else:ordered=sorted(target_segments,key=lambda s:(s['start'],s['end']));win=[]
    ready=approved and not has_blocker
    if not approved:problems.append(issue('source_review_required','review',note='Speech, diarization and acoustic events need verified review; heuristics do not approve labels.'))
    status='approved' if ready else 'quarantined' if has_blocker else 'needs_review'
    revision=fingerprint[:24];rel=pathlib.Path('postprocess/v1/records')/rid/revision
    report={'schema_version':1,'pipeline_version':VERSION,'record_id':rid,'base_input_digest':base_digest,'revision':revision,'audio_sha256':r['sha256'],'source_audio_path':r['path'],'selection_sha256':selection_sha,'split':r['split'],'duration':duration,'status':status,'training_ready':ready,'metrics':metrics,'coverage':coverage,'issues':problems,'context':{'customized_context':context,'provenance':review.get('context_provenance') if review else None},'raw_sha256':digest(rawbytes),'evidence':evidence,'window_count':len(win),'policy':'Candidate supervision only until reviewed. No paraphrasing, invented speaker IDs, or inferred acoustic tags. Windows share original audio and retain source split.'}
    artifacts={'report.json':encode(report),'segments.json':encode({'record_id':rid,'revision':revision,'segments':ordered}),'windows.json':encode(win)}
    if review_bytes:artifacts['review.json']=review_bytes
    for name,data in artifacts.items():store.put(rel/name,data)
    store.obj(rel/'receipt.json',{'record_id':rid,'revision':revision,'artifacts_sha256':{k:digest(v) for k,v in artifacts.items()}})
    store.obj(pathlib.Path('postprocess/v1/current')/f'{rid}.json',{'revision':revision,'path':str(rel),'status':status,'base_input_digest':base_digest})
    update_index(db,r,fingerprint,revision,rel,report,win)
    with db:db.execute('DELETE FROM failures WHERE record_id=?',(rid,))
    logging.info('%s %s: %d windows, %d review signals',r['id'],status,len(win),len(problems));return True

def publish(store,db,total):
    counts=[dict(r) for r in db.execute('SELECT status,count(*) records,sum(duration)/3600 hours FROM recordings GROUP BY status')]
    issues=[dict(r) for r in db.execute('SELECT kind,severity,count(*) n FROM review_queue GROUP BY kind,severity ORDER BY n DESC')]
    summary={'updated_at':time.time(),'version':VERSION,'selected_records':total,'processed_records':db.execute('SELECT count(*) FROM recordings').fetchone()[0],'statuses':counts,'windows':db.execute('SELECT count(*) FROM windows').fetchone()[0],'approved_windows':db.execute('SELECT count(*) FROM windows WHERE training_ready=1').fetchone()[0],'review_signals':issues,'processing_failures':[dict(r) for r in db.execute('SELECT * FROM failures')],'note':'Window hours are repeated views, not additional unique audio. Human review approval is version-bound. No API calls.'}
    store.obj('postprocess/v1/summary.json',summary)
    top=[dict(r) for r in db.execute("SELECT * FROM review_queue ORDER BY CASE severity WHEN 'blocking' THEN 0 WHEN 'review' THEN 1 ELSE 2 END,CASE kind WHEN 'possible_missing_speech' THEN 0 ELSE 1 END,coalesce(end-start,0) DESC LIMIT 500")]
    store.put('postprocess/v1/review-top500.jsonl',b''.join(encode(r).replace(b'\n',b'')+b'\n' for r in top))
    fd,tmp=tempfile.mkstemp(dir=store.backup,suffix='.sqlite3');os.close(fd)
    try:
        snapshot=sqlite3.connect(tmp);db.backup(snapshot);snapshot.close();store.put('postprocess/v1/catalog.sqlite3',pathlib.Path(tmp).read_bytes())
    finally:os.unlink(tmp)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--backup',required=True);p.add_argument('--out');p.add_argument('--once',action='store_true');args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');root=pathlib.Path(args.root);store=Store(root,args.out or root/'datasets/soniox-500h-v1',args.backup,(root/'.asr-volume-id').read_text().strip())
    lock=(store.backup/'postprocess.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:STOP.set())
    while not store.ready():
        if STOP.wait(10):return
    selection=(store.backup/'selection.json').read_bytes();selection_sha=digest(selection)
    if selection_sha!=(store.backup/'selection.sha256').read_text().strip():raise ValueError('Selection checksum mismatch')
    rows=json.loads(selection);groups=collections.defaultdict(set)
    for r in rows:groups[r['show']].add(r['split'])
    if any(len(v)>1 for v in groups.values()):raise ValueError('Show leakage across splits')
    db=connect(store.backup/'postprocess/working.sqlite3')
    while not STOP.is_set():
        changed=0
        if not store.ready():STOP.wait(10);continue
        for r in rows:
            if STOP.is_set():break
            try:
                if process(store,db,r,selection_sha):
                    changed+=1
                    if changed%10==0:publish(store,db,len(rows))
            except Exception as e:
                logging.exception('Postprocessing failed for %s',r['id'])
                changed+=1
                with db:
                    db.execute('INSERT OR REPLACE INTO failures VALUES(?,?,?)',(r['record_id'],f'{type(e).__name__}: {e}',time.time()))
                    db.execute("UPDATE recordings SET fingerprint='retry',status='processing_error' WHERE record_id=?",(r['record_id'],))
                    db.execute('UPDATE windows SET training_ready=0 WHERE record_id=?',(r['record_id'],))
                try:store.obj(pathlib.Path('postprocess/v1/errors')/f"{r['record_id']}.json",{'error':f'{type(e).__name__}: {e}','time':time.time(),'training_ready':False})
                except OSError:pass
        if changed or args.once or not (store.out/'postprocess/v1/summary.json').exists():publish(store,db,len(rows))
        if args.once:break
        STOP.wait(30)
    db.close()
if __name__=='__main__':main()
