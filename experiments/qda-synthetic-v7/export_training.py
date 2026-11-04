"""Export VibeVoice's actual input schema, known speakers, native word timings."""
import io,json,math,os,re,subprocess,sys,time,wave
from pathlib import Path
from core import DATA,REPLICA,HERE,read,save,put,encode,digest,sha,verified
from target_text import canonicalize,native_words

RATE=24000
EXPORT_VERSION=3
COVERAGE_PROCESS=None

def lexical(text):
    return ''.join(c for c in text.casefold() if c.isalnum() or c=='_')


def merge_turns(segments):
    result=[]
    for segment in segments:
        if result and result[-1]['speaker']==segment['speaker'] and segment['start']-result[-1]['end']<1.5:
            result[-1]['text']+=' '+segment['text'];result[-1]['end']=segment['end']
        else:result.append(dict(segment))
    return result

def canonical_words(text,native,offset,speaker):
    """Match by lexical characters, never by timestamp gaps or subword tokens."""
    text,_=canonicalize(text);native=native_words(native)
    source=[];cursor=0
    for w in native:
        letters=lexical(w['text'])
        if letters:
            source.append((cursor,cursor+len(letters),w));cursor+=len(letters)
    spans=list(re.finditer(r'\S+',text))
    expected=''.join(lexical(s.group()) for s in spans)
    actual=''.join(lexical(w['text']) for w in native)
    if expected!=actual:
        raise ValueError('Cartesia normalized text differs from supplied transcript; preserve for review')
    result=[];cursor=0;si=0
    for s in spans:
        letters=lexical(s.group())
        if not letters:continue
        end=cursor+len(letters)
        while si<len(source) and source[si][1]<=cursor:si+=1
        matches=[];j=si
        while j<len(source) and source[j][0]<end:
            matches.append(source[j][2]);j+=1
        if not matches:raise ValueError('Missing native timestamp for canonical word')
        a=min(w['start'] for w in matches)+offset;b=max(w['end'] for w in matches)+offset
        if not (math.isfinite(a) and math.isfinite(b) and 0<=a<=b):raise ValueError('Invalid lexical word duration')
        result.append({'text':s.group(),'start':round(a,6),'end':round(b,6),'speaker':speaker,
                       'char_start':s.start(),'char_end':s.end(),
                       'native_point_timestamp':a==b})
        cursor=end
    return result


def export(directory):
    sid=directory.name;relative=Path('training/receipts')/(sid+'.json')
    source=json.loads((directory/'receipt.json').read_text())
    ready=read(relative)
    if ready and ready.get('export_version')==EXPORT_VERSION:
        if ready['source_composition_sha256']!=source['composition_sha256']:raise ValueError('Source composition changed')
        return ready
    if ready:
        history=f"history/training-export-v{ready.get('export_version',1)}"
        save(history+'/receipts/'+sid+'.json',ready)
        for name,key in [('labels','label_path'),('words','words_path')]:
            put(history+'/'+name+'/'+sid+'.json',(DATA/ready[key]).read_bytes())
    composition=json.loads(verified(Path('sessions')/sid/'composition.json',source['composition_sha256']))
    labels=json.loads(verified(Path('sessions')/sid/'labels.json',source['labels_sha256']))
    voices=read(Path('sessions')/sid/'voices.json');voice_ids={v['speaker']:v['voice_id'] for v in voices}
    if len(set(voice_ids.values()))!=source['speaker_count']:raise ValueError('Different speakers share one voice')
    work=DATA/'training/.working';work.mkdir(parents=True,exist_ok=True)
    temp=work/(sid+'.flac')
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','s16le','-ar',str(RATE),'-ac','1','-i','pipe:0',
                           '-c:a','flac',str(temp)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    cursor=0;segments=[];word_list=[];speaker_map={};reconstructed=[];normalizations=[]
    try:
        for item in composition:
            speaker=speaker_map.setdefault(item['speaker'],len(speaker_map))
            if item['voice_id']!=voice_ids[item['speaker']]:raise ValueError('Speaker voice changed inside recording')
            if item['sample_rate']!=RATE or item['offset_samples']<cursor:raise ValueError('Invalid sample timeline')
            gap=item['offset_samples']-cursor
            proc.stdin.write(b'\0\0'*gap)
            root=Path('utterances')/item['utterance_id'];receipt=read(root/'receipt.json')
            if receipt['audio_sha256']!=item['audio_sha256']:raise ValueError('Wrong utterance receipt')
            audio=verified(root/'audio.wav',item['audio_sha256'])
            native=json.loads(verified(root/'words.json',receipt['words_sha256']))
            quality=read(root/'quality.json',{})
            defects=set(quality.get('warnings',[]))&{'audio_clipping','low_audio_level'}
            if defects:raise ValueError('Audio signal needs review: '+', '.join(sorted(defects)))
            with wave.open(io.BytesIO(audio),'rb') as wav:
                if wav.getparams()[:3]!=(1,2,RATE):raise ValueError('Unexpected original audio format')
                pcm=wav.readframes(wav.getnframes())
            if len(pcm)//2!=item['samples'] or item['end_samples']!=item['offset_samples']+item['samples']:
                raise ValueError('Mismatched source sample count')
            proc.stdin.write(pcm)
            words=canonical_words(item['text'],native,item['offset_samples']/RATE,speaker)
            if not words or words[-1]['end']>item['end_samples']/RATE+.05:raise ValueError('Words extend outside paired utterance')
            target_text,edits=canonicalize(item['text'])
            if edits:normalizations.append({'utterance_id':item['utterance_id'],'edits':edits})
            segments.append({'speaker':speaker,'text':target_text,'start':words[0]['start'],'end':words[-1]['end']})
            word_list.extend({**w,'utterance_id':item['utterance_id']} for w in words)
            reconstructed.append(target_text);cursor=item['end_samples']
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError('FLAC encoding failed: '+err.decode(errors='replace')[:300])
    except Exception:
        proc.kill();proc.wait();raise
    duration=cursor/RATE
    if len(speaker_map)!=source['speaker_count']:raise ValueError('Incorrect exported speaker count')
    if any(not 0<=s['start']<s['end']<=duration+.05 for s in segments):raise ValueError('Invalid exported segment range')
    if [s['text'] for s in segments]!=reconstructed:raise ValueError('Transcript word loss or duplication')
    if any(b['start']<a['end']-.05 for a,b in zip(segments,segments[1:])):raise ValueError('Unexpected overlap in serialized TTS')
    segments=merge_turns(segments)
    if ' '.join(s['text'] for s in segments)!=' '.join(reconstructed):raise ValueError('Merged turn transcript mismatch')
    audio=temp.read_bytes();audio_rel=Path('training/audio')/(sid+'.flac');put(audio_rel,audio)
    target={'audio_duration':duration,'audio_path':'../audio/'+sid+'.flac','segments':segments,
            'customized_context':labels.get('customized_context',''),'language':source['language']}
    label_rel=Path('training/labels')/(sid+'.json');word_rel=Path('training/words')/(sid+'.json')
    put(label_rel,encode(target));put(word_rel,encode(word_list))
    result={'status':'complete','export_version':EXPORT_VERSION,'id':sid,'audio_path':str(audio_rel),'label_path':str(label_rel),'words_path':str(word_rel),
        'audio_duration':duration,'audio_sha256':digest(audio),'label_sha256':digest(encode(target)),
        'words_sha256':digest(encode(word_list)),'sample_rate':RATE,'speaker_count':len(speaker_map),
        'language':source['language'],'cohort':source['cohort'],'split':'train','timestamp_source':'cartesia-native',
        'source_composition_sha256':source['composition_sha256'],'source_labels_sha256':source['labels_sha256'],
        'known_speaker_voice_ids':{str(speaker_map[k]):v for k,v in voice_ids.items()},
        'verified_words':len(word_list),'segments':len(segments),'target_text_normalizations':normalizations,'completed_at':time.time()}
    save(relative,result);temp.unlink();print('exported',sid,round(duration,2),flush=True)
    return result


def cycle():
    global COVERAGE_PROCESS
    for path in sorted((DATA/'sessions').glob('*/receipt.json')):
        sid=path.parent.name
        if read('training/review/'+sid+'.json'):continue
        try:export(path.parent)
        except Exception as e:
            save('training/review/'+sid+'.json',{'id':sid,'error':str(e),'at':time.time()})
            print('export_review',sid,str(e),flush=True)
    rows=[json.loads(p.read_text()) for p in (DATA/'training/receipts').glob('*.json')]
    rows.sort(key=lambda r:r['id'])
    put('training/manifest.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode())
    save('training/status.json',{'sessions':len(rows),'hours':sum(r['audio_duration'] for r in rows)/3600,
        'format':'VibeVoice finetuning-asr input JSON: audio_duration, audio_path, lowercase segments',
        'sample_rate':RATE,'timestamp_source':'cartesia-native','diarization_source':'known voice assignments',
        'review_sessions':len(list((DATA/'training/review').glob('*.json'))),'updated_at':time.time()})
    coverage=read('domain-coverage.json',{})
    if (coverage.get('completed_sessions')!=len(rows) and time.time()-coverage.get('updated_at',0)>300
            and (COVERAGE_PROCESS is None or COVERAGE_PROCESS.poll() is not None)):
        # Coverage grows with corpus size; keep it off the critical export path.
        COVERAGE_PROCESS=subprocess.Popen([sys.executable,str(HERE/'coverage.py')])


if __name__=='__main__':
    while True:
        cycle();time.sleep(10)
