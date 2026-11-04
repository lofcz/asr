def merge_words(tokens):
    words=[]
    for t in tokens:
        if t.get('speaker') is None or t.get('is_audio_event') or t.get('text','').strip() in ('<end>','<fin>') or t.get('start_ms',-1)<0:
            continue
        if words and not t['text'][:1].isspace() and t['speaker']==words[-1]['speaker']:
            w=words[-1];w['text']+=t['text'];w['end_ms']=max(w['end_ms'],t['end_ms']);w['confidence_sum']+=t.get('confidence',0);w['pieces']+=1
        else:words.append({**t,'confidence_sum':t.get('confidence',0),'pieces':1})
    for w in words:w['confidence']=w['confidence_sum']/w['pieces']
    return words

