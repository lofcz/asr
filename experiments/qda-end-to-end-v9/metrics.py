"""Word-safe reviewed targets, explicit event exclusion, and duration-weighted DER."""
import dataclasses
import math
import pathlib
import re
import sys
import unicodedata
from number_wer import score as number_score
from runtime import atomic as atomic_json
def demand(condition, message):
    if not condition:raise RuntimeError(message)

def validate_reference(reference,duration):
    demand(isinstance(reference,list) and bool(reference),'Empty reference')
    events={'[Unintelligible Speech]','[Music]','[Human Sounds]','[Environmental Sounds]','[Noise]','[Silence]'}
    import math
    for s in reference:
        demand(isinstance(s,dict) and {'Start','End','Content'}<=s.keys(),'Invalid reference schema')
        a,b=s['Start'],s['End']
        demand(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (a,b)),'Invalid reference time')
        demand(0<=a<=b<=duration+.02,'Reference time outside clip')
        demand(isinstance(s['Content'],str) and bool(s['Content'].strip()),'Empty reference content')
        demand(s['Content'] in events or (isinstance(s.get('Speaker'),(int,str)) and not isinstance(s.get('Speaker'),bool)),'Speech has no speaker')

EVENTS={'[Unintelligible Speech]','[Music]','[Human Sounds]','[Environmental Sounds]','[Noise]','[Silence]'}
ABSTENTIONS={'[Speech]'}


def validate_hypothesis(rows,duration):
    """Preserve explicit speech abstentions; never invent words or a speaker."""
    demand(isinstance(rows,list) and bool(rows),'Empty hypothesis')
    for row in rows:
        if not isinstance(row,dict) or row.get('Content') not in ABSTENTIONS:
            validate_reference([row],duration);continue
        demand({'Start','End','Content'}<=row.keys(),'Invalid speech-abstention schema')
        a,b=row['Start'],row['End']
        demand(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (a,b)),'Invalid speech-abstention time')
        demand(0<=a<=b<=duration+.02,'Speech abstention outside clip')


def normalized(text):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFC',text).casefold()))
def speech(rows):return [s for s in rows if s['Content'].strip() not in EVENTS]


def metrics(ref,hyp,folder,duration,language):
    import meeteval
    from rapidfuzz.distance import Levenshtein
    abstentions=[s for s in hyp if s['Content'] in ABSTENTIONS]
    ref=speech(ref);hyp=speech([s for s in hyp if s['Content'] not in ABSTENTIONS])
    a=normalized(' '.join(s['Content'] for s in ref));b=normalized(' '.join(s['Content'] for s in hyp));aa=a.split();bb=b.split()
    if not aa:raise ValueError('No scored reference words')
    ops=Levenshtein.editops(aa,bb)
    raw_a=unicodedata.normalize('NFC',' '.join(s['Content'] for s in ref))
    raw_b=unicodedata.normalize('NFC',' '.join(s['Content'] for s in hyp))
    result={'reference_words':len(aa),'hypothesis_words':len(bb),'wer':len(ops)/len(aa),
            'substitutions':sum(o.tag=='replace' for o in ops),'deletions':sum(o.tag=='delete' for o in ops),'insertions':sum(o.tag=='insert' for o in ops),
            'reference_speakers':len({str(s['Speaker']) for s in ref}),'predicted_speakers':len({str(s['Speaker']) for s in hyp}),
            'reference_segments':len(ref),'hypothesis_segments':len(hyp),
            'normalization':'NFC/casefold/Unicode words; diacritics preserved; whole segments joined with whitespace; only explicit acoustic tags excluded'}
    result['raw_cer']=Levenshtein.distance(raw_a,raw_b)/len(raw_a) if raw_a else None
    result['raw_cer_policy']='NFC full transcript including case, punctuation and spaces; spelling and inflection retained'
    result.update(number_score(' '.join(s['Content'] for s in ref),' '.join(s['Content'] for s in hyp),language))
    result['speech_abstention_segments']=len(abstentions)
    result['speech_abstention_seconds']=sum(s['End']-s['Start'] for s in abstentions)
    result['abstention_policy']='[Speech] supplies no words or diarized speaker. Keep all reference words/time: missing transcript and speaker coverage remain errors. Raw output unchanged.'
    def timed(rows):return [{'session_id':'sample','speaker':str(s['Speaker']),'start_time':s['Start'],'end_time':s['End'],'words':normalized(s['Content'])} for s in rows]
    uem=folder/'evaluation.uem';uem.write_text(f'sample 1 0 {duration}\n')
    der=meeteval.der.md_eval_22(timed(ref),timed(hyp),collar=.25,uem=uem)['sample']
    result['der_250ms']=float(der.error_rate);result['der_details']=dataclasses.asdict(der)
    atomic_json(folder/'word-errors.json',[{'operation':o.tag,'reference_index':o.src_pos,'hypothesis_index':o.dest_pos,
                                         'reference_word':aa[o.src_pos] if o.tag!='insert' else None,'hypothesis_word':bb[o.dest_pos] if o.tag!='delete' else None} for o in ops])
    return result
