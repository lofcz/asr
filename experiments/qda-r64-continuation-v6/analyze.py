"""Paired recording-level uncertainty; repeated clips are not independent trials."""
import collections
import json
import math
import pathlib
import random
from contracts import atomic_json


def aggregate(rows):
    metrics=[r['metrics'] for r in rows]
    words=sum(m['number_normalized_reference_words'] for m in metrics)
    speech=sum(float(m['der_details']['scored_speaker_time']) for m in metrics)
    errors=sum(m['number_normalized_errors'] for m in metrics)
    der_errors=sum(sum(float(m['der_details'][k]) for k in ('missed_speaker_time','falarm_speaker_time','speaker_error_time')) for m in metrics)
    return {'number_normalized_wer':errors/words,'errors':errors,'reference_words':words,'der_250ms':der_errors/speech,
            'diarization_error_seconds':der_errors,'speaker_seconds':speech,'raw_json_valid':sum(r.get('complete') is True for r in rows),'samples':len(rows)}


def compare(base,current,draws=20000):
    if {r['window_id'] for r in base}!={r['window_id'] for r in current}:raise ValueError('Panels differ')
    if not all(r.get('complete') for r in base+current):raise ValueError('Invalid or incomplete output: no silent repair or subset scoring')
    groups=collections.defaultdict(lambda:[[],[]])
    for side,rows in enumerate((base,current)):
        for row in rows:groups[row['record_id']][side].append(row)
    pairs=[(aggregate(a),aggregate(b)) for a,b in groups.values()];rng=random.Random(20261023)
    delta={'number_normalized_wer':[],'der_250ms':[]}
    for _ in range(draws):
        chosen=[pairs[rng.randrange(len(pairs))] for _ in pairs]
        for key,num,den in [('number_normalized_wer','errors','reference_words'),('der_250ms','diarization_error_seconds','speaker_seconds')]:
            scores=[sum(pair[i][num] for pair in chosen)/sum(pair[i][den] for pair in chosen) for i in (0,1)]
            delta[key].append(scores[1]-scores[0])
    a,b=aggregate(base),aggregate(current)
    intervals={k:[sorted(v)[int(.025*draws)],sorted(v)[min(draws-1,int(.975*draws))]] for k,v in delta.items()}
    relative=1-b['number_normalized_wer']/a['number_normalized_wer'] if a['number_normalized_wer'] else None
    return {'vanilla':a,'finetuned':b,'relative_word_error_reduction':relative,'paired_recording_bootstrap_95':intervals,
            'recordings':len(groups),'screening_goal_met':relative is not None and relative>=.2 and intervals['number_normalized_wer'][1]<0 and b['der_250ms']<=a['der_250ms']+.005,
            'interpretation':'Development screening, not confirmatory significance after repeated checkpoint selection. Confirm on untouched recordings before scaling.'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('baseline',type=pathlib.Path);p.add_argument('current',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    read=lambda folder:[json.loads(p.read_text()) for p in sorted(folder.glob('*/result.json'))]
    result=compare(read(a.baseline),read(a.current));atomic_json(a.out,result);print(json.dumps(result,indent=2))
