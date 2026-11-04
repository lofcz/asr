"""Balance actual context duration as well as language within a fixed audio budget."""
import collections, random

def duration_class(row):
    return 'short' if row['duration'] <= 660 else 'medium' if row['duration'] <= 1860 else 'long'

def arrange(training, seed, budget_seconds=20*3600):
    rows=sorted(training,key=lambda r:(r['total_tokens'],r['window_id']))
    remaining=[rows[i:i+2] for i in range(0,len(rows),2)]
    random.Random(seed).shuffle(remaining)
    desired={'short':.25,'medium':.55,'long':.20}
    langs=collections.Counter();lengths=collections.Counter();probe=[];seen=0.
    def additions(pair):
        l=collections.Counter();d=collections.Counter()
        for r in pair:l[r['language']]+=r['duration'];d[duration_class(r)]+=r['duration']
        return l,d
    def score(pair):
        l,d=additions(pair);after_l=langs+l;after_d=lengths+d;total=sum(after_l.values())
        return ((after_l['sk']/total-.30)**2/.30 +
                sum((after_d[k]/total-v)**2/v for k,v in desired.items()))
    while remaining:
        fits=[i for i,pair in enumerate(remaining) if seen+sum(r['duration'] for r in pair)<=budget_seconds]
        if not fits:break
        i=min(fits,key=lambda i:(score(remaining[i]),i));pair=remaining.pop(i)
        probe.append(pair);l,d=additions(pair);langs.update(l);lengths.update(d);seen+=sum(l.values())
    assert seen>=budget_seconds*.95,'Insufficient training budget after quality gates'
    assert .25<=langs['sk']/seen<=.35,'Language balance outside declared bounds'
    assert .18<=lengths['long']/seen<=.28,'Too little/much actual long-session exposure'
    assert .18<=lengths['short']/seen<=.32,'Short-session exposure outside declared bounds'
    for language in ('cs','sk'):
        assert any(r['language']==language and r['duration']>=2700 for pair in probe for r in pair),'Missing >=45-minute session: '+language
    return probe+remaining,probe
