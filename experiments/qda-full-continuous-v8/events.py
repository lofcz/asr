"""Only explicit audio-reviewed non-speech gaps become training event targets."""
import json
import pathlib
from contracts import sha

TAGS={'[Music]','[Human Sounds]','[Environmental Sounds]','[Noise]','[Silence]'}


def reviewed_events(data,row,quality):
    path=data/'acoustic-reviews'/(row['window_id']+'.json')
    if not path.exists():return None
    review=json.loads(path.read_text())
    if review.get('decision')!='approve' or not review.get('reviewer') or not review.get('reviewed_at'):return None
    if review.get('audio_sha256')!=row['audio_sha256'] or review.get('label_sha256')!=row['label_sha256']:raise ValueError('Stale acoustic review')
    gaps=quality['unresolved_nonspeech'];regions=review['regions']
    if len(gaps)!=len(regions):raise ValueError('Acoustic review does not cover every flagged gap')
    result=[]
    for (a,b),region in zip(gaps,regions):
        if abs(region['start']-a)>.001 or abs(region['end']-b)>.001 or region.get('tag') not in TAGS or region.get('listened') is not True:
            raise ValueError('Invalid/unreviewed acoustic region')
        result.append({'Start':round(a-row['source_start'],2),'End':round(b-row['source_start'],2),'Speaker':None,'Content':region['tag']})
    return result,sha(path)


def with_events(target,events):return sorted(target+events,key=lambda s:(s['Start'],s['End'],str(s.get('Speaker'))))
