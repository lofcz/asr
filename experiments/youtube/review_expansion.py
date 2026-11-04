import json,pathlib,time
from ytmusicapi import YTMusic
root=pathlib.Path(__file__).resolve().parents[2];p=root/'experiments/youtube/expanded';p.mkdir(exist_ok=True);yt=YTMusic();report=[]
for show in json.loads((root/'queue/candidate-shows.json').read_text()):
 key=show['browse_id'];target=p/(key+'.json')
 try:
  if target.exists():d=json.loads(target.read_text())
  else:d=yt.get_podcast(key,limit=200);target.write_text(json.dumps(d,ensure_ascii=False,indent=2))
  sample=[{'id':r.get('videoId'),'duration':r.get('duration'),'title':r.get('title')} for r in d.get('episodes',[])[:3]]
  r={**show,'author':d.get('author'),'actual_title':d.get('title'),'episodes':len(d.get('episodes',[])),'samples':sample};report.append(r)
  print(json.dumps(r,ensure_ascii=False),flush=True)
 except Exception as e:print(json.dumps({'title':show['title'],'error':str(e)},ensure_ascii=False),flush=True)
 (p/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));time.sleep(.6)
