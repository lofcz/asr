import json,pathlib,time,inspect
from ytmusicapi import YTMusic
ROOT=pathlib.Path(__file__).resolve().parent
yt=YTMusic(language='en',location='CZ')
print('get_podcast',inspect.signature(yt.get_podcast),flush=True)
shows={};episodes={};logs=[]
for query,lang in [('Proti Proudu','cs'),('Čestmír Strakatý','cs'),('český podcast rozhovor','cs'),('Rozhovory ZKH','sk'),('Lužifčák','sk'),('slovenský podcast rozhovory','sk')]:
 start=time.monotonic()
 try:
  results=yt.search(query,filter='podcasts',limit=10)
  (ROOT/(lang+'-'+str(len(logs))+'-search.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2))
  for r in results:
   if r.get('browseId'):shows[r['browseId']]={**r,'language_hint':lang,'query':query}
  log={'query':query,'results':len(results),'seconds':time.monotonic()-start}
 except Exception as e:log={'query':query,'error':str(e)}
 logs.append(log);print(json.dumps(log,ensure_ascii=False),flush=True);time.sleep(1)
for key,show in list(shows.items())[:6]:
 try:
  podcast=yt.get_podcast(key,limit=100)
  (ROOT/(key+'.json')).write_text(json.dumps(podcast,ensure_ascii=False,indent=2))
  for row in podcast.get('episodes',[]):
   if row.get('videoId'):episodes[row['videoId']]={**row,'show':podcast.get('title'),'browse_id':key,'language_hint':show['language_hint'],'url':'https://www.youtube.com/watch?v='+row['videoId']}
  print('PODCAST',podcast.get('title'),len(podcast.get('episodes',[])),flush=True)
 except Exception as e:print('PODCAST ERROR',key,str(e),flush=True)
 time.sleep(1)
(ROOT/'shows.json').write_text(json.dumps(shows,ensure_ascii=False,indent=2))
(ROOT/'episodes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in episodes.values()))
(ROOT/'search-log.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2))
print('TOTAL',len(shows),'shows',len(episodes),'episodes',flush=True)
