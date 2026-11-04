import json,pathlib,time
from ytmusicapi import YTMusic
p=pathlib.Path(__file__).resolve().parent;yt=YTMusic(language='en',location='CZ');out=[]
for query,lang in [('DVTV rozhovory','cs'),('StandaShow','cs'),('Deep Talks Petr Ludwig','cs'),('Vojta Žižka','cs'),('Cukrfree podcast','cs'),('Přepište dějiny','cs'),('V redakcii','sk'),('Startitup rozhovory','sk')]:
 try:
  r=yt.search(query,filter='podcasts',limit=5);out.append({'query':query,'language':lang,'results':r})
  print(query,[(x.get('title'),x.get('browseId')) for x in r[:4]],flush=True)
 except Exception as e:print(query,str(e),flush=True)
 time.sleep(1)
(p/'manual-candidate-search.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
