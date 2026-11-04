import json,pathlib,time
from ytmusicapi import YTMusic
p=pathlib.Path(__file__).resolve().parent;yt=YTMusic();out=[]
choices=[('cs','Čestmír Strakatý','MPSPPLPK5bz7v9zh3oz-oW7ZcxsGuSYEb3Plld'),('cs','StandaShow','MPSPPLewDwxqAudOO8eoHE7QQZ4s4rd8VFUnBK'),('cs','Deep Talks','MPSPPL7w-InZKYvMHJsS9AVpk_cRnlWVxYasC4'),('cs','Vojta Žižka','MPSPPLvcr-uoKj1lNgOlfa8opn7De7c-6UGhrb'),('cs','Cukrfree','MPSPOLJ2zNiXvqoQ46mnNRCcKaAwLCza8z9arxQ'),('cs','Přepište dějiny','MPSPPL_bjTNnRtwB11jSjRgKLV57a5ZmGcOJRE'),('cs','Proti Proudu','MPSPPLf3Q-0TtQ8pdpfeOLBSAyVI-IV3l7FyT1'),('sk','V redakcii','MPSPPL0odj6Omi67-0NQqDA0hB_HCly3G0HZcE'),('sk','Rozhovory ZKH','MPSPPLX-qV3Lrf324gS5tWhzheVbtLpxFRXjXN'),('sk','Startitup','MPSPPLeSQcWDJHljlA1BiWMxbLTNLM2cGc8BAe'),('sk','Lužifčák','MPSPPLI7CuYaZCTmDGIkasKcpwD0htNI_JMdOP')]
for lang,label,key in choices:
 try:
  d=yt.get_podcast(key,limit=20);rows=d.get('episodes',[])[:15]
  out.append({'language':lang,'label':label,'browse_id':key,'podcast':d})
  print('\n',label,'author',d.get('author'),flush=True)
  for r in rows[:8]:print(r.get('videoId'),r.get('duration'),r.get('title'),flush=True)
 except Exception as e:print(label,str(e),flush=True)
 time.sleep(1)
(p/'reviewed-playlists.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
