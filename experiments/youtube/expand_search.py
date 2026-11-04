import json,pathlib,time
from ytmusicapi import YTMusic
p=pathlib.Path(__file__).resolve().parent;yt=YTMusic(language='en',location='CZ');out=[]
queries={'cs':['Host Lucie Výborné','Osobnost Plus','Prostor X','Spotlight Aktuálně','Hausbot Petr Horký','Forbes Byznys','CzechCrunch podcast','Lupa.cz podcast','Na vlně podnikání','Vlevo dole','Kecy a politika','Dobrovský a Šídlo','Ptám se já','5:59','Vinohradská 12','Blízká setkání','Vizitka Vltava','Bourání','Balanc Radio Wave','Diagnóza F','Radio Universum','Aust podcast','NORA Fridrichová','Host Rádia Z','Café Evropa','Hovory Český rozhlas','Hyde Park Civilizace','Interview ČT24','Zdraví v kostce','Neurazitelné rozhovory'],'sk':['Dobré ráno SME','Ráno Nahlas','Braňo Závodský Naživo','Na rovinu Aktuality','Rozhovory so Šimonom','Rozhovory so Zdenom Gáfrikom','Bekimovo horúce kreslo','ADELA trochu inak','Ľudskosť','Ekonómia ľudskou rečou','Index SME','Dejiny SME','Klik SME','Vedátorský podcast','Zvedavosť Denník N','Mozgová Atletika','Startitup','TECHBOX TÉMA','SPODCAST','Život je zmena']}
for lang,items in queries.items():
 for query in items:
  try:
   r=yt.search(query,filter='podcasts',limit=5);out.append({'query':query,'language':lang,'results':r})
   print(lang,query,[(x.get('title'),x.get('browseId')) for x in r[:3]],flush=True)
  except Exception as e:print(query,str(e),flush=True)
  (p/'expanded-search.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
  time.sleep(1)
