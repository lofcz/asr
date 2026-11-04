import json,pathlib
p=pathlib.Path(__file__).resolve().parent
search=json.loads((p/'expanded-search.json').read_text())
# Deliberately selected result positions; unrelated search hits are excluded.
chosen={'Host Lucie Výborné':[0],'Osobnost Plus':[0],'Spotlight Aktuálně':[0],'Hausbot Petr Horký':[1],'CzechCrunch podcast':[1,2],'Lupa.cz podcast':[0],'Na vlně podnikání':[0],'Kecy a politika':[0],'Dobrovský a Šídlo':[0],'Vinohradská 12':[0],'Blízká setkání':[0],'Vizitka Vltava':[0],'Radio Universum':[0],'Aust podcast':[2],'NORA Fridrichová':[1],'Host Rádia Z':[0,1],'Hovory Český rozhlas':[0,1],'Neurazitelné rozhovory':[0],'Dobré ráno SME':[0],'Ráno Nahlas':[0],'Braňo Závodský Naživo':[0,1,2],'Na rovinu Aktuality':[0],'Rozhovory so Šimonom':[1],'Rozhovory so Zdenom Gáfrikom':[0],'Bekimovo horúce kreslo':[0],'ADELA trochu inak':[0],'Ľudskosť':[0],'Ekonómia ľudskou rečou':[0],'Index SME':[0],'Dejiny SME':[0],'Klik SME':[0],'Vedátorský podcast':[0],'Mozgová Atletika':[0]}
rows=[]
for entry in search:
 for i in chosen.get(entry['query'],[]):
  r=entry['results'][i];title=r['title'];fixed=entry['query'] in ['Kecy a politika','Dobrovský a Šídlo','Klik SME','Vedátorský podcast']
  rows.append({'browse_id':r['browseId'],'title':title,'language':entry['language'],'search_query':entry['query'],'max_hours':3 if fixed else (12 if entry['language']=='cs' else 9),'speaker_count_hint':2,'review_status':'candidate_title_matched_pending_channel_review','quota_group':entry['query']})
for g in json.loads((p/'reviewed-playlists.json').read_text()):
 if g['label'] in ['Startitup']:continue
 rows.append({'browse_id':g['browse_id'],'title':g['label'],'language':g['language'],'max_hours':3 if g['label'] in ['Přepište dějiny'] else (8 if g['label']=='Lužifčák' else 12),'speaker_count_hint':3 if g['label']=='Lužifčák' else 2,'review_status':'channel_and_episode_titles_reviewed','quota_group':g['label']})
(pathlib.Path('queue')/'candidate-shows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print('Candidate playlists',len(rows))
