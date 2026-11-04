"""Discover podcast playlists and enumerate episode IDs with ytmusicapi.
Run with .venv/bin/python. Country/query language is not verified spoken language.
"""
import argparse,json,pathlib,time
from ytmusicapi import YTMusic
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--query',action='append',required=True);parser.add_argument('--language-hint',choices=['cs','sk'],required=True);parser.add_argument('--search-limit',type=int,default=10);parser.add_argument('--episode-limit',type=int,default=100);parser.add_argument('--playlist-limit',type=int,default=5);args=parser.parse_args()
    yt=YTMusic(language='en',location='CZ' if args.language_hint=='cs' else 'SK');shows={};episodes={};errors=[]
    for query in args.query:
        try:
            for row in yt.search(query,filter='podcasts',limit=args.search_limit):
                if row.get('browseId'):shows[row['browseId']]={**row,'query':query,'language_hint':args.language_hint}
        except Exception as e:errors.append({'query':query,'error':str(e)})
        time.sleep(1)
    for key,show in list(shows.items())[:args.playlist_limit]:
        try:
            podcast=yt.get_podcast(key,limit=args.episode_limit or None)
            for row in podcast.get('episodes',[]):
                if row.get('videoId'):episodes[row['videoId']]={**row,'show':podcast.get('title'),'browse_id':key,'language_hint':args.language_hint,'language_verified':False,'speaker_count':None,'url':'https://www.youtube.com/watch?v='+row['videoId']}
        except Exception as e:errors.append({'browse_id':key,'error':str(e)})
        time.sleep(1)
    path=ROOT/'data/manifests';path.mkdir(exist_ok=True,parents=True)
    prefix='youtube-'+args.language_hint
    (path/(prefix+'-shows.json')).write_text(json.dumps(shows,ensure_ascii=False,indent=2))
    (path/(prefix+'-episodes.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in episodes.values()))
    (path/(prefix+'-errors.json')).write_text(json.dumps(errors,ensure_ascii=False,indent=2))
    print(json.dumps({'shows':len(shows),'unique_video_ids':len(episodes),'errors':len(errors)}))
if __name__=='__main__':main()
