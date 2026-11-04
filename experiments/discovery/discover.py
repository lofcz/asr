"""Fetch a small, reproducible sample of public catalog pages, without a browser."""
import json, pathlib, re, time, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
for filename, path in [('robots.txt','robots.txt'),('sitemap.txt','sitemap.xml'),('podcasts1.txt','sitemap-podcasts/1.xml'),('podcasts2.txt','sitemap-podcasts/2.xml'),('show.txt','podcast/na-vybornou'),('proti-proudu.txt','podcast/proti-proudu')]:
    start=time.monotonic()
    with urllib.request.urlopen('https://ceskepodcasty.cz/'+path,timeout=30) as response:
        body=response.read()
    (ROOT/filename).write_bytes(body)
    if path.startswith('podcast/'):
        match=re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',body.decode(),re.S)
        data=json.loads(match[1])
        (ROOT/pathlib.Path(filename).with_suffix('.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(filename,len(body),round(time.monotonic()-start,3),flush=True)
    time.sleep(1)
