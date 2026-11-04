"""Audited acronym spellings; raw paid requests and timestamps stay immutable."""
import re

# These five aliases were found in the initial corpus audit. Keep this explicit:
# no fuzzy matching, accent stripping, or rewriting of ordinary Czech/Slovak.
ALIASES={'ej aj':'AI','ej pí aj':'API','á pé í':'API','gé pé ú':'GPU','ó es pé':'OSP'}
PATTERN=re.compile(r'(?<!\w)(?:'+ '|'.join(re.escape(s) for s in sorted(ALIASES,key=len,reverse=True))+r')(?!\w)',re.I)


def canonicalize(text):
    edits=[]
    def replace(m):
        target=ALIASES[m.group().casefold()]
        edits.append({'start':m.start(),'end':m.end(),'source':m.group(),'target':target})
        return target
    return PATTERN.sub(replace,text),edits


def native_words(words):
    """Join only a documented spelled acronym, with its original time union."""
    text='';spans=[]
    for w in words:
        if text:text+=' '
        start=len(text);text+=w['text'];spans.append((start,len(text)))
    _,edits=canonicalize(text)
    groups=[]
    for e in edits:
        ids=[i for i,(a,b) in enumerate(spans) if a<e['end'] and b>e['start']]
        if not ids:continue
        a,b=ids[0],ids[-1]
        if groups and a<=groups[-1][1]:groups[-1]=(groups[-1][0],max(b,groups[-1][1]))
        else:groups.append((a,b))
    result=[];cursor=0
    for a,b in groups:
        result.extend(words[cursor:a]);chunk=words[a:b+1]
        converted,_=canonicalize(' '.join(w['text'] for w in chunk))
        result.append({'text':converted,'start':min(w['start'] for w in chunk),
                       'end':max(w['end'] for w in chunk),'normalization':'audited_acronym'})
        cursor=b+1
    result.extend(words[cursor:])
    return result
