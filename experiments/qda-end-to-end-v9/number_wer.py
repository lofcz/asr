"""Conservative CS/SK integer normalization v1. Never strips diacritics.

Supports cardinal integers 0..9999, standard hundreds/thousands and selected
unambiguous case forms. Ordinals, decimals, dates and leading-zero identifiers
are deliberately left untouched. The raw reference is never modified.
"""
import re, unicodedata
VERSION='cs-sk-cardinals-v1'
LEX={
 'cs': ['nula','jedna','dva','tři','čtyři','pět','šest','sedm','osm','devět','deset','jedenáct','dvanáct','třináct','čtrnáct','patnáct','šestnáct','sedmnáct','osmnáct','devatenáct'],
 'sk': ['nula','jeden','dva','tri','štyri','päť','šesť','sedem','osem','deväť','desať','jedenásť','dvanásť','trinásť','štrnásť','pätnásť','šestnásť','sedemnásť','osemnásť','devätnásť']}
TENS={'cs':['dvacet','třicet','čtyřicet','padesát','šedesát','sedmdesát','osmdesát','devadesát'], 'sk':['dvadsať','tridsať','štyridsať','päťdesiat','šesťdesiat','sedemdesiat','osemdesiat','deväťdesiat']}
FORMS={'cs': {1:'jeden jedno jedné jednoho jednomu jedním jednu',2:'dvě dvou dvěma',3:'třech třem třemi',4:'čtyř čtyřem čtyřmi',5:'pěti',6:'šesti',7:'sedmi',8:'osmi',9:'devíti'}, 'sk':{1:'jedna jedno jednej jedného jednému jedným jednu',2:'dve dvoch dvom dvoma',3:'troch trom tromi',4:'štyroch štyrom štyrmi',5:'piatich piatim piatimi',6:'šiestich šiestim šiestimi',7:'siedmich siedmim siedmimi',8:'ôsmich ôsmim ôsmimi',9:'deviatich deviatim deviatimi'}}

def tables(lang):
 if lang not in LEX:raise ValueError('Explicit cs or sk language is required')
 small={w:i for i,w in enumerate(LEX[lang])}
 for n,words in FORMS[lang].items():small.update({w:n for w in words.split()})
 tens={w:(i+2)*10 for i,w in enumerate(TENS[lang])}
 if lang=='cs':tens.update({w+'i':n for w,n in list(tens.items())})
 hundred={'sto':100}
 for n in range(2,10):
  phrase=(('dvě stě' if n==2 else LEX[lang][n]+(' sta' if n<5 else ' set')) if lang=='cs' else ['','','dvesto','tristo','štyristo','päťsto','šesťsto','sedemsto','osemsto','deväťsto'][n])
  hundred[phrase]=n*100
 return small,tens,hundred

def tokens(text,lang):
 small,tens,hundred=tables(lang)
 # Keep dotted/decimal/hyphenated numeric expressions indivisible: do not
 # reinterpret 1.2, 1,2, 1. or 2026-10-22 as cardinal integers.
 raw=re.findall(r'\d+(?:[.,:/-]\d+)+(?:\.)?|\d+\.|\w+',unicodedata.normalize('NFC',text).casefold())
 def below100(i):
  if i>=len(raw):return None
  w=raw[i]
  if w in tens:
   n=tens[w];j=i+1
   if j<len(raw) and 0<small.get(raw[j],100)<10:n+=small[raw[j]];j+=1
   return n,j
  if w in small:return small[w],i+1
 def below1000(i):
  for size in (2,1):
   phrase=' '.join(raw[i:i+size])
   if len(raw[i:i+size])==size and phrase in hundred:
    n=hundred[phrase];j=i+size;tail=below100(j)
    if tail and tail[0]>0:n+=tail[0];j=tail[1]
    return n,j
  return below100(i)
 out=[];i=0
 while i<len(raw):
  w=raw[i]
  if re.fullmatch(r'0|[1-9]\d{0,3}',w):out.append('#'+w);i+=1;continue
  n=None;j=i
  if w=='tisíc':n=1000;j=i+1
  elif i+1<len(raw) and 1<=small.get(w,100)<=9 and raw[i+1] in ('tisíc','tisíce'):
   n=small[w]*1000;j=i+2
  if n is not None:
   tail=below1000(j)
   if tail and tail[0]>0:n+=tail[0];j=tail[1]
   out.append('#'+str(n));i=j;continue
  found=below1000(i)
  if found:out.append('#'+str(found[0]));i=found[1]
  else:out.append(w);i+=1
 return out

def score(reference,hypothesis,lang):
 from rapidfuzz.distance import Levenshtein
 a=tokens(reference,lang);b=tokens(hypothesis,lang);ops=Levenshtein.editops(a,b)
 return {'number_normalized_wer':len(ops)/len(a) if a else (0 if not b else None),'number_normalized_errors':len(ops),'number_normalized_reference_words':len(a),'number_normalization_version':VERSION,'number_normalization_language':lang}
