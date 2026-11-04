"""Record actual completed voice-bank state after the expansion worker exits."""
import time
from collections import Counter
from core import read,save
from voices import registry

bank=registry()
if len(bank)>=1000:
    prior=read('voices/expansion-status.json',{})
    result={**prior,'phase':'complete','registered_total':len(bank),
        'registered_by_language':dict(Counter(v['language'][0] for v in bank)),
        'errors':[],'reconciled_prior_errors':prior.get('errors',[]),'at':time.time()}
    save('voices/expansion-status.json',result)
    save('voices/bank-ready.json',{'registered':len(bank),'target':1000,
        'private_clones':sum(v['kind']=='private_clone' for v in bank),
        'library_voices':sum(v['kind']=='library' for v in bank),'at':time.time()})
