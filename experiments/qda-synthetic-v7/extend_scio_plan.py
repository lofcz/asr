"""Fill newly verified catalog gaps using only unsent Scio slots, same hours."""
from collections import Counter
import json,time
from pathlib import Path
from core import HERE,DATA,read,save
from revise_plan import catalog,paid


def extend():
    p=read('plan.json');pools,products=catalog();sessions=p['sessions']
    changes=[]
    for lang in ('cs','sk'):
        counts=Counter(s.get('canonical_product') for s in sessions if s['language']==[lang])
        for i,product in enumerate(products):
            name=product['canonical_name']
            if counts[name]:continue
            candidates=[s for s in sessions if s['language']==[lang] and s.get('topic_group')=='scio'
                and s.get('canonical_product') not in ('ScioBot','ScioŠkola','ScioLink')
                and counts[s.get('canonical_product')]>1 and not paid(s)
                and not read('sessions/'+s['id']+'/voices.json')
                and not read('scripts/'+s['id']+'/part-000.json')]
            if not candidates:raise RuntimeError('No unsubmitted Scio slot for '+name+' / '+lang)
            candidates.sort(key=lambda s:(not 60<=s['planned_seconds']<=300,s['planned_seconds'],s['id']))
            s=candidates[0];old_id=s['id'];old_product=s['canonical_product']
            forms=product['forms_'+lang];n=1 if s['planned_seconds']<=60 else 2
            terms=forms[:n]
            acronym='NSZ' if lang=='cs' else 'NPS'
            if acronym in product.get('related_terms',[]):terms=list(dict.fromkeys(terms+[acronym]))
            item=pools['scio'][i]
            s.update({'id':'v3ext-'+old_id,'canonical_product':name,'topic_id':item['id'],
                'topic':item['topic'],'vocabulary':item['vocabulary'],'source_notes':item['source_notes'],
                'scenario':item['scenario'],'roles':item['roles'],'angles':item['angles'],
                'required_terms':terms,'required_terms_by_part':[terms],
                'required_term_forms':forms,'customized_context':', '.join(item['vocabulary'].split(', ')[:4]) if s['seed']%2==0 else ''})
            counts[old_product]-=1;counts[name]+=1
            changes.append({'old_id':old_id,'new_id':s['id'],'old_product':old_product,
                            'new_product':name,'language':[lang],'seconds':s['planned_seconds']})
    if not changes:return
    assert sum(s['planned_seconds'] for s in sessions)==216000
    assert len({s['id'] for s in sessions})==len(sessions)
    save('history/plan-before-expanded-test-catalog.json',read('plan.json'))
    p.update({'sessions':sessions,'scio_products':len(products),'catalog_revision':2,
              'source_topic_counts':{k:len(v) for k,v in pools.items()},
              'scio_product_session_counts':dict(Counter(s.get('canonical_product') for s in sessions if s.get('canonical_product')))})
    save('plan.json',p);save('content-catalog.json',pools)
    for change in changes:
        save('sessions/'+change['old_id']+'/state.json',{'status':'superseded_unsubmitted',
            'reason':'Cover remaining officially listed Scio test families','replacement':change['new_id'],'at':time.time()})
    save('plan-catalog-extension.json',{'changes':changes,'hours_added':0,'catalog_scio_items':len(products),'at':time.time()})
    for lang in ('cs','sk'):
        assert {s.get('canonical_product') for s in sessions if s['language']==[lang]} >= {x['canonical_name'] for x in products}
    print('Catalog extended:',len(products),'Scio items;',len(changes),'unsubmitted slots reassigned; no added hours.')


if __name__=='__main__':extend()
