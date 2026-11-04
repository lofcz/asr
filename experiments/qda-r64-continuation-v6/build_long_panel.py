"""Reserve long development sessions, disjoint from training and the small panel."""
import bisect, hashlib, json, math, os
from pathlib import Path
from types import SimpleNamespace
os.environ['USE_TORCH']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
from transformers import PreTrainedTokenizerFast
from contracts import atomic_json,sha,target_with_spans,content_token_mask
from dataset import safe_cuts,window_segments
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    data=ROOT/'datasets/releases/qda-unseen-v6'
    sources=json.loads((data/'sources.json').read_text())
    plan=json.loads((HERE/'plan.json').read_text())
    excluded={r['record_id'] for r in plan['validation']}
    train=[r for p in plan['probe_pairs'] for r in p]
    tokenizer=PreTrainedTokenizerFast.from_pretrained(HERE.parent/'qda-full-r64-v4/selector-tokenizer')
    chosen=[]
    for language,nominal in [('cs',1800),('sk',3300)]:
        candidates=sorted((s for s in sources if s['split']=='development' and s['language']==language
                           and s['record_id'] not in excluded and s['audio_duration']>=nominal),key=lambda s:s['record_id'])
        for s in candidates:
            label=json.loads((data/s['label_path']).read_text())
            cuts=safe_cuts(label['segments'],s['audio_duration'])
            end=cuts[bisect.bisect_left(cuts,nominal)]
            target=window_segments(label['segments'],0,end,s['audio_duration'])
            body,spans=target_with_spans({'segments':[dict(start=t['Start'],end=t['End'],speaker=t['Speaker'],text=t['Content']) for t in target]})
            template=tokenizer.apply_chat_template([{'role':'assistant','content':body}],tokenize=False,add_generation_prompt=False)
            tokenized=tokenizer(template,add_special_tokens=False,return_offsets_mapping=True)
            offset=template.index(body);cm=content_token_mask(tokenized['offset_mapping'],[(a+offset,b+offset) for a,b in spans])
            nt=len(tokenized['input_ids']);total=nt+math.ceil(end*7.5)+128
            if total>65536:continue
            key=f"{s['record_id']}|{0.:.6f}|{end:.6f}|{s['label_sha256']}"
            chosen.append({**s,'split':'long_evaluation','original_split':'development',
                'window_id':hashlib.sha256(key.encode()).hexdigest()[:24],
                'source_start':0.,'source_end':end,'duration':end,'duration_class':'long',
                'target':target,'target_tokens':nt,'total_tokens':total,'content_tokens':sum(cm),
                'target_json_sha256':hashlib.sha256(body.encode()).hexdigest(),'input_sha256':None,
                'reference_quality':'Held-out long development session; original Soniox reference; never used for training or LR selection'})
            break
        else:raise ValueError('No representable long held-out session: '+language)
    assert not {r['audio_sha256'] for r in train}&{r['audio_sha256'] for r in chosen}
    assert not {r['show'] for r in train}&{r['show'] for r in chosen}
    atomic_json(HERE/'long-evaluation-selection.json',{'rows':chosen,'plan_sha256':sha(HERE/'plan.json'),
        'role':'Long-session transfer evaluation; separate from the fixed trajectory panel; confirmation sources remain untouched',
        'training_overlap_seconds':0,'evaluation_timing':'At completion, before claiming long-session readiness'})
    print(json.dumps([{'language':r['language'],'minutes':r['duration']/60,'tokens':r['total_tokens']} for r in chosen]))


if __name__=='__main__':main()
