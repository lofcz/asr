"""Evaluate vanilla and the final adapter on the reserved longer sessions once."""
import argparse,json,pathlib,time
from contracts import sha,atomic_json
from gates import demand
HERE=pathlib.Path(__file__).resolve().parent


def prepare(rank):
    import torch
    from prepare import FeatureEncoder,checked_input
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    plan=json.loads((HERE/'plan.json').read_text())
    selection=json.loads((HERE/'long-evaluation-selection.json').read_text())
    demand(selection['plan_sha256']==sha(HERE/'plan.json'),'Long evaluation plan changed')
    row=selection['rows'][rank]
    demand(row['split']=='long_evaluation' and row['original_split']=='development','Unexpected evaluation source role')
    FeatureEncoder().write(row,plan)
    actual=checked_input(row,plan)
    atomic_json(HERE/f'long-evaluation-ready-{rank}.json',actual)


def evaluate_final(adapter):
    from evaluation import parallel,PY
    from final_report import compare_all
    parallel([[PY,'long_evaluation.py','--prepare-rank',str(i)] for i in (0,1)],'prepare-long-heldout')
    rows=[json.loads((HERE/f'long-evaluation-ready-{i}.json').read_text()) for i in (0,1)]
    panel=HERE/'long-evaluation-ready.json';atomic_json(panel,{'rows':rows})
    values={}
    for name,checkpoint in [('vanilla',None),('r64-full-final',adapter)]:
        out=HERE/'long-evaluations'/name
        flags=['--adapter',str(checkpoint)] if checkpoint else []
        parallel([[PY,'evaluate_panel.py','--rank',str(i),'--out',str(out),'--panel',str(panel),'--wall-budget','7200',*flags] for i in (0,1)],'long-evaluation-'+name)
        values[name]=[json.loads((out/r['window_id']/'result.json').read_text()) for r in rows]
        demand(all(r.get('complete') for r in values[name]),'Long output incomplete; preserve raw output and do not omit any session')
    result=compare_all(values['vanilla'],values['r64-full-final'])
    result.update(role='Reserved long development sessions; distinct from fixed marker/short-panel trajectory',
        per_recording=[{'language':r['language'],'duration':r['duration'],'metrics':r['metrics'],'output_issues':r.get('output_issues',[])} for r in values['r64-full-final']],
        adapter_sha256=sha(adapter/'adapter_model.safetensors'),panel_sha256=sha(panel),completed_at=time.time())
    atomic_json(HERE/'long-evaluation-result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-rank',type=int,choices=[0,1],required=True)
    prepare(parser.parse_args().prepare_rank)
