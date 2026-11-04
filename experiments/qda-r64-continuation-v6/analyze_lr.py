"""Publish the upstream finder suggestion alongside raw held-out measurements."""
import json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=['/data/work/asr-experiments/qda-lr-calibration-v1/vendor',str(HERE)]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
from contracts import atomic_json,sha


def main():
    root=HERE/'result';h=json.loads((root/'history.json').read_text());config=json.loads((root/'config.json').read_text())
    if h['running']:raise RuntimeError('Range test has not finished')
    rows=h['records'];best=min(rows,key=lambda r:r['validation_content_loss'])
    finder=LRFinder.__new__(LRFinder);finder.history=h['upstream_history']
    plt.style.use('dark_background');fig,axes=plt.subplots(1,2,figsize=(14,5))
    _,suggestion=finder.plot(ax=axes[0],suggest_lr=True,skip_start=10,skip_end=5)
    axes[0].set_title('Upstream finder · smoothed held-out content loss')
    axes[1].semilogx([r['lr'] for r in rows],[r['validation_content_loss'] for r in rows],label='Held-out content',color='#71dfb5')
    axes[1].semilogx([r['lr'] for r in rows],[r['validation_structure_loss'] for r in rows],label='Held-out structure',color='#a69bf2')
    axes[1].axvspan(1e-4,1e-3,color='#81a9df',alpha=.12,label='Candidate band')
    axes[1].set_yscale('log');axes[1].set_xlabel('Learning rate');axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Raw fixed-panel measurements');axes[1].legend()
    fig.tight_layout();fig.savefig(root/'lr-range.png',dpi=170);fig.savefig(root/'lr-range.svg');plt.close(fig)
    samples=[]
    import math
    for requested in (1e-4,2e-4,3e-4,5e-4,1e-3):
        r=min(rows,key=lambda r:abs(math.log(r['lr']/requested)))
        samples.append({'requested_lr':requested,**{k:r[k] for k in ('step','lr','validation_content_loss','validation_structure_loss','validation_loss')}})
    result={'method':'torch-lr-finder 0.2.2 exponential LR range test; upstream steepest-gradient suggestion',
        'history_sha256':sha(root/'history.json'),'config_sha256':sha(root/'config.json'),
        'steps':len(rows),'upstream_steepest_descent_lr':float(suggestion),
        'best_raw_content_loss':best['validation_content_loss'],'best_raw_content_lr':best['lr'],
        'comparisons':samples,'calibration_discarded':h.get('calibration_discarded'),
        'restoration_verified':h.get('restoration_verified'),'stop_reason':h.get('stop_reason') or
            ('upstream divergence threshold' if len(rows)<config['iterations'] else 'completed range'),
        'limits':['Sequential LR range test, not independent fixed-LR trials.',
                  'Short training windows and two fixed held-out interviews; validate long-session trajectory during the actual run.',
                  'Steepest descent and minimum loss are diagnostics, not a WER result or an optimized decay schedule.'],
        'full_training_started':False,'scheduler_selected':False}
    atomic_json(root/'analysis.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
