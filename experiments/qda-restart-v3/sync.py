"""SSD owns human reviews; mlserver owns derived audit results. Never delete either."""
import pathlib
import subprocess
import time
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'datasets/releases/qda-clean-v3'
REMOTE='/data/work/asr-datasets/qda-clean-v3'

while True:
    try:
        for name in ('reviews','acoustic-reviews','review-history'):
            reviews=DATA/name
            if reviews.exists():
                subprocess.run(['ssh','mlserver','mkdir','-p',REMOTE+'/'+name],check=True,timeout=30)
                subprocess.run(['rsync','-a',str(reviews)+'/',f'mlserver:{REMOTE}/{name}/'],check=True,timeout=120)
        subprocess.run(['rsync','-a','--exclude=reviews','--exclude=acoustic-reviews','--exclude=review-history','--exclude=review-audio',f'mlserver:{REMOTE}/',str(DATA)+'/'],check=True,timeout=300)
        subprocess.run(['rsync','-a','--include=curation-status.json','--include=experiment-status.json','--include=ready.json','--include=*-result.json','--include=*-stopped.json','--include=*-10h.json','--include=lr-*.json','--include=preflight-*.json','--include=audit-environment.txt','--exclude=*',
                        'mlserver:/data/work/asr-experiments/qda-restart-v3/',str(HERE)+'/'],check=True,timeout=60)
        for folder in ('runs','markers','evaluations'):
            subprocess.run(['rsync','-a','--exclude=*.building*','--include=*/','--include=*.json','--include=*.jsonl','--exclude=*',f'mlserver:/data/work/asr-experiments/qda-restart-v3/{folder}/',str(HERE/folder)+'/'],timeout=60,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        # Only atomically published checkpoints: preserve optimizer/RNG as well as adapters.
        subprocess.run(['rsync','-a','--exclude=*.building*','--include=*/','--include=adapter_model.safetensors','--include=state.pt','--exclude=*',
                        'mlserver:/data/work/asr-experiments/qda-restart-v3/runs/',str(HERE/'runs')+'/'],timeout=300,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError,OSError) as error:print(type(error).__name__+': '+str(error),flush=True)
    time.sleep(30)
