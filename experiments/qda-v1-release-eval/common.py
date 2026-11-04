import pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/data/work/asr-experiments/qda-full-continuous-v8')
sys.path[:0]=[str(BASE),'/data/work/asr-pilot/long-context-pilot','/data/work/asr-pilot/VibeVoice','/data/work/asr-evaluation/periodic/code']
from contracts import sha,atomic_json

def code_hashes():
    return {str(p):sha(p) for p in [HERE/'worker.py',HERE/'repetition.py',BASE/'scoring.py',BASE/'final_report.py',BASE/'gates.py',pathlib.Path('/data/work/asr-evaluation/periodic/code/number_wer.py')]}
