"""Durable, resumable preparation followed by both user-authorized fresh probes."""
import json
import pathlib
import subprocess
import time
from contracts import atomic_json

HERE=pathlib.Path(__file__).resolve().parent
PY='/data/work/asr-pilot/venv/bin/python'


def main():
    try:
        if not (HERE/'ready.json').exists():
            subprocess.run([PY,'-u','launch.py','prepare'],cwd=HERE,check=True)
        subprocess.run([PY,'-u','launch.py','both'],cwd=HERE,check=True)
        atomic_json(HERE/'experiment-status.json',{'stage':'completed','running':False,'updated_at':time.time()})
    except Exception as error:
        path=HERE/'experiment-status.json'
        previous=json.loads(path.read_text()) if path.exists() else {}
        atomic_json(path,{**previous,'stage':'stopped','previous_stage':previous.get('stage'),'running':False,'error':str(error),'updated_at':time.time()})
        raise


if __name__=='__main__':main()
