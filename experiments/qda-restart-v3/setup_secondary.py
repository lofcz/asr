"""Prepare an isolated secondary ASR environment; no modification of training packages."""
import pathlib
import subprocess
import sys

HERE=pathlib.Path(__file__).resolve().parent
VENV=HERE/'audit-venv'
if not (VENV/'bin/python').exists():subprocess.run([sys.executable,'-m','venv',str(VENV)],check=True)
subprocess.run([str(VENV/'bin/python'),'-m','pip','install','faster-whisper==1.2.1','rapidfuzz==3.14.1'],check=True)
(HERE/'audit-environment.txt').write_text(subprocess.check_output([str(VENV/'bin/python'),'-m','pip','freeze'],text=True))
subprocess.run([str(VENV/'bin/python'),'-c',
    "from huggingface_hub import snapshot_download; snapshot_download('dropbox-dash/faster-whisper-large-v3-turbo',revision='0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf',local_dir='secondary-model',allow_patterns=['*.json','*.bin','*.txt'])"],cwd=HERE,check=True)
