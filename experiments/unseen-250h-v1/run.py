"""Evaluation only. Never resumes baseline training or starts continuation experiments."""
import pathlib,subprocess,os,fcntl,sys,json,time
HERE=pathlib.Path(__file__).resolve().parent
PY='/data/work/asr-pilot/venv/bin/python'
def main():
 lock=(HERE/'work.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 logs=HERE/'logs';logs.mkdir(exist_ok=True)
 with (logs/'prepare.log').open('a') as f:subprocess.run([PY,'-u',str(HERE/'prepare.py')],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
 handles=[];jobs=[]
 try:
  for gpu,variant in enumerate(('vanilla','finetuned')):
   f=(logs/(variant+'.log')).open('a');handles.append(f)
   jobs.append(subprocess.Popen([PY,'-u',str(HERE/'evaluate.py'),'--variant',variant,'--gpu',str(gpu)],cwd=HERE,stdout=f,stderr=subprocess.STDOUT))
  codes=[p.wait() for p in jobs]
  if any(codes):raise RuntimeError('Evaluation process failed: '+str(codes))
  subprocess.run([PY,str(HERE/'report.py')],cwd=HERE,check=True)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate();p.wait()
  for f in handles:f.close()
if __name__=='__main__':main()
