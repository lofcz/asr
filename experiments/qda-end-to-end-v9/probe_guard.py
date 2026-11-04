"""Temporarily reserve idle GPU 0 without interrupting the existing long decode.

Only the orchestrator is SIGSTOP'ed. Existing inference children continue.
A finally block plus systemd ExecStopPost releases it even if the probe fails.
"""
import argparse,json,os,pathlib,signal,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/data/work/asr-experiments/qda-full-continuous-v8')
LEASE=HERE/'probe-lease.json'

def identity(pid):
 p=pathlib.Path('/proc')/str(pid)
 return {'pid':pid,'starttime':(p/'stat').read_text().rsplit(')',1)[1].split()[19],
  'cwd':os.readlink(p/'cwd'),'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()}

def release():
 if not LEASE.exists():return
 value=json.loads(LEASE.read_text())
 try:
  if identity(value['parent']['pid'])==value['parent']:
   os.kill(value['parent']['pid'],signal.SIGCONT)
 except FileNotFoundError:pass
 value['released_at']=time.time();LEASE.write_text(json.dumps(value,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--release',action='store_true');a=p.parse_args()
 if a.release:release();return
 stage=json.loads((BASE/'experiment-status.json').read_text())['stage']
 if not stage.startswith('long-evaluation-'):raise RuntimeError('Only reserve during long evaluation')
 parents=[]
 for entry in pathlib.Path('/proc').iterdir():
  if not entry.name.isdigit():continue
  try:
   value=identity(int(entry.name))
   if value['cwd']==str(BASE) and value['cmdline'].endswith('run_full.py'):parents.append(value)
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 if len(parents)!=1:raise RuntimeError('Ambiguous orchestrator identity')
 parent=parents[0];value={'parent':parent,'reserved_at':time.time(),'max_seconds':900,'gpu':0,'existing_decoder_children_untouched':True}
 LEASE.write_text(json.dumps(value,indent=2)+'\n')
 def stop(signum,frame):raise KeyboardInterrupt('Probe service stopped')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  os.kill(parent['pid'],signal.SIGSTOP)
  gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  if int(gpu.strip())>500:raise RuntimeError('GPU 0 is occupied; leave evaluation running')
  subprocess.run([sys.executable,'-u',str(HERE/'gradient_probe.py'),'--gpu','0'],cwd=HERE,check=True,timeout=840)
 finally:release()
if __name__=='__main__':main()
