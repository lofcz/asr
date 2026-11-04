"""Content verification of a copied release; no API access and no mutation of labels."""
import argparse,concurrent.futures,hashlib,json,pathlib,os,tempfile,time

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--workers',type=int,default=2);a=p.parse_args();root=pathlib.Path(a.root);expected={}
 for name in ('checksums.original-audio.json','checksums.training.json','checksums.metadata.json'):
  for rel,digest in json.loads((root/name).read_bytes()).items():
   if rel in expected and expected[rel]!=digest:raise ValueError('Conflicting checksum manifests')
   expected[rel]=digest
 selected=json.loads((root/'selection.json').read_bytes())
 if sha(root/'selection.json')!=(root/'selection.sha256').read_text().strip():raise ValueError('Selection checksum mismatch')
 for r in selected:
  folder=root/'records'/r['record_id'];receipt=json.loads((folder/'receipt.json').read_bytes());state=json.loads((folder/'state.json').read_bytes())
  if receipt['record_id']!=r['record_id'] or receipt['audio_sha256']!=r['sha256'] or receipt['transcription_id']!=state['transcription_id']:raise ValueError('Paid receipt/source mismatch')
  if sha(folder/'receipt.json')!=state['receipt_sha256']:raise ValueError('Paid receipt checksum mismatch')
  for name,digest in receipt['artifacts_sha256'].items():expected[str((folder/name).relative_to(root))]=digest
 started=time.monotonic();errors=[]
 def check(item):
  rel,digest=item;path=root/rel
  try:
   if sha(path)!=digest:return rel+': checksum mismatch'
  except Exception as e:return rel+': '+str(e)
  return None
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for i,error in enumerate(pool.map(check,expected.items())):
   if error:errors.append(error)
   if (i+1)%250==0:print(json.dumps({'verified':i+1,'total':len(expected),'errors':len(errors)}),flush=True)
 result={'status':'verified' if not errors else 'failed','checked_at':time.time(),'files':len(expected),'source_records':len(selected),'elapsed_seconds':time.monotonic()-started,'errors':errors,'selection_sha256':sha(root/'selection.json'),'training_manifest_sha256':sha(root/'training/sources.jsonl'),'training_catalog_sha256':sha(root/'training/catalog.sqlite3'),'checks':'Original and canonical audio, immutable paid pairing receipts and all referenced artifacts, normalized labels, windows, indexes, provenance and release metadata.'}
 fd,tmp=tempfile.mkstemp(dir=root)
 with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,root/'verification.json');print(json.dumps(result),flush=True)
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
