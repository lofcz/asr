import {SQL} from 'bun';
import {mkdirSync,writeFileSync,renameSync,openSync,fsyncSync,closeSync,chmodSync,readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const language=process.argv[2]??'sk';
if(!['cs','sk'].includes(language))throw Error('Expected cs or sk');
const root='/run/media/lofcz/ssd_external/GitHub/asr/experiments/production-anecdote-'+language+(language==='cs'?'-clean':'');
const id=language==='sk'?'aad63d12-a699-4d7a-88cb-aaedeb108c05':'ad4a57e0-de6b-4762-9b5f-0ae1475170ef';
mkdirSync(root,{recursive:true,mode:0o700});chmodSync(root,0o700);
function save(name:string,data:Uint8Array|string){const path=root+'/'+name;writeFileSync(path+'.partial',data,{mode:0o600});const fd=openSync(path+'.partial','r');fsyncSync(fd);closeSync(fd);renameSync(path+'.partial',path);const dir=openSync(root,'r');fsyncSync(dir);closeSync(dir);}
function json(name:string,value:unknown){save(name,JSON.stringify(value,null,2)+'\n');}
const sha=(data:Uint8Array|string)=>createHash('sha256').update(data).digest('hex');
let url=process.env.PRODUCTION_DB_URL!;if(url.includes('[YOUR-PASSWORD]'))url=url.replace('[YOUR-PASSWORD]',encodeURIComponent(process.env.PRODUCTION_DB_PW!));
const db=new SQL(url,{max:1,tls:true});
try{
 await db`set default_transaction_read_only=on`;
 const rows=await db`select t.id,t.user_id,t.title,t.status,t.storage_path,t.mime,t.size_bytes,t.duration_ms,t.soniox_transcription_id,t.language_codes,t.segment_count,t.created_at,t.updated_at,t.reviewed_count,t.segments_revision from session_transcripts t join profiles p on p.id=t.user_id where t.id=${id} and p.email='matej.stagl@scioskola.cz'`;
 if(rows.length!==1)throw Error('Expected one owned production transcript');const row=rows[0];
 const segments=await db`select id,idx,speaker,role,start_ms,end_ms,language,raw_text,corrected_text,corrected_at,reverted,words,revision,checked_at,edited_text,edited_at,completed_at from transcript_segments where transcript_id=${id} order by idx`;
 const after=await db`select segments_revision,updated_at from session_transcripts where id=${id}`;
 if(String(after[0].segments_revision)!==String(row.segments_revision)||String(after[0].updated_at)!==String(row.updated_at))throw Error('Transcript changed during snapshot; retry');
 json('production.metadata.json',row);json('production.segments.json',segments);
 let audio:Uint8Array;let audioOrigin:Record<string,unknown>;
 if(/^ta1\/[0-9a-f]{32}$/.test(row.storage_path)){
  const endpoint=new URL('https://app.sciobot.cz/api/v1/ssh/tazatelka/artifacts/download-url');endpoint.searchParams.set('key',process.env.SCIOBOT_KEY!);
  const signed=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({storageKey:row.storage_path,expiresIn:600,download:true}),signal:AbortSignal.timeout(30000)});
  if(!signed.ok)throw Error('Artifact download authorization returned HTTP '+signed.status);
  const ticket=await signed.json() as {url:string};const response=await fetch(ticket.url,{signal:AbortSignal.timeout(600000)});
  if(!response.ok)throw Error('Audio download returned HTTP '+response.status);
  audio=new Uint8Array(await response.arrayBuffer());audioOrigin={method:'production encrypted artifact download'};
 }else{
  if(language!=='cs'||row.title!=='cs-remote-work.wav')throw Error('Unsupported legacy sample');
  const objects=await db`select metadata from storage.objects where bucket_id='session-audio' and name=${row.storage_path}`;
  if(objects.length!==1)throw Error('Production storage object missing');
  const metadata=objects[0].metadata;audio=readFileSync('/run/media/lofcz/ssd_external/GitHub/tazatelka/reference/transcript-fixtures/cs-remote-work.wav');
  const size=16*1024*1024;const hashes=[];
  for(let i=0;i<audio.length;i+=size)hashes.push(createHash('md5').update(audio.subarray(i,i+size)).digest());
  const etag=createHash('md5').update(Buffer.concat(hashes)).digest('hex')+'-'+hashes.length;
  if(etag!==String(metadata.eTag).replaceAll('"','')||audio.length!==Number(metadata.size))throw Error('Local original does not match production multipart ETag and size');
  audioOrigin={method:'Local original upload reused after matching production object size and multipart ETag',multipart_part_bytes:size,verified_etag:etag,production_storage_metadata:metadata};
  json('production.storage-verification.json',audioOrigin);
 }
 if(audio.length!==Number(row.size_bytes))throw Error('Audio byte length mismatch');save('audio.wav',audio);
 const speakers=[...new Set(segments.map(s=>String(s.speaker)))];const mapping=Object.fromEntries(speakers.map((s,i)=>[s,i]));
 for(const variant of ['raw','display']){
  const formatted=segments.map(s=>({Start:s.start_ms/1000,End:s.end_ms/1000,Speaker:mapping[String(s.speaker)],Content:variant==='raw'?s.raw_text:(s.edited_text??(s.reverted||s.corrected_text===null?s.raw_text:s.corrected_text))}));
  json('reference.'+variant+'.json',formatted);save('reference.'+variant+'.txt',formatted.map(s=>`[${s.Start.toFixed(2)}–${s.End.toFixed(2)}] Speaker ${s.Speaker}: ${s.Content}`).join('\n')+'\n');
 }
 const hash=sha(audio);const training=JSON.parse(readFileSync('/run/media/lofcz/ssd_external/GitHub/asr/datasets/releases/qda-496h-v1/selection.json','utf8'));if(training.some((r:any)=>r.sha256===hash))throw Error('Audio overlaps training release');
 const names=['audio.wav','production.metadata.json','production.segments.json','reference.raw.json','reference.display.json','reference.raw.txt','reference.display.txt'];
 json('manifest.json',{purpose:'anecdotal evaluation only; never training',audio_origin:audioOrigin,downloaded_at:new Date().toISOString(),production_transcript_id:id,production_segments_revision:row.segments_revision,title:row.title,audio_sha256:hash,audio_bytes:audio.length,duration_seconds:row.duration_ms/1000,segments:segments.length,speaker_mapping:mapping,reference_quality:'Production Soniox raw segment text plus application display text; not independently human-verified gold',corrected_segments:segments.filter(s=>s.corrected_text!==null).length,edited_segments:segments.filter(s=>s.edited_text!==null).length,reviewed_segments:segments.filter(s=>s.checked_at!==null).length,training_overlap_by_exact_audio_sha256:false,checksums:Object.fromEntries(names.map(n=>[n,sha(readFileSync(root+'/'+n))]))});
 console.log(JSON.stringify({saved_to:root,title:row.title,duration_seconds:row.duration_ms/1000,segments:segments.length,speakers:speakers.length,audio_bytes:audio.length,audio_sha256:hash}));
}finally{await db.close()}
