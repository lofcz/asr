import importlib.util,pathlib,sys,unittest,copy,tempfile,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import soniox_postprocess as p

def token(text,a,b,speaker='A',**kw):return {'text':text,'start_ms':a*1000,'end_ms':b*1000,'speaker':speaker,'confidence':.95,'language':'cs',**kw}
class PostprocessTests(unittest.TestCase):
 def test_subwords_and_disfluencies_preserved_exactly(self):
  t=[token(' A',0,1),token('hoj',1,2),token(', ehm',2,3),token(' ehm',3,4),{'text':'<end>'}]
  s,v,m,e=p.normalize(t,5);self.assertEqual(''.join(x['text'] for x in s),' Ahoj, ehm ehm');self.assertEqual(sum(len(x['token_indices']) for x in s),4);self.assertEqual(m['explicitly_excluded_controls'][0]['token_index'],4)
 def test_unknown_speaker_blocks_does_not_invent(self):
  s,v,m,e=p.normalize([token(' speech',0,1,None)],2);self.assertIsNone(s[0]['speaker']);self.assertTrue(any(x['kind']=='speakerless_speech' for x in e))
 def test_untimed_real_words_not_silently_dropped(self):
  s,v,m,e=p.normalize([{'text':'lost speech'}],2);self.assertTrue(any(x['severity']=='blocking' for x in e))
 def test_nonfinite_and_negative_timing_block(self):
  for a,b in [(float('nan'),1),(-1,1),(2,1),(1,10)]:
   _,_,_,e=p.normalize([token(' word',a,b)],5);self.assertTrue(any(x['severity']=='blocking' for x in e))
 def test_overlap_preserved_and_counted(self):
  s,v,m,e=p.normalize([token(' a',0,3,'A'),token(' b',1,4,'B')],5);_,_=p.quality(v,s,m,5,None,'cs');self.assertEqual(m['multi_speaker_overlap_seconds'],2)
 def test_gaps_not_automatically_event_tags(self):
  s,v,m,e=p.normalize([token(' hello',1,2)],10);vad={'full_decode_verified':True,'speech_intervals_seconds':[[1,2],[5,7]],'speech_fraction':.3};cov,e=p.quality(v,s,m,10,vad,'cs');self.assertTrue(cov['speech_without_transcript']);self.assertFalse(any(x['text'] in p.EVENTS for x in s));self.assertTrue(any(x['kind']=='non_speech_needs_classification' for x in e))
 def test_window_boundaries_never_cut_overlap(self):
  s=[{'start':i*30.,'end':i*30.+29,'text':' word','session_speaker':'A','speaker':0,'is_event':False,'token_indices':[i]} for i in range(41)]
  s.append({'start':110,'end':145,'text':' overlap','session_speaker':'B','speaker':1,'is_event':False,'token_indices':[41]})
  ordered,windows=p.windows(s,1230,'rid')
  for w in windows:
   a,b=w['source_start'],w['source_end'];self.assertFalse(any(x['start']<a<x['end'] or x['start']<b<x['end'] for x in s));label=p.candidate_label(ordered,w);self.assertTrue(all(0<=x['start']<=x['end']<=w['audio_duration']+.001 for x in label['segments']))
  for bucket in (120,600,1800,3600):
   view=[w for w in windows if any(v['nominal_seconds']==bucket for v in w['curriculum_views'])];indices=[i for w in view for x in ordered[w['segment_lo']:w['segment_hi']] for i in x['token_indices']];self.assertEqual(sorted(indices),list(range(42)))
 def test_large_overlap_component_not_cut_arbitrarily(self):
  s=[{'start':0,'end':500,'text':'hello','session_speaker':'A'}];o,w=p.windows(s,600,'r');self.assertEqual(len(w),1);self.assertTrue(w[0]['curriculum_views'][0]['oversized'])
 def test_speaker_remap_local_and_context_defaults_empty(self):
  s=[{'start':5,'end':6,'text':' hello','session_speaker':'99'}];o,w=p.windows(s,10,'r');label=p.candidate_label(o,w[0]);self.assertEqual(label['segments'][0]['speaker'],0);self.assertEqual(label['customized_context'],[])
 def test_stale_review_cannot_approve(self):
  _,_,_,approved,e=p.apply_review({'tokens':[]},{'input_digest':'old'},'new',10);self.assertFalse(approved);self.assertEqual(e[0]['kind'],'stale_review')
 def test_override_requires_exact_original_hash_and_preserves_raw(self):
  raw={'tokens':[token(' wrong',0,1)]};original=copy.deepcopy(raw);review={'input_digest':'x','reviewed_by':'reviewer','reviewed_at':'date','token_overrides':[{'token_index':0,'original_sha256':p.digest(p.encode(raw['tokens'][0])),'changes':{'text':' corrected'},'reason':'listened'}]};t,_,_,approved,e=p.apply_review(raw,review,'x',10);self.assertEqual(t[0]['text'],' corrected');self.assertEqual(raw,original);self.assertFalse(approved)
 def test_all_three_reviews_required_and_context_provenance(self):
  review={'input_digest':'x','reviewed_by':'reviewer','reviewed_at':'date','speech_verified':True,'diarization_verified':True,'acoustic_events_verified':True};self.assertTrue(p.apply_review({'tokens':[]},review,'x',10)[3]);review['customized_context']=['secret answer'];self.assertFalse(p.apply_review({'tokens':[]},review,'x',10)[3])
 def test_accepted_background_event_retains_overlapping_speech(self):
  raw={'tokens':[token(' words',1,2)]};review={'input_digest':'x','reviewed_by':'r','reviewed_at':'date','accepted_events':[{'start':0,'end':3,'text':'[Music]','evidence':'listened'}]};t,ev,_,_,e=p.apply_review(raw,review,'x',4);self.assertEqual(t,raw['tokens']);self.assertEqual(ev[0]['text'],'[Music]');self.assertEqual(e,[])
 def test_control_and_event_candidates_never_became_spoken_text(self):
  s,v,m,e=p.normalize([token('[Music]',0,1,None,is_audio_event=True),token(' words',1,2),'bad'][:2],3);self.assertEqual(m['speech_token_count'],1);self.assertTrue(s[0]['is_event'])
 def test_verified_music_is_clipped_without_forcing_long_speech_windows(self):
  speech=[{'start':i*30.,'end':i*30.+20,'text':' word','session_speaker':'A','is_event':False} for i in range(20)]
  music={'start':0,'end':600,'text':'[Music]','session_speaker':None,'is_event':True}
  ordered,windows=p.windows(speech+[music],600,'music')
  short=[w for w in windows if any(v['nominal_seconds']==120 for v in w['curriculum_views'])]
  self.assertGreater(len(short),1)
  for w in short:
   events=[s for s in p.candidate_label(ordered,w)['segments'] if s['text']=='[Music]'];self.assertEqual(len(events),1);self.assertEqual(events[0]['start'],0);self.assertAlmostEqual(events[0]['end'],w['audio_duration'])

class IntegrationTests(unittest.TestCase):
 def setUp(self):
  from soniox_durable import Store
  self.tmp=tempfile.TemporaryDirectory();root=pathlib.Path(self.tmp.name)/'root';root.mkdir();(root/'.asr-volume-id').write_text('test');(root/'source.mp3').write_bytes(b'abc')
  self.store=Store(root,root/'dataset',pathlib.Path(self.tmp.name)/'backup','test');sha=p.digest(b'abc');self.r={'record_id':sha,'sha256':sha,'path':'source.mp3','bytes':3,'seconds':20,'split':'train','language':'cs','id':'source','show':'show'};self.rel=pathlib.Path('records')/sha
  self.raw=p.encode({'id':'job1','tokens':[token(' word',0,1)]});self.store.put(self.rel/'soniox.raw.json',self.raw)
  receipt={'audio_sha256':sha,'transcription_id':'job1','artifacts_sha256':{'soniox.raw.json':p.digest(self.raw)}};self.store.obj(self.rel/'receipt.json',receipt);self.store.set(self.r,{'phase':'done','record_id':sha,'transcription_id':'job1','receipt_sha256':p.digest(p.encode(receipt))})
  self.vad={'audio_sha256':sha,'full_decode_verified':True,'decoded_seconds':20,'speech_intervals_seconds':[[0,1]],'speech_fraction':.05,'probabilities_sha256':p.digest(b'probs')};self.store.obj(self.rel/'vad.json',self.vad);self.store.put(self.rel/'vad.probabilities.npy.gz',b'probs');self.db=p.connect(self.store.backup/'working.sqlite3')
 def tearDown(self):self.db.close();self.tmp.cleanup()
 def test_progressive_review_revisions_and_readiness_gate(self):
  from soniox_dataset import load_window
  self.assertTrue(p.process(self.store,self.db,self.r,'selection'));self.assertFalse(p.process(self.store,self.db,self.r,'selection'));p.publish(self.store,self.db,1)
  window=self.db.execute('SELECT window_id FROM windows').fetchone()[0]
  with self.assertRaises(ValueError):load_window(self.store.out,window)
  sample=load_window(self.store.out,window,True);self.assertEqual(sample['label']['segments'][0]['text'],' word');self.assertEqual(sample['split'],'train')
  current=json.loads((self.store.out/'postprocess/v1/current'/f"{self.r['record_id']}.json").read_text());review={'input_digest':current['base_input_digest'],'reviewed_by':'reviewer','reviewed_at':'2026-10-22T12:00:00Z','speech_verified':True,'diarization_verified':True,'acoustic_events_verified':True}
  path=self.store.out/'reviews'/f"{self.r['record_id']}.json";path.parent.mkdir();path.write_bytes(p.encode(review));self.assertTrue(p.process(self.store,self.db,self.r,'selection'));p.publish(self.store,self.db,1)
  self.assertTrue(load_window(self.store.out,window)['training_ready']);self.assertEqual((self.store.out/self.rel/'soniox.raw.json').read_bytes(),self.raw)
  self.vad['speech_fraction']=.06;self.store.obj(self.rel/'vad.json',self.vad);self.assertTrue(p.process(self.store,self.db,self.r,'selection'));p.publish(self.store,self.db,1)
  self.assertEqual(self.db.execute('SELECT status FROM recordings').fetchone()[0],'quarantined');self.assertEqual(self.db.execute('SELECT count(*) FROM windows WHERE training_ready=1').fetchone()[0],0)
 def test_source_identity_mismatch_quarantines(self):
  (self.store.root/'source.mp3').write_bytes(b'bad');p.process(self.store,self.db,self.r,'selection');self.assertEqual(self.db.execute('SELECT status FROM recordings').fetchone()[0],'quarantined')
class ExportTests(unittest.TestCase):
 def test_export_crops_audio_and_writes_matching_label_receipt(self):
  import wave,array,subprocess
  from unittest import mock
  import soniox_dataset as d
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);source=root/'source.wav'
   with wave.open(str(source),'wb') as f:
    f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b''.join(array.array('h',[level]*16000).tobytes() for level in (0,3000,6000,9000)))
   sample={'window_id':'window','record_id':'record','revision':'revision','split':'test','training_ready':True,'audio_source':{'path':'source.wav','sha256':p.file_hash(source),'start':1,'end':3},'label':{'audio_duration':2,'segments':[{'speaker':0,'text':'fixture','start':0,'end':2}],'customized_context':[]}}
   with mock.patch.object(d,'load_window',return_value=sample):d.export_window(root,root,'window',root/'out')
   label=json.loads((root/'out/sample.json').read_bytes());receipt=json.loads((root/'out/receipt.json').read_bytes());self.assertEqual(label['audio_path'],'audio.flac');self.assertEqual(receipt['source']['start'],1);self.assertAlmostEqual(receipt['decoded_duration'],2,places=2)
   pcm=subprocess.check_output(['ffmpeg','-v','error','-i',str(root/'out/audio.flac'),'-f','s16le','-']);samples=array.array('h');samples.frombytes(pcm);self.assertEqual(len(samples),48000);self.assertAlmostEqual(sum(samples[1000:2000])/1000,3000,delta=2);self.assertAlmostEqual(sum(samples[30000:31000])/1000,6000,delta=2)
if __name__=='__main__':unittest.main()
