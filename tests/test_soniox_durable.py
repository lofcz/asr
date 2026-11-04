import hashlib, importlib.util, json, pathlib, tempfile, unittest
from unittest import mock
SPEC=importlib.util.spec_from_file_location('durable',pathlib.Path(__file__).resolve().parents[1]/'scripts/soniox_durable.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
class FakeAPI:
    def __init__(self):self.posts=[];self.files=[];self.jobs=[];self.deleted=[];self.crash=False;self.reject=None;self.downloads=0
    def obj(self,method,path,**kw):
        if method=='POST':
            self.posts.append(path)
            if self.reject and path=='/transcriptions':raise m.APIError(self.reject)
            if path=='/files':
                f={'id':'f1','size':3,'client_reference_id':kw['data']['client_reference_id']};self.files.append(f);return f
            j={**kw['json'],'id':'t1','status':'completed'};self.jobs.append(j)
            if self.crash: self.crash=False;raise ConnectionError('response lost after server accepted paid job')
            return j
        if method=='GET':return self.jobs[0]
    def find(self,kind,ref):return [x for x in (self.files if kind=='files' else self.jobs) if x['client_reference_id']==ref]
    def call(self,method,path,**kw):
        if method=='DELETE':self.deleted.append(path);return b''
        self.downloads+=1
        return m.encode({'tokens':[{'text':' hello','start_ms':0,'end_ms':400,'speaker':'1'},{'text':' unknown','start_ms':500,'end_ms':900,'speaker':None},{'text':'[music]','start_ms':1200,'end_ms':1500,'is_audio_event':True},{'text':'<end>'}]})
class DurableTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();base=pathlib.Path(self.tmp.name);root=base/'ssd';root.mkdir();(root/'.asr-volume-id').write_text('test');(root/'audio.mp3').write_bytes(b'abc')
        self.store=m.Store(root,root/'dataset',base/'internal','test');self.r={'record_id':m.digest(b'abc'),'sha256':m.digest(b'abc'),'id':'source1','path':'audio.mp3','bytes':3,'seconds':10,'language':'cs','split':'train','context':{}};self.api=FakeAPI();self.rel=pathlib.Path('records')/self.r['record_id']
    def tearDown(self):self.tmp.cleanup()
    def run_job(self):m.process(self.store,self.r,'unused','selectionhash',self.api)
    def test_archive_pairing_and_cleanup(self):
        self.run_job();s=self.store.state(self.r);self.assertEqual(s['phase'],'done');self.assertEqual(self.api.deleted,[])
        derived=json.loads((self.store.out/self.rel/'transcript.json').read_text());self.assertEqual(len(derived['timed_tokens']),3);self.assertEqual(derived['timed_tokens'][1]['speaker'],None);self.assertEqual(derived['untimed_token_indices'],[3])
        self.run_job();self.assertEqual(len(self.api.posts),2);self.assertEqual(len(self.api.deleted),2)
        receipt=json.loads((self.store.out/self.rel/'receipt.json').read_text());self.assertEqual(receipt['audio_sha256'],self.r['sha256']);m.verify_done(self.store,self.r,self.store.state(self.r))
    def test_paid_response_lost_reconciles_without_rebilling(self):
        self.api.crash=True;self.run_job();self.assertEqual(self.store.state(self.r)['phase'],'submission_intent')
        self.run_job();self.assertEqual(self.store.state(self.r)['phase'],'done');self.assertEqual(self.api.posts.count('/transcriptions'),1)
    def test_ambiguous_missing_job_never_resubmits(self):
        s={'phase':'submission_intent','record_id':self.r['record_id'],'file_id':'f1','reference':'asr500-v1-'+self.r['record_id']};self.store.set(self.r,s)
        self.run_job();self.run_job();self.assertEqual(self.api.posts,[]);self.assertEqual(self.store.state(self.r)['phase'],'submission_intent')
    def test_duplicate_remote_matches_hold(self):
        self.api.crash=True;self.run_job();self.api.jobs.append(self.api.jobs[0].copy());self.run_job();self.assertEqual(self.store.state(self.r)['phase'],'submission_intent');self.assertEqual(self.api.posts.count('/transcriptions'),1)
    def test_hash_mismatch_prevents_upload(self):
        (self.store.root/'audio.mp3').write_bytes(b'bad');self.run_job();self.assertEqual(self.api.posts,[])
    def test_missing_ssd_prevents_calls(self):
        (self.store.root/'.asr-volume-id').unlink();self.run_job();self.assertEqual(self.api.posts,[])
    def test_replica_corruption_repaired_without_transcribing(self):
        self.run_job();(self.store.out/self.rel/'soniox.raw.json').write_bytes(b'corrupt');self.run_job();self.assertEqual(len(self.api.posts),2);self.assertEqual((self.store.out/self.rel/'soniox.raw.json').read_bytes(),(self.store.backup/self.rel/'soniox.raw.json').read_bytes())
    def test_lost_both_replicas_no_cleanup(self):
        self.run_job()
        for base in (self.store.out,self.store.backup):(base/self.rel/'soniox.raw.json').unlink()
        self.run_job();self.assertEqual(self.api.deleted,[]);self.assertEqual(len(self.api.posts),2)
    def test_budget_rejection_pauses_paid_submissions(self):
        self.api.reject=402;self.run_job();self.assertTrue(self.store.paused());self.assertEqual(self.store.state(self.r)['phase'],'uploaded');self.run_job();self.assertEqual(self.api.posts.count('/transcriptions'),1)
    def test_rate_limit_is_explicit_rejection_and_safe_to_retry(self):
        self.api.reject=429;self.run_job();self.assertEqual(self.store.state(self.r)['phase'],'uploaded');self.api.reject=None;self.run_job();self.assertEqual(self.store.state(self.r)['phase'],'done');self.assertEqual(len(self.api.jobs),1)
    def test_crash_during_ssd_raw_write_recovers_existing_job(self):
        original=self.store.put
        def broken(rel,data):
            if str(rel).endswith('soniox.raw.json'):m.atomic(self.store.backup/rel,data);raise OSError('SSD disconnected')
            original(rel,data)
        with mock.patch.object(self.store,'put',side_effect=broken):self.run_job()
        self.assertEqual(self.store.state(self.r)['phase'],'archiving');self.run_job();self.assertEqual(self.store.state(self.r)['phase'],'done');self.assertEqual(self.api.posts.count('/transcriptions'),1)
    def test_completed_remote_file_mismatch_cannot_archive(self):
        original=self.api.obj
        def wrong(method,path,**kw):
            result=original(method,path,**kw)
            if method=='GET':return {**result,'file_id':'wrong-audio'}
            return result
        self.api.obj=wrong;self.run_job()
        self.assertEqual(self.store.state(self.r)['phase'],'submitted')
        self.assertFalse((self.store.out/self.rel/'receipt.json').exists())
        self.assertEqual(self.api.deleted,[])
    def test_upload_response_lost_adopts_exact_remote_file(self):
        original=self.api.obj
        def lost(method,path,**kw):
            result=original(method,path,**kw)
            if method=='POST' and path=='/files':raise ConnectionError('upload response lost')
            return result
        self.api.obj=lost;self.run_job()
        self.assertEqual(self.store.state(self.r)['phase'],'upload_intent')
        self.api.obj=original;self.run_job()
        self.assertEqual(self.store.state(self.r)['phase'],'done')
        self.assertEqual(self.api.posts.count('/files'),1)
        self.assertEqual(self.api.posts.count('/transcriptions'),1)
    def test_paginated_reconciliation(self):
        api=m.API('unused');api.obj=mock.Mock(side_effect=[{'files':[],'next_page_cursor':'p2'},{'files':[{'client_reference_id':'ref','id':'f'}],'next_page_cursor':None}]);self.assertEqual(api.find('files','ref')[0]['id'],'f');self.assertEqual(api.obj.call_args.kwargs['params']['cursor'],'p2')
class AdmissionTests(unittest.TestCase):
    def plan(self,active=0,files=None,states=None,rows=None):
        rows=rows or [{'record_id':str(i),'bytes':100} for i in range(20)]
        states=states or [{'phase':'pending'} for r in rows]
        return m.admission_plan(rows,states,{'files':files or [],'transcriptions':[{'status':'processing'}]*active},now=100)
    def test_eighty_cap_includes_processing_and_queued(self):
        tasks,cap=self.plan(active=79);self.assertEqual(len(tasks),1)
        tasks,cap=self.plan(active=80);self.assertEqual(tasks,[])
    def test_upload_batch_is_eight_even_with_large_headroom(self):self.assertEqual(len(self.plan()[0]),8)
    def test_account_storage_reserved_for_whole_batch(self):
        tasks,_=self.plan(files=[{'size':7_999_999_750}]);self.assertEqual(len(tasks),2)
    def test_external_files_count_toward_capacity(self):
        tasks,_=self.plan(files=[{'size':1}]*900);self.assertEqual(tasks,[])
    def test_ambiguous_upload_reserves_slot(self):
        states=[{'phase':'upload_intent','retry_after':200}]+[{'phase':'pending'}]*19
        tasks,_=self.plan(active=79,states=states);self.assertEqual(tasks,[])
    def test_cleanup_and_polling_continue_when_full(self):
        states=[{'phase':'done'},{'phase':'submitted'}]+[{'phase':'pending'}]*18
        tasks,_=self.plan(active=80,files=[{'size':8_000_000_000}],states=states);self.assertEqual(len(tasks),2)
    def test_existing_uploaded_file_cannot_bypass_job_limit(self):
        states=[{'phase':'uploaded'}]*20
        tasks,_=self.plan(active=80,states=states);self.assertEqual(tasks,[])
    def test_existing_and_new_submissions_share_slots(self):
        states=[{'phase':'uploaded'}]*2+[{'phase':'pending'}]*18
        tasks,_=self.plan(active=77,states=states);self.assertEqual(len(tasks),3)
    def test_inflight_poll_respects_interval(self):
        states=[{'phase':'submitted','next_poll':101}]*20
        tasks,_=self.plan(active=20,states=states);self.assertEqual(tasks,[])
if __name__=='__main__':unittest.main()
