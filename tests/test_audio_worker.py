import importlib.util,json,pathlib,sqlite3,subprocess,tempfile,time,unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('worker',pathlib.Path(__file__).parents[1]/'scripts/audio_worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
ID='SU6bjDUe4Rk'
class DurableQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name);self.state=self.root/'state';self.db=w.connect(self.state);self.queue=self.root/'queue.txt'
    def tearDown(self):self.db.close();self.tmp.cleanup()
    def seed(self):
        self.queue.write_text(ID+'\n');w.ingest(self.db,self.queue);return w.claim(self.db)
    def test_partial_append_and_duplicate_replay(self):
        self.queue.write_bytes((ID+'\n'+ID+'\n{"id":"abcdefghijk"').encode());self.assertEqual(w.ingest(self.db,self.queue),1)
        with self.queue.open('ab') as f:f.write(b'}\n')
        self.assertEqual(w.ingest(self.db,self.queue),1);self.assertEqual(w.ingest(self.db,self.queue),0)
    def test_malformed_line_does_not_block_valid_work(self):
        self.queue.write_text('{broken}\n'+ID+'\n');self.assertEqual(w.ingest(self.db,self.queue),1)
        self.assertEqual(self.db.execute('SELECT count(*) FROM ingestion_errors').fetchone()[0],1)
    def test_queue_replacement_and_removal_do_not_lose_jobs(self):
        self.seed();self.queue.write_text('abcdefghijk\n');w.ingest(self.db,self.queue);self.queue.unlink()
        self.assertEqual(w.ingest(self.db,self.queue),0);self.assertEqual(self.db.execute('SELECT count(*) FROM jobs').fetchone()[0],2)
    def test_claim_is_exclusive_and_restart_recovers(self):
        first=self.seed();other=w.connect(self.state)
        self.assertIsNone(w.claim(other));other.close();w.recover(self.db)
        self.assertEqual(w.claim(self.db)['id'],first['id'])
    def test_concurrent_claims_do_not_duplicate_jobs(self):
        import concurrent.futures
        self.queue.write_text(ID+'\nabcdefghijk\n12345678901\n');w.ingest(self.db,self.queue)
        def take():
            db=w.connect(self.state)
            try:
                row=w.claim(db);return row['id'] if row else None
            finally:db.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            claimed=list(pool.map(lambda _:take(),range(8)))
        self.assertEqual(len([x for x in claimed if x]),3)
        self.assertEqual(len(set(x for x in claimed if x)),3)
    def test_failure_is_retained_with_backoff(self):
        job=self.seed();w.fail(self.db,job,'temporary failure');self.assertIsNone(w.claim(self.db))
        row=self.db.execute('SELECT * FROM jobs').fetchone();self.assertEqual(row['status'],'retry');self.assertEqual(row['attempts'],1);self.assertGreater(row['next_attempt'],time.time())
    def test_crash_after_publish_before_database_commit(self):
        job=self.seed();stage=self.root/'data/youtube/staging'/ID;stage.mkdir(parents=True)
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=1','-y',str(stage/(ID+'.wav'))],check=True)
        config={'root':str(self.root)}
        with patch.object(w,'complete',side_effect=RuntimeError('simulated power loss before DB commit')):
            with self.assertRaises(RuntimeError):w.publish(self.db,job,config)
        self.assertTrue((self.root/'data/youtube/audio'/(ID+'.wav')).exists())
        w.recover(self.db);job=w.claim(self.db);self.assertTrue(w.publish(self.db,job,config))
        self.assertEqual(self.db.execute('SELECT status FROM jobs').fetchone()[0],'done')
    def test_bad_media_is_quarantined_not_marked_done(self):
        job=self.seed();stage=self.root/'data/youtube/staging'/ID;stage.mkdir(parents=True);(stage/(ID+'.webm')).write_text('broken')
        self.assertFalse(w.publish(self.db,job,{'root':str(self.root)}));self.assertTrue(list((self.root/'data/youtube/quarantine').iterdir()))
    def test_backup_is_consistent_and_readable(self):
        self.seed();w.backup(self.db,self.state);b=sqlite3.connect(self.state/'backups/latest.sqlite3')
        self.assertEqual(b.execute('PRAGMA integrity_check').fetchone()[0],'ok');self.assertEqual(b.execute('SELECT count(*) FROM jobs').fetchone()[0],1);b.close()
    def test_parser_only_accepts_individual_youtube_videos(self):
        self.assertEqual(w.parse_line('https://youtu.be/'+ID)['id'],ID)
        self.assertEqual(w.parse_line('https://www.youtube.com/watch?v='+ID+'&t=2')['id'],ID)
        with self.assertRaises(ValueError):w.parse_line('https://example.com/watch?v='+ID)
        with self.assertRaises(ValueError):w.parse_line('https://youtube.com/playlist?list=abc')
    def test_wrong_volume_marker_is_rejected(self):
        (self.root/'.asr-volume-id').write_text('different-volume')
        with patch.object(w.os.path,'ismount',return_value=True):
            self.assertFalse(w.storage_ready({'root':str(self.root),'mount':str(self.root),'volume_id':'expected'}))
    def test_missing_completed_file_is_requeued(self):
        job=self.seed();receipt={'path':'data/youtube/audio/missing.webm','bytes':4000,'duration_seconds':1}
        w.complete(self.db,job,receipt);w.audit(self.db,self.root)
        self.assertEqual(self.db.execute('SELECT status FROM jobs').fetchone()[0],'retry')
    def test_missing_storage_does_not_create_directories(self):
        root=self.root/'missing';self.assertFalse(w.storage_ready({'root':str(root),'mount':str(root),'volume_id':'test'}));self.assertFalse(root.exists())
if __name__=='__main__':unittest.main()
