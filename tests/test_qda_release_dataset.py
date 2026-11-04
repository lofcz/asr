import sys,pathlib,tempfile,unittest,json,sqlite3,wave,array,os
from unittest import mock
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import qda_release_dataset as d
class ReleaseTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name);(self.root/'training/labels').mkdir(parents=True)
  with wave.open(str(self.root/'audio.wav'),'wb') as f:
   f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(b''.join(array.array('h',[level]*24000).tobytes() for level in (0,3000,6000,9000)))
  self.doc={'record_id':'r','split':'train','audio_sha256':d.sha(self.root/'audio.wav'),'segments':[{'start':1,'end':3,'text':' actual words','session_speaker':'speaker7','is_event':False}],'customized_context':[]};(self.root/'training/labels/r.json').write_text(json.dumps(self.doc))
  self.window={'segment_lo':0,'segment_hi':1,'source_start':1,'source_end':3,'audio_duration':2,'session_to_window_speaker':{'speaker7':0}}
  db=sqlite3.connect(self.root/'training/catalog.sqlite3');db.executescript('CREATE TABLE sources(record_id TEXT,audio_path TEXT,audio_sha256 TEXT,label_path TEXT,label_sha256 TEXT);CREATE TABLE windows(window_id TEXT,record_id TEXT,split TEXT,duration REAL,views_json TEXT,window_json TEXT);');db.execute('INSERT INTO sources VALUES(?,?,?,?,?)',('r','audio.wav',self.doc['audio_sha256'],'training/labels/r.json',d.sha(self.root/'training/labels/r.json')));db.execute('INSERT INTO windows VALUES(?,?,?,?,?,?)',('w','r','train',2,json.dumps([{'nominal_seconds':120,'oversized':False}]),json.dumps(self.window)));db.commit();db.close()
 def tearDown(self):self.tmp.cleanup()
 def test_label_and_real_audio_use_identical_offsets(self):
  ds=d.ReleaseDataset(self.root,'train',120)
  with mock.patch.dict(os.environ,{'FFMPEG':'/usr/bin/ffmpeg'}):sample,audio=ds.decode(0)
  self.assertEqual(len(audio),48000);self.assertEqual(sample['segments'][0]['start'],0);self.assertEqual(sample['segments'][0]['end'],2);self.assertEqual(sample['segments'][0]['speaker'],0);self.assertAlmostEqual(float(audio[1000:2000].mean()),3000/32768,places=5);self.assertAlmostEqual(float(audio[30000:31000].mean()),6000/32768,places=5);self.assertEqual(json.loads(d.target_json(sample))[0]['Content'],' actual words')
 def test_tampered_label_rejected(self):
  (self.root/'training/labels/r.json').write_text('{}')
  with self.assertRaises(ValueError):d.ReleaseDataset(self.root,'train',120).label(0)
 def test_tampered_audio_rejected(self):
  (self.root/'audio.wav').write_bytes(b'wrong')
  with self.assertRaises(ValueError):d.ReleaseDataset(self.root,'train',120).decode(0)
 def test_holdout_cannot_enter_training_split(self):self.assertEqual(len(d.ReleaseDataset(self.root,'validation',120)),0)
if __name__=='__main__':unittest.main()
