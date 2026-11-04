import json,tempfile,pathlib,unittest
from metrics import summarize,event_metrics,read_events
class MetricsTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)
  (self.root/'config.json').write_text(json.dumps({'steps':3}));(self.root/'plan.json').write_text(json.dumps([{'seconds':600,'bucket':600}]*6))
 def tearDown(self):self.tmp.cleanup()
 def event(self,n):return {'event':'step','step':n,'time':n*30,'lr':1e-5,'ranks':[{'rank':0,'seconds':600,'target_tokens':100,'loss':1.,'wall_seconds':25,'grad_norm':.2,'peak_gpu_gib':8},{'rank':1,'seconds':600,'target_tokens':300,'loss':3.,'wall_seconds':25,'grad_norm':.2,'peak_gpu_gib':8}]}
 def test_restart_removes_uncheckpointed_future_progress(self):
  events=[self.event(1),self.event(2),self.event(3),{'event':'validation','step':3,'time':91,'loss':.2},{'event':'step_start','step':2,'time':100},self.event(2)]
  s=summarize(self.root,events,now=120);self.assertEqual(s['step'],2);self.assertAlmostEqual(s['hours_done'],2400/3600);self.assertEqual(len(s['history']),2);self.assertEqual(s['validation'],[]);self.assertEqual(s['rollbacks'][0]['from'],3)
 def test_token_weighted_loss(self):self.assertEqual(event_metrics(self.event(1),1)['train/loss'],2.5)
 def test_eta_includes_unseen_long_stage_cost(self):
  plan=[{'seconds':600,'bucket':600}]*2+[{'seconds':3600,'bucket':3600}]*4;(self.root/'plan.json').write_text(json.dumps(plan));s=summarize(self.root,[self.event(1)],now=40);self.assertGreater(s['eta_seconds'],330);self.assertNotIn(3600,s['eta_basis']['measured_buckets'])
 def test_partial_last_log_record_does_not_break_monitor(self):
  (self.root/'events.jsonl').write_text(json.dumps(self.event(1))+'\n{"event":');self.assertEqual(len(read_events(self.root)),1)
if __name__=='__main__':unittest.main()
