import copy,pathlib,types,unittest
from unittest.mock import patch
import numpy as np
from prepare import decode,pcm_crop
from train_restart import compatible_resume,HERE


class DecoderRecoveryTests(unittest.TestCase):
 def row(self):return {'window_id':'test','audio_file':'audio.m4a','audio_sha256':'expected','source_start':.5,'duration':1.}
 def result(self,wave):return types.SimpleNamespace(stdout=wave.astype('<f4').tobytes(),stderr=b'')
 def test_valid_old_crop_is_bit_identical_and_decoded_once(self):
  wave=np.arange(24000,dtype=np.float32)
  with patch('prepare.sha',return_value='expected'),patch('prepare.subprocess.run',return_value=self.result(wave)) as run:
   np.testing.assert_array_equal(decode(self.row()),wave);self.assertEqual(run.call_count,1)
 def test_timestamp_overrun_uses_exact_full_pcm_samples(self):
  source=np.arange(72000,dtype=np.float32);overrun=np.zeros(26400,dtype=np.float32)
  with patch('prepare.sha',return_value='expected'),patch('prepare.subprocess.run',side_effect=[self.result(overrun),self.result(source)]) as run,patch('prepare.atomic_json') as receipt:
   result=decode(self.row());np.testing.assert_array_equal(result,source[12000:36000])
   self.assertEqual(run.call_count,2);self.assertNotIn('-ss',run.call_args.args[0]);receipt.assert_called_once()
 def test_missing_source_audio_is_never_padded(self):
  with self.assertRaisesRegex(RuntimeError,'does not contain'):pcm_crop(np.zeros(10000),.5,1.)
 def test_nonfinite_audio_remains_a_hard_failure(self):
  with patch('prepare.sha',return_value='expected'),patch('prepare.subprocess.run',return_value=self.result(np.full(24000,np.nan))) as run:
   with self.assertRaisesRegex(RuntimeError,'Nonfinite'):decode(self.row())
   self.assertEqual(run.call_count,1)


class ResumeCompatibilityTests(unittest.TestCase):
 def configs(self):
  old={'lr':.0003,'rank':64,'plan_sha256':'plan','schedule':'linear','warmup_steps':132,'code_sha256':'old','runtime_sha256':{'old':'old'}}
  new={**old,'code_sha256':'new','runtime_sha256':{str(HERE/'train_restart.py'):'new'},'runtime_repair_sha256':'receipt'}
  return old,new,{'old_config':old,'new_runtime_sha256':new['runtime_sha256']}
 def test_exact_known_checkpoint_restores_under_documented_runtime_repair(self):
  old,new,repair=self.configs();self.assertTrue(compatible_resume(old,new,repair));self.assertTrue(compatible_resume(new,new,repair))
 def test_lr_plan_rank_and_schedule_changes_are_rejected(self):
  old,new,repair=self.configs()
  for key,value in [('lr',.001),('rank',16),('plan_sha256','different'),('schedule','constant'),('warmup_steps',0)]:
   changed={**new,key:value};self.assertFalse(compatible_resume(old,changed,repair),key)
  self.assertFalse(compatible_resume(old,new,None));self.assertFalse(compatible_resume({**old,'lr':.001},new,repair))


if __name__=='__main__':unittest.main()
