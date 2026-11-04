import unittest
from supervise_full import classify

class SupervisionTests(unittest.TestCase):
 def test_retry_only_known_transient_failures(self):
  for text in ('CUDA out of memory','ConnectionResetError: peer','NCCL watchdog timeout'):
   self.assertTrue(classify(text,1))
  self.assertTrue(classify('Killed',-9))
 def test_integrity_and_numeric_errors_override_secondary_nccl_errors(self):
  for text in ('Crop duration/nonfinite mismatch','Artifact hash mismatch','Resume checkpoint/config mismatch','Audio identity changed'):
   self.assertFalse(classify(text+'\nNCCL peer terminated',1))
  self.assertFalse(classify('Unknown runtime bug',1))

if __name__=='__main__':unittest.main()
