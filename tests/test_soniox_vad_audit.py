import sys,pathlib,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from soniox_vad_audit import intervals, difference, union
class AuditTests(unittest.TestCase):
 def test_no_speech_is_not_an_event_label(self):self.assertEqual(intervals([0.]*100,3.2),[])
 def test_short_burst_rejected(self):self.assertEqual(intervals([0.]*10+[.9]*3+[0.]*10,.736),[])
 def test_hysteresis_and_bounds(self):
  r=intervals([.9]*20+[0.]*10, .96);self.assertEqual(len(r),1);self.assertEqual(r[0][0],0);self.assertAlmostEqual(r[0][1],.67)
 def test_gaps_preserve_overlap_evidence(self):
  self.assertEqual(difference([[0,10]],[[1,3],[4,6],[8,12]]),[[0,1],[3,4],[6,8]])
  self.assertEqual(union([[0,4],[2,8],[8,10]]),[[0,10]])
 def test_tail_not_dropped(self):self.assertEqual(intervals([.9]*20,.625),[[0,.625]])
if __name__=='__main__':unittest.main()
