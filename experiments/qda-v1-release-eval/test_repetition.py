import unittest
from repetition import repetition_period
class Tests(unittest.TestCase):
    def test_non_divisor_period(self):self.assertEqual(repetition_period(list(range(11))*30),11)
    def test_constant(self):self.assertEqual(repetition_period([4]*256),1)
    def test_ordinary_repetition_is_not_aborted(self):self.assertIsNone(repetition_period(list(range(11))*4))
    def test_diverse_tail(self):self.assertIsNone(repetition_period(list(range(1000))))
    def test_loop_after_normal_prefix(self):self.assertEqual(repetition_period(list(range(700))+[8,7,6,5,4,3,2]*48),7)
if __name__=='__main__':unittest.main()
