import unittest
from balance import balance
class Balance(unittest.TestCase):
 def test_exact_suffix_permutation(self):
  rows=[{'window_id':str(i),'bucket':1800 if i<9 else 3600,'seconds':i+1} for i in range(20)]
  lengths={r['window_id']:{'tokens':100+(int(r['window_id'])*71)%101,'target_tokens':40} for r in rows}
  out=balance(rows,2,lengths)
  self.assertEqual(out[:4],rows[:4]);self.assertCountEqual(out[4:],rows[4:]);self.assertEqual([r['bucket'] for r in rows],[r['bucket'] for r in out]);self.assertEqual(out,balance(rows,2,lengths))
 def test_finished(self):
  self.assertEqual(balance([],0,{}),[])
if __name__=='__main__':unittest.main()
