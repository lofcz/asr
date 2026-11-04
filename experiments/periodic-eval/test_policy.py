import unittest
from policy import due
class Milestones(unittest.TestCase):
 def test_boundary_and_resume(self):
  p={'enabled':True,'next_hours':150}
  self.assertFalse(due(149.99,370,705,p));self.assertTrue(due(150.4,371,705,p))
  p['next_hours']=200;self.assertFalse(due(151.4,372,705,p));self.assertTrue(due(200.1,400,705,p))
 def test_final(self):
  self.assertTrue(due(466.7,705,705,{'enabled':True,'next_hours':500}))
  self.assertFalse(due(466.7,705,705,{'enabled':False,'next_hours':500}))
if __name__=='__main__':unittest.main()
