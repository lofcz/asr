import unittest,tempfile,pathlib,json
from unittest.mock import patch
import controller
from history import atomic
class Recovery(unittest.TestCase):
 def test_timeout_records_failure_advances_and_resumes(self):
  with tempfile.TemporaryDirectory() as folder:
   base=pathlib.Path(folder);request=base/'request.json'
   atomic(request,{'id':'checkpoint-400','step':400,'hours':150.3,'final':False})
   atomic(base/'policy.json',{'enabled':True,'next_hours':150})
   with patch.object(controller,'BASE',base),patch.object(controller,'REQUEST',request),patch.object(controller.subprocess,'run') as command,patch.object(controller,'build'):
    controller.recover()
    command.assert_called_once_with(['systemctl','--user','start','qda496-training.service'],check=True)
    controller.recover();self.assertEqual(command.call_count,1)
   self.assertFalse(request.exists());self.assertEqual(json.loads((base/'policy.json').read_text())['next_hours'],200)
   self.assertTrue((base/'runs/checkpoint-400/failed.json').exists())
 def test_successful_final_resumes_to_mark_training_complete(self):
  with tempfile.TemporaryDirectory() as folder:
   base=pathlib.Path(folder);request=base/'request.json'
   atomic(request,{'id':'final','step':705,'hours':466.7,'final':True})
   atomic(base/'policy.json',{'enabled':True,'next_hours':500})
   atomic(base/'runs/final/completed.json',{'status':'complete'})
   with patch.object(controller,'BASE',base),patch.object(controller,'REQUEST',request),patch.object(controller.subprocess,'run'),patch.object(controller,'build'):controller.recover()
   self.assertFalse(json.loads((base/'policy.json').read_text())['enabled'])
   self.assertFalse((base/'runs/final/failed.json').exists())
if __name__=='__main__':unittest.main()
