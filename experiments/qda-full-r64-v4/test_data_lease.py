"""The preflight subprocess can read a completed dataset while writers stay out."""
import fcntl, pathlib, subprocess, sys, tempfile, unittest
from gates import lease_data


class DataLeaseTests(unittest.TestCase):
    def child_can_lock(self,root,exclusive):
        script='''
import fcntl, pathlib, sys
with (pathlib.Path(sys.argv[1])/'.curation.lock').open('a') as handle:
    try:
        fcntl.flock(handle,(fcntl.LOCK_EX if sys.argv[2]=='exclusive' else fcntl.LOCK_SH)|fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(2)
'''
        code=subprocess.run([sys.executable,'-c',script,str(root),'exclusive' if exclusive else 'shared']).returncode
        self.assertIn(code,(0,2))
        return code==0

    def test_preflight_read_lease_after_downgrade_keeps_mutators_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder)
            with lease_data(root,exclusive=True) as lock:
                self.assertFalse(self.child_can_lock(root,False))
                self.assertFalse(self.child_can_lock(root,True))
                fcntl.flock(lock,fcntl.LOCK_SH)
                self.assertTrue(self.child_can_lock(root,False))
                self.assertFalse(self.child_can_lock(root,True))
            self.assertTrue(self.child_can_lock(root,True))


if __name__=='__main__':unittest.main()
