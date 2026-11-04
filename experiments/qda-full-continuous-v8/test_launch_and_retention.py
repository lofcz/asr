import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import approval
import checkpoint_policy
from contracts import atomic_json,sha


class ApprovalTests(unittest.TestCase):
    def test_exact_package_must_be_approved(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(approval,'HERE',Path(directory)):
            root=Path(directory)
            for name in ('run-spec.json','plan.json'):atomic_json(root/name,{'version':8})
            atomic_json(root/'prepared.json',{'version':8,'runtime_sha256':{}})
            with self.assertRaisesRegex(RuntimeError,'Awaiting'):approval.require_approval()
            atomic_json(root/'approval.json',{'approved':True,'files':{name:sha(root/name) for name in ('run-spec.json','plan.json','prepared.json')}})
            with patch('runtime_state.runtime_hashes',return_value={}):
                self.assertTrue(approval.require_approval()['approved'])
            atomic_json(root/'plan.json',{'version':9})
            with self.assertRaisesRegex(RuntimeError,'changed'):approval.require_approval()


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'runs/r64-full';self.run.mkdir(parents=True)
        for step in (5,10,15,20):
            folder=self.run/f'checkpoint-{step:06d}';folder.mkdir()
            for name in ('state.pt','adapter_model.safetensors','adapter_config.json'):(folder/name).write_bytes(f'{step}-{name}'.encode())
        atomic_json(self.root/'plan.json',{'probe_pairs':[[{'window_id':'training'}]]})
        (self.root/'cache').mkdir()
        for name in ('training.features.npz','training.features.json','heldout.features.npz'):(self.root/'cache'/name).write_bytes(b'cached')
        for name,value in [('HERE',self.root),('verify_resume',lambda p:{'files':{f.name:sha(f) for f in p.iterdir()}})]:
            mock=patch.object(checkpoint_policy,name,value);mock.start();self.addCleanup(mock.stop)

    def test_keeps_first_latest_two_and_heldout_features(self):
        checkpoint_policy.prune(self.run)
        self.assertTrue((self.run/'checkpoint-000005').exists())
        self.assertFalse((self.run/'checkpoint-000010').exists())
        self.assertTrue((self.run/'checkpoint-000015/state.pt').exists())
        self.assertTrue((self.run/'checkpoint-000020/state.pt').exists())
        self.assertTrue((self.root/'cache/heldout.features.npz').exists())

    def test_milestone_needs_exact_verified_backup(self):
        atomic_json(self.run/'evaluation-50h.json',{'checkpoint':'checkpoint-000010'})
        checkpoint_policy.prune(self.run)
        old=self.run/'checkpoint-000010';self.assertTrue(old.exists())
        receipt={'source':str(old),'destination':'/verified-ssd/checkpoint-000010','verified_at':1,
                 'files':{p.name:sha(p) for p in old.iterdir()}}
        atomic_json(self.run/'backup-receipts/checkpoint-000010.json',receipt)
        checkpoint_policy.prune(self.run)
        self.assertFalse(old.exists())

    def test_wrong_namespace_is_never_pruned(self):
        with self.assertRaisesRegex(RuntimeError,'namespace'):checkpoint_policy.prune(self.root/'other/runs/r64-full')


if __name__=='__main__':unittest.main()
