import json, pathlib, tempfile, unittest
from contracts import atomic_json,sha
from checkpoint_retention import retain


class RetentionTests(unittest.TestCase):
    def make_run(self,base):
        run=pathlib.Path(base)/'runs/r64-full'
        for step in (5,10,15,20,25):
            p=run/f'checkpoint-{step:06d}';p.mkdir(parents=True)
            for name in ('state.pt','adapter_model.safetensors','adapter_config.json'):(p/name).write_bytes(f'{step} {name}'.encode())
            atomic_json(run/'backup-receipts'/(p.name+'.json'),{'destination':str(p),'files':{f.name:sha(f) for f in p.iterdir()}})
        atomic_json(run/'evaluation-50h.json',{'checkpoint':'checkpoint-000015'})
        atomic_json(run/'latest.json',{'checkpoint':'checkpoint-000025'})
        return run

    def test_only_old_nonmilestone_state_is_retired_and_rerun_is_safe(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.make_run(d);retain(run,{'checkpoint-000020','checkpoint-000025'});retain(run,{'checkpoint-000020','checkpoint-000025'})
            for step in (5,15,20,25):self.assertTrue((run/f'checkpoint-{step:06d}'/'state.pt').exists())
            self.assertFalse((run/'checkpoint-000010/state.pt').exists())
            self.assertEqual(len(list(run.glob('checkpoint-*/adapter_model.safetensors'))),5)
            self.assertIn('state_retired_at',json.loads((run/'backup-receipts/checkpoint-000010.json').read_text()))

    def test_corrupt_newer_backup_preserves_old_state(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.make_run(d);(run/'checkpoint-000025/state.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError,'checksum'):retain(run,set())
            self.assertTrue((run/'checkpoint-000010/state.pt').exists())

    def test_state_is_kept_while_remote_copy_is_present(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.make_run(d);retain(run,{'checkpoint-000010'})
            self.assertTrue((run/'checkpoint-000010/state.pt').exists())


if __name__=='__main__':unittest.main()
