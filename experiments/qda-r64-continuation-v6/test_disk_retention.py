import json
from pathlib import Path
import tempfile
import unittest
from disk_retention import choose


class ProtectedCheckpoints(unittest.TestCase):
    def test_superseded_backed_directory_respects_transfer_and_milestone(self):
        from superseded_checkpoints import prune_superseded
        import time
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'runs/r64-full';run.mkdir(parents=True)
            for step in (5,10,15,20,25,30):(run/f'checkpoint-{step:06d}').mkdir()
            receipts=run/'backup-receipts';receipts.mkdir()
            for step in (10,20,25):
                name=f'checkpoint-{step:06d}'
                (receipts/(name+'.json')).write_text(json.dumps({'source':str(run/name),
                    'destination':'/verified-ssd/'+name,'verified_at':1,
                    'files':{k:'digest' for k in ('state.pt','adapter_model.safetensors','adapter_config.json')}}))
            (run/'evaluation-50h.json').write_text(json.dumps({'checkpoint':'checkpoint-000015'}))
            leases=run/'backup-in-progress';leases.mkdir()
            lease=leases/'checkpoint-000010.json'
            lease.write_text(json.dumps({'checkpoint':'checkpoint-000010','expires_at':time.time()+60}))
            anchors=['checkpoint-000020','checkpoint-000025']
            self.assertEqual(prune_superseded(run,anchors),[])
            lease.unlink()
            self.assertEqual(prune_superseded(run,anchors),['checkpoint-000010'])
            for step in (5,15,20,25,30):self.assertTrue((run/f'checkpoint-{step:06d}').exists())

    def test_first_latest_evaluation_transfer_are_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for step in range(5, 55, 5):
                p = run / f'checkpoint-{step:06d}'
                p.mkdir()
                (p / 'state.pt').write_text('state')
            (run / 'evaluation-50h.json').write_text(json.dumps({'checkpoint': 'checkpoint-000015'}))
            (run / 'latest.json').write_text(json.dumps({'checkpoint': 'checkpoint-000050'}))
            (run / 'backup-in-progress').mkdir()
            (run / 'backup-in-progress/transfer.json').write_text(json.dumps({'checkpoint': 'checkpoint-000025', 'expires_at': 20}))
            (run / 'backup-receipts').mkdir()
            (run / 'backup-receipts/checkpoint-000035.json').write_text('{}')
            candidates, anchors, protected = choose(run, now=10)
            self.assertEqual([p.name for p in anchors], ['checkpoint-000045', 'checkpoint-000050'])
            self.assertEqual([p.name for p in candidates], ['checkpoint-000010', 'checkpoint-000020', 'checkpoint-000030', 'checkpoint-000035', 'checkpoint-000040'])
            candidates, _, _ = choose(run, now=30)
            self.assertIn('checkpoint-000025', [p.name for p in candidates])

    def test_fewer_than_three_full_states_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for step in (5, 10):
                p = run / f'checkpoint-{step:06d}'
                p.mkdir()
                (p / 'state.pt').write_text('state')
            self.assertEqual(choose(run)[0], [])


if __name__ == '__main__':
    unittest.main()
