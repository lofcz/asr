import json,pathlib,tempfile,unittest
from contracts import atomic_json
from superseded_checkpoints import prune_superseded


class SupersededTests(unittest.TestCase):
    def make_run(self,base):
        run=pathlib.Path(base)/'runs/r64-full'
        for step in (5,10,15,20,25,30,35,40):(run/f'checkpoint-{step:06d}').mkdir(parents=True)
        for step in (20,30):
            name=f'checkpoint-{step:06d}'
            atomic_json(run/'backup-receipts'/(name+'.json'),{'source':str(run/name),'destination':'/ssd/'+name,'verified_at':1,
                'files':{k:'verified' for k in ('state.pt','adapter_model.safetensors','adapter_config.json')}})
        atomic_json(run/'evaluation-50h.json',{'checkpoint':'checkpoint-000015'})
        atomic_json(run/'latest.json',{'checkpoint':'checkpoint-000040'})
        return run

    def test_only_intermediate_snapshot_with_two_newer_backups_is_retired(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.make_run(d);retired=prune_superseded(run,['checkpoint-000020','checkpoint-000030'])
            self.assertEqual(retired,['checkpoint-000010'])
            for step in (5,15,20,25,30,35,40):self.assertTrue((run/f'checkpoint-{step:06d}').exists())
            self.assertTrue((run/'superseded/checkpoint-000010.json').exists())
            self.assertEqual(prune_superseded(run,['checkpoint-000020','checkpoint-000030']),[])

    def test_missing_or_retired_full_anchor_blocks_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.make_run(d)
            with self.assertRaisesRegex(RuntimeError,'Two full'):prune_superseded(run,['checkpoint-000030'])
            p=run/'backup-receipts/checkpoint-000020.json';r=json.loads(p.read_text());r['state_retired_at']=2;atomic_json(p,r)
            with self.assertRaisesRegex(RuntimeError,'full verified'):prune_superseded(run,['checkpoint-000020','checkpoint-000030'])
            self.assertTrue((run/'checkpoint-000010').exists())


if __name__=='__main__':unittest.main()
