"""Verify actual scheduler/optimizer restoration rather than a duplicate LR formula."""
import copy, json, pathlib, tempfile, unittest
import torch
from schedule import make_scheduler, write_preview


class ResumeTests(unittest.TestCase):
    def test_interruption_preserves_lr_sequence_and_parameter_updates(self):
        spec=json.loads((pathlib.Path(__file__).parent/'run-spec.json').read_text())
        def make():
            p=torch.nn.Parameter(torch.tensor([.25],dtype=torch.float64))
            o=torch.optim.AdamW([p],lr=spec['learning_rate'],weight_decay=0.)
            return p,o,make_scheduler(o,spec,100)
        p,o,s=make();used=[];saved=None
        for step in range(100):
            used.append(o.param_groups[0]['lr']);o.zero_grad();(p.square().sum()).backward();o.step();s.step()
            if step==42:saved=(p.detach().clone(),copy.deepcopy(o.state_dict()),copy.deepcopy(s.state_dict()))
        q,qo,qs=make()
        with torch.no_grad():q.copy_(saved[0])
        qo.load_state_dict(saved[1]);qs.load_state_dict(saved[2]);resumed=[]
        for step in range(43,100):
            resumed.append(qo.param_groups[0]['lr']);qo.zero_grad();q.square().sum().backward();qo.step();qs.step()
        self.assertEqual(resumed,used[43:]);self.assertTrue(torch.equal(p,q))
        self.assertEqual(qs.last_epoch,100)

    def test_preview_uses_whole_plan_and_is_independent_of_evaluations(self):
        spec=json.loads((pathlib.Path(__file__).parent/'run-spec.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder)
            (root/'run-spec.json').write_text(json.dumps(spec))
            normalized=write_preview(root)
            self.assertFalse(normalized['exact_plan'])
            self.assertIsNone(normalized['audio_hours'])
            # Unequal duration pairs make steps and audio hours visibly different.
            plan={'probe_pairs':[[{'duration':60*(i+1)}] for i in range(100)]}
            (root/'plan.json').write_text(json.dumps(plan))
            actual=write_preview(root)
            self.assertTrue(actual['exact_plan']);self.assertEqual(actual['optimizer_steps'],100)
            self.assertEqual(actual['warmup_steps'],10)
            self.assertAlmostEqual(actual['audio_hours'],sum(range(1,101))/60)
            self.assertEqual(actual['points'][0]['lr'],0.)
            self.assertEqual(actual['points'][10]['lr'],spec['learning_rate'])
            self.assertEqual(actual['terminal_lr'],0.)
            self.assertAlmostEqual(actual['points'][50]['lr'],spec['learning_rate']*5/9)
            self.assertTrue(all(a['lr']>=b['lr'] for a,b in zip(actual['points'][10:],actual['points'][11:])))
            spec['evaluation_every_audio_hours']=1
            (root/'run-spec.json').write_text(json.dumps(spec))
            self.assertEqual(write_preview(root)['points'],actual['points'])


if __name__=='__main__':unittest.main()
