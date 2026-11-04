import contextlib
import json
import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch
from gates import validate_reference,development_reference
from analyze import compare


def scored(rid,errors):
    return {'window_id':rid,'record_id':rid,'complete':True,'metrics':{'number_normalized_reference_words':1000,
            'number_normalized_errors':errors,'der_details':{'scored_speaker_time':600,'missed_speaker_time':3,'falarm_speaker_time':2,'speaker_error_time':1}}}


class SafetyTests(unittest.TestCase):
    def test_recorded_quality_stop_cannot_resume_updates(self):
        import launch
        with tempfile.TemporaryDirectory() as folder,patch.object(launch,'HERE',pathlib.Path(folder)),patch.object(launch,'run') as run:
            out=pathlib.Path(folder)/'runs/r16-probe';out.mkdir(parents=True);(out/'stopped.json').write_text('{}')
            with self.assertRaises(launch.TrialRejected):launch.train(16,'probe',3e-5,'r16-probe')
            run.assert_not_called()

    def test_one_capacity_quality_failure_does_not_cancel_the_other(self):
        import launch
        with tempfile.TemporaryDirectory() as folder,contextlib.ExitStack() as stack:
            root=pathlib.Path(folder);(root/'ready.json').write_text(json.dumps({'runtime_sha256':{},'plan_sha256':'same','admission_sha256':'same'}))
            for name,value in [('HERE',root),('require_old_jobs_stopped',lambda:None),('check_data',lambda *a,**k:{}),
                               ('lease_data',lambda *a,**k:contextlib.nullcontext()),('sha',lambda p:'same'),('runtime_hashes',lambda:{}),('markers',lambda n:{})]:
                stack.enter_context(patch.object(launch,name,value))
            option=stack.enter_context(patch.object(launch,'option',side_effect=[launch.TrialRejected('trajectory regressed'),[]]))
            stack.enter_context(patch('sys.argv',['launch.py','both']));launch.main()
            self.assertEqual([c.args[0] for c in option.call_args_list],[16,64])
            self.assertEqual(json.loads((root/'r16-probe-stopped.json').read_text())['reason'],'trajectory regressed')

    def test_soniox_reference_does_not_require_human_review(self):
        target=[{'Start':0.,'End':4.,'Speaker':0,'Content':'Dobrý den.'}]
        with tempfile.TemporaryDirectory() as folder:
            reference,quality,fingerprint=development_reference(pathlib.Path(folder),{'window_id':'abc','target':target,'duration':10.})
        self.assertEqual(reference,target)
        self.assertEqual(quality,'Soniox pseudo-reference')
        self.assertIsNone(fingerprint)

    def test_nonspeech_pause_is_allowed_but_missing_speech_is_not(self):
        from select_windows import quality
        policy=json.loads((pathlib.Path(__file__).parent/'policy.json').read_text())
        label={'words':[{'start':0.,'end':3.,'speaker':0,'token_indices':[0],'confidence':.99},
                        {'start':8.,'end':20.,'speaker':1,'token_indices':[1],'confidence':.99}], 'token_confidences':[.99,.99]}
        acoustic={'windows':[{'start':0.,'end':20.,'scores':{'music':0.,'noise':0.},'clipped_fraction':0.}]}
        source={'target_language_fraction':1.}
        good=quality(source,label,{'speech_intervals_seconds':[[0.,3.],[8.,20.]]},acoustic,0.,20.,policy)
        self.assertEqual(good['reasons'],[])
        self.assertTrue(good['unresolved_nonspeech'])
        bad=quality(source,label,{'speech_intervals_seconds':[[0.,20.]]},acoustic,0.,20.,policy)
        self.assertIn('untranscribed_speech_run',bad['reasons'])

    def test_invalid_json_fields_never_become_scored_transcripts(self):
        for ref in ([{'Start':float('nan'),'End':2,'Speaker':0,'Content':'Text'}],
                    [{'Start':0,'End':12,'Speaker':0,'Content':'Text'}],
                    [{'Start':0,'End':2,'Content':'Text'}]):
            with self.assertRaises(RuntimeError):validate_reference(ref,10)
        validate_reference([{'Start':0,'End':2,'Content':'[Music]'}],10)

    def test_bootstrap_identical_predictions_are_not_improvement(self):
        rows=[scored(str(i),30+i) for i in range(6)];result=compare(rows,rows,draws=200)
        self.assertEqual(result['paired_recording_bootstrap_95']['number_normalized_wer'],[0.,0.])
        self.assertFalse(result['screening_goal_met'])

    def test_bootstrap_does_not_accept_a_missing_or_repaired_output(self):
        rows=[scored(str(i),30) for i in range(6)];changed=[scored(str(i),20) for i in range(6)];changed[-1]['complete']=False
        with self.assertRaises(ValueError):compare(rows,changed,draws=100)
        with self.assertRaises(ValueError):compare(rows,changed[:-1],draws=100)

    def test_failed_admission_cannot_launch_training_or_preflight(self):
        import launch
        with tempfile.TemporaryDirectory() as folder,patch.object(launch,'HERE',pathlib.Path(folder)),patch.object(launch,'require_old_jobs_stopped'),patch.object(launch,'check_data',side_effect=RuntimeError('mismatched audio')),patch.object(launch,'run') as run,patch.object(launch,'parallel') as parallel,patch('sys.argv',['launch.py','both']):
            with self.assertRaisesRegex(RuntimeError,'mismatched audio'):launch.main()
            run.assert_not_called();parallel.assert_not_called()

    def test_lora_scaling_is_fixed_alpha32_for_both_capacities(self):
        import torch
        import model
        class Fake(torch.nn.Module):
            def __init__(self,rank):
                super().__init__();self.peft_config={'default':types.SimpleNamespace(lora_alpha=rank*2)}
                for name in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):
                    m=torch.nn.Module();m.lora_A=torch.nn.ModuleDict({'default':torch.nn.Linear(8,rank,bias=False)});m.lora_B=torch.nn.ModuleDict({'default':torch.nn.Linear(rank,8,bias=False)});m.lora_B['default'].weight.data.zero_();m.lora_alpha={'default':rank*2};m.scaling={'default':2.};setattr(self,name,m)
            def named_parameters(self,*a,**k):
                for name,p in super().named_parameters(*a,**k):yield 'model.'+name,p
        for rank in (16,64):
            fake=Fake(rank)
            with patch.object(model,'load_model',return_value=fake):result=model.fresh_model(rank)
            self.assertEqual(result.peft_config['default'].lora_alpha,32)
            self.assertEqual(result.q_proj.scaling['default'],32/rank)
        fake=Fake(16);fake.q_proj.lora_B['default'].weight.data.fill_(1)
        with patch.object(model,'load_model',return_value=fake):
            with self.assertRaisesRegex(RuntimeError,'not vanilla'):model.fresh_model(16)


if __name__=='__main__':unittest.main()
