"""CPU verification of the frozen recipe and one actual contextual input."""
import json
from pathlib import Path
import time
import unittest
from contracts import atomic_json,sha
from gates import demand
from runtime_state import runtime_hashes

HERE=Path(__file__).resolve().parent
TESTS=('test_combined_dataset','test_targets','test_scoring',
       'test_scheduler_resume','test_objective','test_launch_and_retention')


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(TESTS))
    demand(result.wasSuccessful(),'Package regression tests failed')
    plan=json.loads((HERE/'plan.json').read_text())
    rows=[row for pair in plan['probe_pairs'] for row in pair]
    contexts=[r for r in rows if r.get('customized_context')]
    context_check={'windows_with_context':len(contexts)}
    if contexts:
        import torch
        from model_paths import model_paths
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
        from prepare import materialize_input,tokenized_target
        path,tok=model_paths()
        processor=VibeVoiceASRProcessor.from_pretrained(path,language_model_pretrained_name=tok)
        row=min(contexts,key=lambda r:r['duration'])
        saved,_=materialize_input(row,processor,plan['cache_contract_sha256'])
        x=torch.load(HERE/'cache'/(row['window_id']+'.input.pt'),weights_only=True)
        body,target,content=tokenized_target(processor,row)
        n=len(x['ids'])-len(target)
        prefix=processor.tokenizer.decode(x['ids'][:n],skip_special_tokens=False)
        demand(row['customized_context'] in prefix,'Customized context is absent from the model input')
        demand((x['labels'][:n]==-100).all() and not x['content_mask'][:n].any(),'Prompt leaks into supervised loss')
        demand(torch.equal(x['labels'][n:],torch.tensor(target)) and not x['mask'][n:].any(),'Target/input boundary differs')
        demand(json.loads(body)==row['target'],'Contextual sample target differs from authoritative labels')
        context_check.update(passed=True,window_id=row['window_id'],input_sha256=saved['input_sha256'],
            source_audio_sha256=row['audio_sha256'],prefix_fully_masked=True,reference_unchanged=True)
    atomic_json(HERE/'verification.json',{'passed':True,'verified_at':time.time(),
        'tests_run':result.testsRun,'test_files':{name+'.py':sha(HERE/(name+'.py')) for name in TESTS},
        'runtime_sha256':runtime_hashes(),'plan_sha256':sha(HERE/'plan.json'),
        'context_input':context_check,'optimizer_updates':0})
    print(json.dumps({'passed':True,'tests_run':result.testsRun,'context_input':context_check}))


if __name__=='__main__':main()
