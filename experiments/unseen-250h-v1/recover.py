"""Supplementary scores with a narrow, logged JSON timestamp repair.

Raw inference outputs and raw result.json validity remain unchanged. Apply the
same rule to either model, and never repair transcript wording or speaker IDs.
"""
import json
import pathlib
import re

import rescore
from contracts import atomic_json, sha

HERE = pathlib.Path(__file__).resolve().parent
RULE = 'all-zero-malformed-timestamp-v1'
ZERO_TIME = re.compile(r'(?P<prefix>[{,]\s*"(?:Start|End)"\s*:\s*)(?P<value>0(?:\.0){2,})(?=\s*[,}])')


def main():
    panel = json.loads((HERE / 'panel.json').read_text())
    repaired = []
    for variant in ('vanilla', 'finetuned'):
        for sample in panel['samples']:
            source = HERE / variant / sample['sample_id']
            receipt = source / 'result.json'
            if not receipt.exists():
                continue
            raw = json.loads(receipt.read_text())
            if raw.get('complete') or raw.get('termination') != 'eos':
                continue
            text = (source / 'output.txt').read_text()
            try:
                json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                continue  # No repair for semantic, timestamp-bound or other failures.
            changes = []
            def replace(match):
                changes.append({'offset': match.start('value'), 'before': match['value'], 'after': '0.0'})
                return match['prefix'] + '0.0'
            corrected = ZERO_TIME.sub(replace, text)
            if not changes:
                continue
            try:
                hyp = json.loads(corrected)
                assert isinstance(hyp, list) and hyp
                speech, events = [], []
                for segment in hyp:
                    assert isinstance(segment, dict) and {'Start', 'End', 'Content'} <= segment.keys()
                    assert isinstance(segment['Content'], str)
                    assert 0 <= float(segment['Start']) <= float(segment['End']) <= sample['audio_duration'] + 1
                    if re.fullmatch(r'\s*\[[^\[\]]+\]\s*', segment['Content']):
                        events.append(segment)
                    else:
                        assert 'Speaker' in segment
                        speech.append(segment)
            except (AssertionError, KeyError, ValueError, TypeError):
                continue
            reference = HERE / 'inputs' / sample['sample_id'] / 'reference.json'
            assert sha(reference) == raw['reference_sha256']
            target = HERE / 'recovered' / variant / sample['sample_id']
            target.mkdir(parents=True, exist_ok=True)
            metrics = rescore.scoring.metrics(json.loads(reference.read_text()), hyp, target,
                                             sample['audio_duration'], sample['language'])
            assert 'der_250ms' in metrics
            result = {**raw, 'complete': True, 'raw_output_valid': False,
                      'scoring_status': 'syntax_recovered', 'recovery_rule': RULE,
                      'recovery_code_sha256': sha(HERE / 'recover.py'),
                      'original_output_sha256': sha(source / 'output.txt'),
                      'repairs': changes, 'acoustic_event_count': len(events),
                      'speech_segment_count': len(speech),
                      'metrics': json.loads(json.dumps(metrics, default=str))}
            result['raw_error'] = result.pop('error')
            (target / 'output.recovered.txt').write_text(corrected)
            atomic_json(target / 'hypothesis.json', hyp)
            atomic_json(target / 'result.json', result)
            repaired.append({'sample': sample['sample_id'], 'variant': variant,
                             'repairs': changes, 'normalized_wer': metrics['number_normalized_wer'],
                             'der': metrics['der_250ms']})
    print(json.dumps({'rule': RULE, 'recovered': repaired}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
