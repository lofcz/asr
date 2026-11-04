import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import future_data_repair as repair


class FutureDataRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.original_turn = {'Start': 0., 'End': 2.71, 'Speaker': 0, 'Content': ' Earlier sentence.'}
        self.restored = {'Start': 1.68, 'End': 2.24, 'Speaker': 1, 'Content': ' Tak to pro mě bylo.'}
        self.old = {'window_id': 'old', 'record_id': 'source', 'split': 'train', 'audio_sha256': 'audio',
                    'source_start': 1400., 'source_end': 1402.71, 'duration': 2.71, 'audio_duration': 1500.,
                    'target': [self.original_turn], 'target_tokens': 20, 'total_tokens': 100, 'content_tokens': 5,
                    'target_json_sha256': 'before', 'label_sha256': 'label', 'label_path': 'old.json', 'reuse_root': '/old'}
        self.new = {**self.old, 'window_id': 'new', 'amended_from_window_id': 'old',
                    'target': [self.original_turn, self.restored], 'target_tokens': 53, 'total_tokens': 133,
                    'content_tokens': 12, 'target_json_sha256': 'after', 'label_sha256': 'label2', 'label_path': 'new.json'}
        self.new.pop('reuse_root')
        self.plan = {'probe_pairs': [[{'window_id': 'already-seen', 'duration': 10}], [self.old]],
                     'probe_target_tokens': 21, 'validation': ['unchanged'], 'train_hours': .1}
        label = {'segments': [{'start': 1400., 'end': 1402.71, 'speaker': '2', 'text': ' Earlier sentence.'},
                              {'start': 1401.68, 'end': 1402.24, 'speaker': '1', 'text': ' Tak to pro mě bylo.'}]}
        (self.root / 'label.json').write_text(json.dumps(label))
        self.value = {'version': 1, 'update': 2, 'rank': 0, 'original_row_sha256': repair.digest(self.old),
                      'original_window_id': 'old', 'replacement': self.new, 'restored_turn': self.restored,
                      'target_token_delta': 33, 'corrected_label_file': str(self.root / 'label.json')}
        self.amendment = self.root / 'amendment.json'
        self.amendment.write_text(json.dumps(self.value))
        self.receipt = {'amendment_path': str(self.amendment)}

    def test_only_future_row_changes_no_plan_reordering_or_mutation(self):
        before = copy.deepcopy(self.plan)
        result = repair.apply_repair(self.plan, self.receipt)
        self.assertEqual(self.plan, before)
        self.assertEqual(result['probe_pairs'][0], before['probe_pairs'][0])
        self.assertEqual(result['validation'], before['validation'])
        self.assertEqual(result['train_hours'], before['train_hours'])
        self.assertEqual(len(result['probe_pairs']), len(before['probe_pairs']))
        self.assertEqual(result['probe_target_tokens'], before['probe_target_tokens'] + 33)
        self.assertEqual(result['probe_pairs'][1][0]['target'][-1], self.restored)

    def test_rejects_audio_cut_existing_word_speaker_or_cache_changes(self):
        for field, value in [('source_start', 1399.), ('audio_sha256', 'wrong'), ('reuse_root', '/stale')]:
            item = copy.deepcopy(self.value)
            item['replacement'][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                repair.validate_replacement(self.plan, item)
        for field, value in [('Content', 'Changed previous text.'), ('Speaker', 1)]:
            item = copy.deepcopy(self.value)
            item['replacement']['target'][0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                repair.validate_replacement(self.plan, item)

    def test_rejects_zero_or_outside_timestamp(self):
        for start, end in [(2.71, 2.71), (2., 3.)]:
            item = copy.deepcopy(self.value)
            item['replacement']['target'][-1].update(Start=start, End=end)
            item['restored_turn'] = item['replacement']['target'][-1]
            with self.subTest(start=start), self.assertRaises(RuntimeError):
                repair.validate_replacement(self.plan, item)

    def test_old_checkpoint_must_precede_changed_example(self):
        receipt = {**self.receipt, 'old_config': {'lr': .0003}}
        repair.validate_resume_step(1, receipt['old_config'], receipt)
        with self.assertRaises(RuntimeError):
            repair.validate_resume_step(2, receipt['old_config'], receipt)

    def test_resume_only_allows_exact_metadata_and_token_delta(self):
        old = {'lr': .0003, 'optimizer': 'adamw', 'steps': 3110, 'warmup_steps': 311,
               'target_tokens': 1000, 'runtime_sha256': {}, 'code_sha256': 'old'}
        receipt = {'old_config': old, 'new_runtime_sha256': {str(repair.HERE / 'train_restart.py'): 'new'},
                   'target_token_delta': 33}
        current = {**old, 'runtime_sha256': receipt['new_runtime_sha256'], 'code_sha256': 'new',
                   'runtime_repair_sha256': 'proof', 'target_tokens': 1033}
        with patch.object(repair, 'sha', return_value='proof'):
            self.assertTrue(repair.compatible_resume(old, current, receipt))
            for key, wrong in [('lr', .0001), ('warmup_steps', 312), ('steps', 100), ('optimizer', 'sgd'),
                               ('target_tokens', 1034), ('code_sha256', 'arbitrary')]:
                with self.subTest(key=key):
                    self.assertFalse(repair.compatible_resume(old, {**current, key: wrong}, receipt))


if __name__ == '__main__':
    unittest.main()
