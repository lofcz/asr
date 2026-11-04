"""Offline regressions for exhausted old drafts, bounded retries, and cached work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate_scripts as g
from core import digest, encode


class ScriptRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = dict(id='offline-test', seed=1, speaker_count=1,
                            language=['cs'], source_files=[], topic='', vocabulary='',
                            planned_seconds=60)
        self.contract = dict(target_words=100, mandatory_word_cost=10,
                             terms=[dict(id='test-term', name='ScioBot', family='scio',
                                         forms=['ScioBot'], minimum=10,
                                         required_forms=['ScioBot'], description='Test fixture')])
        self.contract['sha256'] = digest(encode(self.contract))
        self.rel = 'scripts-v4/offline-test/part-000'
        self.old = dict(continuity_notes='', turns=[dict(speaker=0, text=' '.join(['ScioBot']*72+['slovo']*20))])
        self.corrected = dict(continuity_notes='', turns=[dict(speaker=0, text=' '.join(['ScioBot']*10+['slovo']*180))])
        for target, value in [('DATA', self.root), ('read', self.read),
                              ('save', self.save), ('put', self.put),
                              ('vocabulary_contract', lambda *_: self.contract)]:
            p = patch.object(g, target, value)
            p.start()
            self.addCleanup(p.stop)

    def read(self, rel, default=None):
        path = self.root / rel
        return json.loads(path.read_bytes()) if path.exists() else default

    def put(self, rel, data):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def save(self, rel, data):
        self.put(rel, encode(data))

    def old_attempts(self, policy=None):
        for i in range(1, 9):
            stem = self.rel+f'.attempt-{i}'
            self.save(stem+'.output.json', self.old)
            self.save(stem+'.result.json', {'exit_code': 0})
            self.save(stem+'.intent.json', {'repair_policy': policy})

    def test_old_exhausted_drafts_get_one_corrected_budget(self):
        self.old_attempts()
        calls = []
        owner = self
        class Process:
            returncode = 0
            def __init__(self, cmd, **kwargs):
                self.output = Path(cmd[cmd.index('--output-last-message')+1])
            def communicate(self, prompt, timeout):
                calls.append(prompt.decode())
                self.output.write_bytes(encode(owner.corrected))
        with patch.object(g.subprocess, 'Popen', Process):
            result = g.make_part(self.session, 0, 20, True)
        self.assertEqual(len(calls), 1)
        self.assertIn('Final spoken-word target: 190 words', calls[0])
        self.assertEqual(result['words'], 190)
        self.assertEqual(result['vocabulary_counts']['test-term'], 10)
        self.assertEqual(self.read(self.rel+'.attempt-1.output.json'), self.old)
        self.assertEqual(self.read(self.rel+'.attempt-9.intent.json')['prompt_sha256'], digest(calls[0].encode()))

    def test_current_policy_cannot_retry_forever(self):
        self.old_attempts(g.REPAIR_POLICY)
        with patch.object(g.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(ValueError, 'attempts exhausted'):
                g.make_part(self.session, 0, 20, True)
        process.assert_not_called()

    def test_quota_blocks_new_work_but_preserves_cached_work(self):
        self.save('scripts-control.json', {'status': 'quota_blocked'})
        with patch.object(g.subprocess, 'Popen') as process:
            with self.assertRaises(g.ScriptQuotaBlocked):
                g.make_part(self.session, 0, 20, True)
            self.save(self.rel+'.json', self.corrected)
            self.assertEqual(g.make_part(self.session, 0, 20, True), self.corrected)
        process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
