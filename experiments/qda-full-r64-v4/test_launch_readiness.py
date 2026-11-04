"""A queued launch must not use a partial, failed or still-mutating preparation."""
import json,pathlib,tempfile,unittest
from await_ready import preparation_complete


class LaunchTests(unittest.TestCase):
    def test_requires_successful_terminal_preparation_and_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder)
            self.assertFalse(preparation_complete(root,'active'))
            with self.assertRaises(RuntimeError):preparation_complete(root,'inactive')
            (root/'preparation-status.json').write_text(json.dumps({'stage':'data-and-runtime-prepared'}))
            with self.assertRaises(RuntimeError):preparation_complete(root,'inactive')
            (root/'prepared.json').write_text('{}')
            self.assertFalse(preparation_complete(root,'active'))
            with self.assertRaises(RuntimeError):preparation_complete(root,'failed')
            self.assertTrue(preparation_complete(root,'inactive'))


if __name__=='__main__':unittest.main()
