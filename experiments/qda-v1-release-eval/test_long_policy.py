import unittest
from unittest.mock import patch
import worker

class LongPolicyTests(unittest.TestCase):
    def test_long_budget_is_unbounded_when_requested(self):
        # The worker's policy is represented directly in the source to prevent
        # accidentally restoring the old 1200-second cutoff.
        source=open(worker.__file__).read()
        self.assertIn("wall_budget=None if args.phase=='long' and args.unbounded_long",source)
        self.assertIn("--unbounded-long",source)
        self.assertIn("--rerun-incomplete",source)

if __name__=='__main__':unittest.main()
