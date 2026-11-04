import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from collection_guard import stop_reason


class GuardTests(unittest.TestCase):
    config = {'target_hours': 1000, 'budget_stop_bytes': 95_000_000_000}

    def test_continue_below_both_limits(self):
        self.assertIsNone(stop_reason(999.9, 94_999_999_999, self.config))

    def test_target_stops_at_boundary(self):
        self.assertEqual(stop_reason(1000, 80_000_000_000, self.config), 'duration_target_reached')

    def test_storage_stops_even_before_target(self):
        self.assertEqual(stop_reason(900, 95_000_000_000, self.config), 'storage_cutoff')

    def test_storage_takes_precedence(self):
        self.assertEqual(stop_reason(1001, 96_000_000_000, self.config), 'storage_cutoff')
