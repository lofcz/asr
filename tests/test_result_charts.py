"""Protect the chart comparisons from unequal panels and misleading aggregation."""
import copy
import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("result_charts", ROOT / "scripts/build_result_charts.py")
charts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(charts)


class ChartEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "charts/results.json").read_text())

    def test_languages_have_equal_weight_even_when_lengths_differ(self):
        value = charts.balanced_wer({
            "cs": {"word_errors": 1, "reference_words": 10},
            "sk": {"word_errors": 1, "reference_words": 100},
        })
        self.assertAlmostEqual(value, .055)

    def test_zero_edits_are_a_measurement_but_missing_language_is_not(self):
        rows = {lang: {"word_errors": 0, "reference_words": 10} for lang in ["cs", "sk"]}
        self.assertEqual(charts.balanced_wer(rows), 0)
        del rows["sk"]
        with self.assertRaises(ValueError):
            charts.balanced_wer(rows)

    def test_failed_output_cannot_silently_leave_the_comparison(self):
        self.data["checkpoint_comparison"]["models"]["450h"]["scored"] -= 1
        with self.assertRaises(AssertionError):
            charts.validate(self.data)

    def test_different_references_cannot_share_a_commercial_comparison(self):
        self.data["commercial_comparison"]["models"]["gemini"]["languages"]["cs"]["reference_words"] += 1
        with self.assertRaises(AssertionError):
            charts.validate(self.data)

    def test_reference_provider_is_not_a_zero_error_competitor(self):
        self.data["commercial_comparison"]["models"]["soniox"] = copy.deepcopy(
            self.data["commercial_comparison"]["models"]["gemini"]
        )
        with self.assertRaises(AssertionError):
            charts.validate(self.data)

    def test_saved_snapshot_preserves_raw_validity_separately(self):
        charts.validate(self.data)
        row = self.data["checkpoint_comparison"]["models"]["450h"]
        self.assertEqual(row["scored"], 8)
        self.assertEqual(row["strict_output_valid"], 7)

    def whisper_expected(self):
        return {row["id"]: {
            **{key: row[key] for key in ["cohort", "language", "audio_sha256", "reference_sha256", "source_start", "duration_seconds"]},
            "reference_words": row["metrics"]["number_normalized_reference_words"],
        } for row in self.data["whisper_benchmark"]["rows"]}

    def test_whisper_run_must_use_the_same_source_audio(self):
        expected = self.whisper_expected()
        run = copy.deepcopy(self.data["whisper_benchmark"])
        run["rows"][0]["audio_sha256"] = "different-source-audio"
        with self.assertRaises(AssertionError):
            charts.attach_whisper(self.data, run, expected)

    def test_whisper_run_cannot_duplicate_one_clip_and_omit_another(self):
        expected = self.whisper_expected()
        run = copy.deepcopy(self.data["whisper_benchmark"])
        run["rows"][-1] = copy.deepcopy(run["rows"][-2])
        with self.assertRaises(AssertionError):
            charts.attach_whisper(self.data, run, expected)

    def test_text_only_baseline_has_no_invented_diarization(self):
        charts.attach_whisper(self.data, self.data["whisper_benchmark"], self.whisper_expected())
        baseline = self.data["checkpoint_comparison"]["baselines"]["whisper"]
        self.assertNotIn("diarization_error_seconds", baseline)
        baseline["diarization_error_seconds"] = 0
        with self.assertRaises(AssertionError):
            charts.validate(self.data)



if __name__ == "__main__":
    unittest.main()
