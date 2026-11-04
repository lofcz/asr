import copy, unittest
from full_dataset import validate_intervals
from schedule import settings, require_decision


class FullRunContractTests(unittest.TestCase):
    def test_separate_windows_of_one_record_are_allowed_but_audio_overlap_is_not(self):
        a=dict(window_id='a',audio_sha256='source',source_start=0.,source_end=10.,duration=10.)
        b=dict(window_id='b',audio_sha256='source',source_start=10.,source_end=20.,duration=10.)
        validate_intervals([a,b])
        b.update(source_start=9.,duration=11.)
        with self.assertRaisesRegex(RuntimeError,'overlapping'):validate_intervals([a,b])

    def test_same_audio_under_two_different_record_ids_is_still_overlap(self):
        a=dict(window_id='a',record_id='first',audio_sha256='same',source_start=0.,source_end=10.,duration=10.)
        b={**a,'record_id':'second','window_id':'b'}
        with self.assertRaisesRegex(RuntimeError,'overlapping'):validate_intervals([a,b])

    def test_missing_decision_fails_before_model_load(self):
        with self.assertRaisesRegex(RuntimeError,'not finalized'):require_decision({'learning_rate':None})
        with self.assertRaisesRegex(RuntimeError,'unset'):settings({'scheduler':None},100)

    def test_no_implicit_warmup_or_custom_curve(self):
        with self.assertRaisesRegex(RuntimeError,'Explicit warmup'):settings({'scheduler':{'name':'linear'}},100)
        with self.assertRaisesRegex(RuntimeError,'standard'):settings({'scheduler':{'name':'invented','warmup_ratio':.1}},100)


if __name__=='__main__':unittest.main()
