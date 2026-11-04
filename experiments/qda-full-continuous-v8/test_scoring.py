import pathlib
import tempfile
import unittest
from scoring import metrics,validate_hypothesis


class ScoringTests(unittest.TestCase):
    def test_speech_abstention_counts_missing_words_and_diarization(self):
        ref=[{'Start':0.,'End':1.,'Speaker':0,'Content':'Ahoj světe'},
             {'Start':2.,'End':3.,'Speaker':0,'Content':'Dobrý den'}]
        hyp=[{'Start':0.,'End':1.,'Content':'[Speech]'},dict(ref[1])]
        before=repr(hyp);validate_hypothesis(hyp,4.)
        with tempfile.TemporaryDirectory() as folder:result=metrics(ref,hyp,pathlib.Path(folder),4.,'cs')
        self.assertEqual(repr(hyp),before)
        self.assertEqual(result['number_normalized_errors'],2)
        self.assertEqual(result['reference_words'],4)
        self.assertAlmostEqual(result['der_250ms'],.5)
        self.assertEqual(result['speech_abstention_segments'],1)
        with self.assertRaises(RuntimeError):validate_hypothesis([{'Start':0.,'End':1.,'Content':'Ahoj'}],4.)
        with self.assertRaises(RuntimeError):validate_hypothesis([{'Start':0.,'End':float('nan'),'Content':'[Speech]'}],4.)

    def test_reviewed_word_boundaries_and_events_do_not_corrupt_wer_or_der(self):
        ref=[{'Start':0.,'End':1.,'Speaker':0,'Content':'Ahoj'},
             {'Start':1.1,'End':2.,'Speaker':0,'Content':'světe'},
             {'Start':3.,'End':4.,'Speaker':None,'Content':'[Music]'}]
        with tempfile.TemporaryDirectory() as folder:
            result=metrics(ref,ref,pathlib.Path(folder),5.,'cs')
        self.assertEqual(result['reference_words'],2)
        self.assertEqual(result['number_normalized_wer'],0)
        self.assertEqual(result['der_250ms'],0)
        self.assertEqual(result['reference_speakers'],1)

    def test_czech_tri_stays_different_from_three(self):
        ref=[{'Start':0.,'End':1.,'Speaker':0,'Content':'tři'}]
        with tempfile.TemporaryDirectory() as folder:
            correct=metrics(ref,[dict(ref[0],Content='3')],pathlib.Path(folder),2.,'cs')
            incorrect=metrics(ref,[dict(ref[0],Content='tri')],pathlib.Path(folder),2.,'cs')
        self.assertEqual(correct['number_normalized_errors'],0)
        self.assertEqual(incorrect['number_normalized_errors'],1)


if __name__=='__main__':unittest.main()
