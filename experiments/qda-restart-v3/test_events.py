import pathlib
import tempfile
import unittest
from contracts import atomic_json
from events import reviewed_events,with_events


class EventTests(unittest.TestCase):
    def test_vad_gap_has_no_label_without_explicit_review(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(reviewed_events(pathlib.Path(folder),{'window_id':'abc'},{'unresolved_nonspeech':[[1.,5.]]}))

    def test_review_is_bound_to_audio_labels_intervals_and_listening(self):
        row={'window_id':'abc','audio_sha256':'audio','label_sha256':'label','source_start':10.}
        q={'unresolved_nonspeech':[[12.,16.]]}
        review={'decision':'approve','reviewer':'unit-test fixture','reviewed_at':1,'audio_sha256':'audio','label_sha256':'label',
                'regions':[{'start':12.,'end':16.,'tag':'[Silence]','listened':True}]}
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);path=root/'acoustic-reviews/abc.json';atomic_json(path,review)
            events,digest=reviewed_events(root,row,q);self.assertEqual(events,[{'Start':2.,'End':6.,'Speaker':None,'Content':'[Silence]'}])
            target=[{'Start':0.,'End':1.,'Speaker':0,'Content':'Slovo'}];self.assertEqual(with_events(target,events)[0],target[0])
            review['regions'][0]['listened']=False;atomic_json(path,review)
            with self.assertRaises(ValueError):reviewed_events(root,row,q)
            review['regions'][0]['listened']=True;review['audio_sha256']='changed';atomic_json(path,review)
            with self.assertRaises(ValueError):reviewed_events(root,row,q)


if __name__=='__main__':unittest.main()
