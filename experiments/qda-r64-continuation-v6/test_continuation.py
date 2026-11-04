import unittest
from dataset import complement, window_segments, verify_partition


class ContinuationContracts(unittest.TestCase):
    def test_exact_complement_includes_tiny_unseen_tail(self):
        self.assertEqual(complement(20.01,[(0,5),(10,20)]),[(5,10),(20,20.01)])

    def test_consumed_overlap_is_an_error(self):
        with self.assertRaises(ValueError):complement(20,[(0,10),(9,20)])

    def test_zero_duration_word_belongs_to_one_crop(self):
        segments=[{'start':0,'end':2,'speaker':'a','text':'A'},
                  {'start':5,'end':5,'speaker':'a','text':'B'},
                  {'start':8,'end':10,'speaker':'b','text':'C'},
                  {'start':10,'end':10,'speaker':'b','text':'D'}]
        first=window_segments(segments,0,5,10)
        second=window_segments(segments,5,10,10)
        self.assertEqual(''.join(s['Content'] for s in first),'A')
        self.assertEqual(''.join(s['Content'] for s in second),'BCD')
        self.assertEqual([s['Speaker'] for s in second],[0,1,1])

    def test_cut_cannot_truncate_speech(self):
        with self.assertRaises(ValueError):
            window_segments([{'start':0,'end':10,'speaker':'a','text':'word'}],0,5,10)

    def test_partition_rejects_missing_repeat_or_heldout(self):
        s={'record_id':'r','audio_sha256':'a','label_sha256':'l','show':'train','split':'train','audio_duration':10}
        def row(a,b,id):return {**s,'window_id':id,'source_start':a,'source_end':b,'duration':b-a}
        old=[row(0,4,'old')];new=[row(4,10,'new')]
        self.assertEqual(verify_partition([s],old,new)['overlap_seconds'],0)
        for wrong in ([row(5,10,'new')],[row(3,10,'new')]):
            with self.assertRaises(ValueError):verify_partition([s],old,wrong)
        with self.assertRaises(ValueError):verify_partition([s,dict(s,split='development')],old,new)


if __name__=='__main__':unittest.main()
