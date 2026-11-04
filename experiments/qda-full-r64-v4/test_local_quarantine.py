import unittest
from targets import reconstruct, independent_verify, window_segments
from test_targets import token


class LocalQuarantineTests(unittest.TestCase):
    def test_local_quarantine_keeps_verbatim_stream_and_marks_entire_word(self):
        raw=[token('První věta.',0,1000),token(' Aper',4000,4100),token('ture.',7200,7500),
             token(' Další věta.',10000,12000)]
        segments,words,_=reconstruct(raw,15,local_quarantine=True)
        independent_verify(raw,segments,15)
        bad=[w for w in words if w.get('quarantine')]
        self.assertEqual(len(bad),1)
        self.assertEqual(bad[0]['text'],' Aperture.')
        self.assertEqual(bad[0]['token_indices'],[1,2])
        self.assertEqual(window_segments(segments,9,15)[0]['Content'],' Další věta.')

    def test_speaker_conflict_still_fails_without_reassignment(self):
        with self.assertRaisesRegex(ValueError,'speaker change inside word'):
            reconstruct([token('tř',0,200),token('i',300,400,'2')],1,local_quarantine=True)


if __name__=='__main__': unittest.main()
