import copy
import unittest
from targets import reconstruct, independent_verify, safe_windows, window_segments


def token(text,start,end,speaker='1'):
    return dict(text=text,start_ms=start,end_ms=end,speaker=speaker,confidence=.99,language='cs')


class TargetTests(unittest.TestCase):
    def test_real_failure_pattern_does_not_split_word_on_gap(self):
        raw=[token('To je velm',0,900),token('i',1900,2000),token(' cenné.',2100,2600)]
        segments,words,repairs=reconstruct(raw,3)
        independent_verify(raw,segments,3)
        self.assertEqual(' '.join(s['text'] for s in segments),'To je velmi cenné.')
        self.assertEqual(len(repairs),1)

    def test_do_not_guess_speaker_inside_word(self):
        with self.assertRaisesRegex(ValueError,'speaker change inside word'):
            reconstruct([token('tř',0,200),token('i',250,400,'2')],1)

    def test_long_gap_is_quarantined_instead_of_inventing_timing(self):
        with self.assertRaisesRegex(ValueError,'intraword timestamp gap'):
            reconstruct([token('Aper',0,100),token('ture',3100,3300)],4)

    def test_standalone_whitespace_preserved(self):
        raw=[token('tři',0,200),token(' ',200,200),token('roky',210,500)]
        segments,_,_=reconstruct(raw,1)
        independent_verify(raw,segments,1)
        self.assertEqual(''.join(s['text'] for s in segments),'tři roky')

    def test_segment_assembly_does_not_mutate_lexical_token_ownership(self):
        raw=[token('Ne',0,100),token('můžeme',100,200),token(' to',300,400)]
        segments,words,_=reconstruct(raw,1)
        self.assertEqual([w['token_indices'] for w in words],[[0,1],[2]])
        self.assertEqual(segments[0]['token_indices'],[0,1,2])

    def test_independent_verifier_catches_previous_lossless_but_broken_target(self):
        raw=[token('jakob',0,200),token('y',1500,1700)]
        broken=[dict(start=0,end=.2,text='jakob',speaker='1',token_indices=[0]),
                dict(start=1.5,end=1.7,text='y',speaker='1',token_indices=[1])]
        with self.assertRaisesRegex(ValueError,'lexical word split'):
            independent_verify(raw,broken,2)

    def test_real_pause_preserved_and_no_speech_is_cut(self):
        raw=[token('První věta.',0,1500),token(' Druhá věta.',4000,5500)]
        segments,_,_=reconstruct(raw,6)
        self.assertEqual(len(segments),2)
        windows=safe_windows(segments,6,3)
        self.assertEqual(windows,[(0.,2.75),(2.75,6)])
        self.assertEqual(window_segments(segments,*windows[1])[0]['Start'],1.25)
        with self.assertRaisesRegex(ValueError,'cuts speech'):
            window_segments(segments,0,1)

    def test_tampered_content_and_duplicate_indices_rejected(self):
        raw=[token('Správný text.',0,200)]
        segments,_,_=reconstruct(raw,1)
        changed=copy.deepcopy(segments);changed[0]['text']='Jiný text.'
        with self.assertRaises(ValueError):independent_verify(raw,changed,1)
        changed=copy.deepcopy(segments);changed[0]['token_indices'].append(0)
        with self.assertRaises(ValueError):independent_verify(raw,changed,1)


if __name__=='__main__':unittest.main()
