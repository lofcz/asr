import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import build
from contracts import atomic_json,sha
from dataset import merge_target_turns,window_segments


class CombinedDatasetTests(unittest.TestCase):
    def test_repeated_utterance_hash_is_not_a_single_timeline_occurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sid='fixture';composition=[];words=[]
            for offset in (0.,.7):
                composition.append({'utterance_id':'same-content','voice_id':'voice',
                    'end_samples':round((offset+.5)*24000),'text':'Ano.'})
                words.append({'utterance_id':'same-content','text':'Ano.','start':offset+.05,
                              'end':offset+.5+.0000003,'speaker':0})
            atomic_json(root/'sessions'/sid/'composition.json',composition)
            atomic_json(root/'words.json',words)
            atomic_json(root/'labels.json',{'audio_duration':1.2,'segments':[{'start':.05,'end':1.2,'speaker':0,'text':'Ano. Ano.'}]})
            receipt={'id':sid,'label_path':'labels.json','words_path':'words.json',
                'label_sha256':sha(root/'labels.json'),'words_sha256':sha(root/'words.json'),
                'source_composition_sha256':sha(root/'sessions'/sid/'composition.json'),
                'known_speaker_voice_ids':{'0':'voice'},'audio_duration':1.2}
            with patch.object(build,'SYNTH',root):result=build.units_for_synthetic(receipt)
            self.assertEqual(len(result['segments']),2)
            self.assertEqual(result['segments'][1]['start'],.75)
            self.assertEqual(result['segments'][1]['end'],1.2)
            self.assertEqual(len(result['native_endpoint_adjustments']),1)

    def test_merging_keeps_wording_speakers_and_real_gaps(self):
        rows=[{'Start':0.,'End':1.,'Speaker':0,'Content':'My o vlku'},
              {'Start':1.2,'End':2.,'Speaker':0,'Content':'a vlk za dveřmi.'},
              {'Start':2.3,'End':3.,'Speaker':1,'Content':'Ano.'},
              {'Start':5.,'End':6.,'Speaker':1,'Content':'Další odpověď.'}]
        result=merge_target_turns(rows)
        self.assertEqual(len(result),3)
        self.assertEqual(result[0]['Content'],'My o vlku a vlk za dveřmi.')
        self.assertEqual(result[0]['End'],2.)
        self.assertEqual(rows[0]['Content'],'My o vlku')

    def test_long_recording_partition_has_no_lost_or_repeated_speech(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return [1]*(len(messages[0]['content'])//3)
            def encode(self,text,**kwargs):return list(text)
        class Processor:tokenizer=Tokenizer()
        units=[{'start':i*600+.1,'end':i*600+590.,'speaker':i%2,'text':'Odpověď '+chr(65+i)} for i in range(8)]
        source={'audio_duration':4800.,'audio_sha256':'a','label_sha256':'b'}
        rows=build.split_source(source,units,Processor())
        self.assertEqual(rows[0]['source_start'],0.)
        self.assertEqual(rows[-1]['source_end'],4800.)
        self.assertEqual(sum(r['duration'] for r in rows),4800.)
        self.assertTrue(all(r['duration']<=3600 for r in rows))
        self.assertTrue(all(a['source_end']==b['source_start'] for a,b in zip(rows,rows[1:])))
        self.assertEqual([t['Content'] for r in rows for t in r['target']],[s['text'] for s in units])


if __name__=='__main__':unittest.main()
