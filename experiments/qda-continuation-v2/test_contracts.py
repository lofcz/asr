import json,unittest
from contracts import target_with_spans,content_token_mask,wsd_lr
class Contracts(unittest.TestCase):
 def test_json_content_boundaries(self):
  strings=['Řekl: "Content": tři.','a\\b\nŽluťoučký','tri 3','']
  sample={'segments':[{'start':i,'end':i+1,'speaker':i%2,'text':s} for i,s in enumerate(strings)]}
  text,spans=target_with_spans(sample);rows=json.loads(text)
  self.assertEqual([r['Content'] for r in rows],strings)
  for s,(a,b) in zip(strings,spans):self.assertEqual(json.loads('"'+text[a:b]+'"'),s)
  mask=content_token_mask([(i,i+1) for i in range(len(text))],spans)
  self.assertFalse(mask[0]);self.assertTrue(mask[spans[0][0]])
 def test_schedule(self):
  self.assertEqual(wsd_lr(1,50,1000,100),.5)
  self.assertEqual(wsd_lr(1,400,1000,100),1)
  self.assertAlmostEqual(wsd_lr(1,900,1000,100),.55)
  self.assertAlmostEqual(wsd_lr(1,1000,1000,100),.1)
  with self.assertRaises(ValueError):wsd_lr(1,1,10,10)
if __name__=='__main__':unittest.main()
