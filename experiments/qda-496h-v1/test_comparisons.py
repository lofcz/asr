import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent/'periodic-eval'))
from comparisons import align
class Alignment(unittest.TestCase):
 def test_language_sensitive_numbers(self):
  _,a=align('Mám 3 děti.','Mám tři děti.','cs');self.assertEqual(a[0]['kind'],'format')
  _,b=align('Mám 3 děti.','Mám tri děti.','cs');self.assertEqual(b[0]['kind'],'wording')
  _,c=align('Mám 3 deti.','Mám tri deti.','sk');self.assertEqual(c[0]['kind'],'format')
 def test_diacritics(self):
  _,e=align('Mám kávu.','Mam kávu.','cs');self.assertEqual(e[0]['kind'],'accent')
 def test_omission_and_insertions(self):
  t,e=align('jedno velmi dlouhé slovo','jedno slovo navíc','cs')
  self.assertEqual({x['kind'] for x in e},{'missing','extra'})
  self.assertTrue(all(0<=x['anchor']<4 for x in t))
 def test_case_and_punctuation(self):
  t,e=align('Dobrý den!','dobrý den.','cs');self.assertEqual(e,[]);self.assertEqual(''.join(x['text'] for x in t),'dobrý den.')
if __name__=='__main__':unittest.main()
