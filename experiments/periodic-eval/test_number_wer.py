import unittest
from number_wer import tokens
class Numbers(unittest.TestCase):
 def test_languages(self):
  self.assertEqual(tokens('3','cs'),tokens('tři','cs'))
  self.assertNotEqual(tokens('3','cs'),tokens('tri','cs'))
  self.assertEqual(tokens('3','sk'),tokens('tri','sk'))
  self.assertNotEqual(tokens('3','sk'),tokens('tři','sk'))
 def test_compounds(self):
  for lang,text in [('cs','dvě stě třicet čtyři'),('sk','dvesto tridsať štyri')]:self.assertEqual(tokens('234',lang),tokens(text,lang))
  self.assertEqual(tokens('2026','cs'),tokens('dva tisíce dvacet šest','cs'))
 def test_case_forms(self):
  self.assertEqual(tokens('3 1 2','cs'),tokens('třech jedné dvou','cs'))
  self.assertEqual(tokens('3 1 2','sk'),tokens('troch jednej dvoch','sk'))
 def test_no_overreach(self):
  for value in ['003','3.','1,2','1.2','2026-10-22']:self.assertEqual(tokens(value,'cs'),[value])
  self.assertEqual(tokens('jedna dvě','cs'),['#1','#2'])
  self.assertNotEqual(tokens('pět','cs'),tokens('pet','cs'))
  self.assertEqual(tokens('třetí','cs'),['třetí'])
 def test_language_required(self):
  with self.assertRaises(ValueError):tokens('three','en')
if __name__=='__main__':unittest.main()
