import copy
import unittest
from final_report import aggregate, output_issues


class OutputAccounting(unittest.TestCase):
    def test_late_words_are_preserved_and_marked(self):
        rows=[{'Start':596.49,'End':606.11,'Speaker':0,'Content':' late words'},
              {'Start':606.17,'End':608.43,'Speaker':0,'Content':' more words'}]
        before=copy.deepcopy(rows)
        issues=output_issues(rows,598.98)
        self.assertEqual(len(issues),2)
        self.assertAlmostEqual(max(i['overrun_seconds'] for i in issues),9.45)
        self.assertEqual(rows,before)

    def test_invalid_numeric_times_are_not_scoreable(self):
        for start,end in [(float('nan'),2),(3,2),(-1,1)]:
            with self.assertRaises(ValueError):output_issues([dict(Start=start,End=end,Content='word')],5)

    def test_failed_strict_output_still_contributes_errors(self):
        def row(errors,issues):
            return {'output_issues':issues,'metrics':{'number_normalized_reference_words':100,
                'number_normalized_errors':errors,'der_details':{'scored_speaker_time':100,
                'missed_speaker_time':1,'falarm_speaker_time':2,'speaker_error_time':0}}}
        result=aggregate([row(2,[]),row(40,[{'kind':'prediction_exceeds_clip'}])])
        self.assertEqual(result['errors'],42)
        self.assertEqual(result['samples'],2)
        self.assertEqual(result['raw_json_valid'],2)
        self.assertEqual(result['strict_output_valid'],1)


if __name__=='__main__':unittest.main()
