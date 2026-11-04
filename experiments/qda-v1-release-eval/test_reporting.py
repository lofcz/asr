import json,pathlib,tempfile,unittest
from unittest.mock import patch
from common import HERE
from report import aggregate,paired
from number_wer import score

def row(key,lang,errors=10):
    return {'window_id':key,'record_id':key,'show':lang+' show','language':lang,'complete':True,'metrics':{
        'number_normalized_reference_words':100,'number_normalized_errors':errors,'reference_words':100,
        'substitutions':errors,'insertions':0,'deletions':0,
        'der_details':{'scored_speaker_time':100,'missed_speaker_time':1,'falarm_speaker_time':0,'speaker_error_time':0}}}

class ReportingTests(unittest.TestCase):
    def test_failed_recording_remains_in_denominator(self):
        result=aggregate([row('a','cs'),{'complete':False,'error':'loop'}],3)
        self.assertEqual((result['expected'],result['finished'],result['scored'],result['failures']),(3,2,1,1))
        self.assertAlmostEqual(result['number_normalized_wer'],.1)
        self.assertIn('Completed valid outputs only',result['metric_scope'])

    def test_paired_identity_has_zero_delta(self):
        rows={str(i):row(str(i),'cs' if i%2==0 else 'sk') for i in range(4)}
        result=paired(rows,rows,draws=100)
        self.assertEqual(result['balanced_language_wer_delta_95'],[0.,0.])
        self.assertEqual(result['der_delta_95'],[0.,0.])

    def test_paired_improvement_direction_and_show_clustering(self):
        a={str(i):row(str(i),'cs' if i%2==0 else 'sk',10) for i in range(4)}
        b={str(i):row(str(i),'cs' if i%2==0 else 'sk',5) for i in range(4)}
        result=paired(a,b,'show',draws=100)
        self.assertEqual(result['clusters_by_language'],{'cs':1,'sk':1})
        for value in result['balanced_language_wer_delta_95']:self.assertAlmostEqual(value,-.05)

    def test_czech_diacritics_and_inflections_remain_errors(self):
        self.assertEqual(score('tři','3','cs')['number_normalized_errors'],0)
        self.assertEqual(score('tri','3','cs')['number_normalized_errors'],1)
        self.assertEqual(score('všemi','všema','cs')['number_normalized_errors'],1)
        self.assertEqual(score('karcinogenními','karcinogeními','cs')['number_normalized_errors'],1)

    def test_final_report_and_candidate_are_written_after_complete_matrix(self):
        import report
        with tempfile.TemporaryDirectory(prefix='qda-report-fixture-') as temp:
            root=pathlib.Path(temp)
            main=[{'window_id':str(i),'cohort':'existing-development' if i<8 else 'new-development' if i<24 else 'new-confirmation'} for i in range(32)]
            long=[{'window_id':str(i),'cohort':'long-session'} for i in range(32,34)]
            models={name:{'adapter_sha256':name} for name in ['300h','450h','final','vanilla']}
            (root/'selection.json').write_text(json.dumps({'rows':main,'long_rows':long,'models':models}))
            (root/'feature-contract.json').write_text(json.dumps({'model_revision':'test-fixture'}))
            for name in models:
                for source in main+long:
                    key=source['window_id'];item=row(key,'cs' if int(key)%2==0 else 'sk',{'vanilla':10,'300h':6,'450h':4,'final':4}[name])
                    item.update(adapter_sha256=name,cohort=source['cohort'],strict_output_valid=True)
                    item['metrics']['der_details']['speaker_error_time']={'vanilla':4,'300h':3,'450h':2,'final':1}[name]
                    folder=root/'evaluations'/name/key;folder.mkdir(parents=True);(folder/'result.json').write_text(json.dumps(item))
            with patch.object(report,'HERE',root),patch.object(report,'BASE',root/'unused-baseline'),patch('sys.argv',['report.py','--final']):report.main()
            result=json.loads((root/'report.json').read_text());candidate=json.loads((root/'v1-candidate.json').read_text())
            self.assertEqual(result['finished'],136)
            self.assertEqual(result['recommendation']['checkpoint'],'final')
            self.assertTrue(candidate['main_all_completed'] and candidate['long_all_completed'])
            self.assertEqual(candidate['report_sha256'],report.sha(root/'report.json'))
            self.assertIn('WER delta 95% interval',(root/'report.md').read_text())
            self.assertTrue((root/'word-errors.csv').exists())

if __name__=='__main__':unittest.main()
