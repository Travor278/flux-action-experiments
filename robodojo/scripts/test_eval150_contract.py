import copy,unittest
from eval150_contract import validate_result,aggregate
def result():
    details={str(i):{'layout_id':i,'success':i<20,'score':1.0 if i<20 else .25} for i in range(50)}
    return {'eval_time':50,'success_rate':.4,'score':55.0,'details':details}
class ContractTests(unittest.TestCase):
    def test_score_is_not_success_rate(self):self.assertAlmostEqual(validate_result(result(),list(range(50)))['score'],55)
    def test_reject_missing_duplicate_and_reordered_cases(self):
        for change in ['missing','duplicate','order']:
            r=result()
            if change=='missing':del r['details']['49']
            elif change=='duplicate':r['details']['49']['layout_id']=48
            else:r['details']['0']['layout_id'],r['details']['1']['layout_id']=1,0
            with self.assertRaises(AssertionError):validate_result(r,list(range(50)))
    def test_reject_score_tampering_nonfinite_and_success_mismatch(self):
        for value in [float('nan'),float('inf'),-.1,1.1,0]:
            r=result();r['details']['0']['score']=value
            with self.assertRaises(AssertionError):validate_result(r,list(range(50)))
        r=result();r['score']=40
        with self.assertRaises(AssertionError):validate_result(r,list(range(50)))
    def test_aggregate_requires_three_seeds_same_model_same_plan(self):
        rows=[{'eval_seed':i,'model_sha256':'m','plan_sha256':'p','verified':True,**validate_result(result(),list(range(50)))} for i in range(3)]
        a=aggregate(rows);self.assertEqual(a['total'],150);self.assertEqual(a['successes'],60);self.assertAlmostEqual(a['score_mean'],55)
        for field,value in [('eval_seed',1),('model_sha256','other'),('plan_sha256','other'),('verified',False),('total',49)]:
            bad=copy.deepcopy(rows);bad[2][field]=value
            with self.assertRaises(AssertionError):aggregate(bad)
        with self.assertRaises(AssertionError):aggregate(rows[:2])
if __name__=='__main__':unittest.main()
