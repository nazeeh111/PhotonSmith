"""Bounded actual-helper and fail-closed evidence review, not a benchmark."""
import sys,json,copy,hashlib,tempfile
from pathlib import Path
import numpy as np

ROOT=Path('/Users/nazeeh/Documents/ChatGPT/Continue/engineering-accomplishment-2026-10-29')
SOURCE=ROOT/'validation-v4/PhotonSmith'
OUT=ROOT/'review/fd-final'
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(SOURCE/'scripts'))
from photonsmith import gradient_check as check
import summarize_boundary as summary

def main():
    controls=[]
    a=np.array([1.,-1.,.5,-.25,.75,-.5])
    functions=[('constant',lambda x:1.,lambda x:np.zeros(6)),
               ('linear',lambda x:float(a@x),lambda x:a),
               ('quartic',lambda x:float(np.sum(a*x**4)),lambda x:4*a*x**3),
               ('exponential',lambda x:float(np.sum(a*np.exp(.3*x))),lambda x:.3*a*np.exp(.3*x))]
    for p in [np.array([.15,.85,.5,0.,1.,.5]),np.array([.15001,.84999,.4,.00001,.99999,.3])]:
        original=p.copy()
        for name,f,g in functions:
            for h in check.STEPS:
                record=check.forward_difference(f,p,h)
                expected=g(p)
                err=float(np.max(abs(np.array(record['gradient'])-expected)/np.maximum(1,abs(expected))))
                assert err<1e-7,(name,h,err)
                assert np.array_equal(original,p)
                for k,component in enumerate(record['components']):
                    for sample in component['samples']:
                        q=np.array(sample['params']);assert np.all(q>=check.LOWER) and np.all(q<=check.UPPER)
                        assert np.array_equal(q[np.arange(6)!=k],p[np.arange(6)!=k])
                controls.append({'params':p.tolist(),'function':name,'step':h,'error':err})
    frozen=SOURCE/'benchmarks/boundary/confirmation-v3'
    cases=json.loads((frozen/'cases.json').read_text())
    raw=[json.loads(x) for x in (frozen/'raw.jsonl').read_text().splitlines()]
    manifest=json.loads((frozen/'manifest.json').read_text())
    ids=[c['case']['id'] for c in cases]
    manifest.update(finite_difference_scheme=check.SCHEME,finite_difference_case_ids=ids,
                    finite_difference_steps=list(check.STEPS),finite_difference_normalization='both')
    # Toy linear oracles exercise record validation only. These are not scientific data.
    for case in cases:
        records=[]
        for h in check.STEPS:
            rec=check.forward_difference(lambda x:float(a@x),case['case']['params'],h)
            rec['oracle']='photonsmith.physics.solve:dense-expm-forward-only'
            records.append(rec)
        case['verification']['finite_differences']=records
    mutations=[]
    with tempfile.TemporaryDirectory(dir=OUT,prefix='schema-fixture-') as temporary:
        folder=Path(temporary)
        (folder/'inputs.json').write_bytes((frozen/'inputs.json').read_bytes())
        summary.validate_evidence(folder,cases,raw,manifest)
        for name in ('missing_late_case','missing_step','lost_scheme','wrong_sample','wrong_value','wrong_oracle','duplicate_step'):
            c=copy.deepcopy(cases);m=copy.deepcopy(manifest)
            if name=='missing_late_case':c[12]['verification'].pop('finite_differences')
            elif name=='missing_step':c[0]['verification']['finite_differences'].pop()
            elif name=='lost_scheme':m.pop('finite_difference_scheme')
            elif name=='wrong_sample':c[0]['verification']['finite_differences'][0]['components'][0]['samples'][0]['params'][1]+=.01
            elif name=='wrong_value':c[0]['verification']['finite_differences'][0]['components'][0]['samples'][0]['value']+=.01
            elif name=='wrong_oracle':c[0]['verification']['finite_differences'][0]['oracle']='native'
            elif name=='duplicate_step':c[0]['verification']['finite_differences'][1]=copy.deepcopy(c[0]['verification']['finite_differences'][0])
            try:summary.validate_evidence(folder,c,raw,m)
            except (ValueError,KeyError,IndexError):mutations.append({'mutation':name,'rejected':True})
            else:raise AssertionError('accepted '+name)
    result={'scope':'actual v4 module controls and schema mutation checks; no held-out scientific scoring',
            'sources':{p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in ['photonsmith/gradient_check.py','scripts/summarize_boundary.py','scripts/benchmark_boundary.py']},
            'control_checks':len(controls),'max_control_normalized_error':max(x['error'] for x in controls),
            'controls':controls,'schema_mutations':mutations}
    (OUT/'actual-v4-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='controls'},indent=2))

if __name__=='__main__':main()
