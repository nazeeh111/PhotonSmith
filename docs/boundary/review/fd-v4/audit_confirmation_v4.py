"""Independent raw/statistical audit; does not import the benchmark scorer."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,math,statistics,random,hashlib,collections
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path('/Users/nazeeh/Documents/ChatGPT/Continue/engineering-accomplishment-2026-10-29')
SOURCE=ROOT/'validation-v4/PhotonSmith'
FOLDER=SOURCE/'benchmarks/boundary/confirmation-v4'
PLAN=ROOT/'validation-v4/confirmation-plan'
OUT=ROOT/'review/fd-final'
sys.path.insert(0,str(SOURCE))
from photonsmith import physics

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b,tol=2e-12):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(a,b,rel_tol=tol,abs_tol=1e-14),(a,b)

def main():
    inputs=json.loads((FOLDER/'inputs.json').read_text())
    manifest=json.loads((FOLDER/'manifest.json').read_text())
    protocol=json.loads((PLAN/'protocol.json').read_text())
    cases=json.loads((FOLDER/'cases.json').read_text())
    raw=[json.loads(x) for x in (FOLDER/'raw.jsonl').read_text().splitlines()]
    published=json.loads((FOLDER/'summary.json').read_text())
    input_hash=sha(FOLDER/'inputs.json')
    assert input_hash==sha(PLAN/'inputs.json')==manifest['cases_sha256']==protocol['input_sha256']=='db8d5d5ff4abe5146af409474424ad65bd1573ef72ab91cc15e3d1c445e10bd6'
    rng=np.random.default_rng(protocol['seed']);regenerated=[]
    for m in (3,5,9,15):
        for i in range(12):
            regenerated.append({'id':f'v4-m{m}-{i:02d}','harmonics':m,
                                'params':np.r_[rng.uniform(.15,.85,3),rng.uniform(0,1,3)].tolist(),
                                'wavelength':float(rng.uniform(1.25,1.9)),
                                'period':2.4,'layer_height':.3,'material_index':2.,'target_order':1})
    regenerated=[regenerated[i] for i in rng.permutation(48)]
    assert inputs==regenerated and len(inputs)==48
    assert collections.Counter(c['harmonics'] for c in inputs)=={3:12,5:12,9:12,15:12}
    ids=[c['id'] for c in inputs];assert len(set(ids))==48
    assert [c['case'] for c in cases]==inputs
    assert manifest['expected_case_ids']==manifest['finite_difference_case_ids']==ids
    assert manifest['finite_difference_scheme']=='dense-fourth-order-v1'
    assert manifest['finite_difference_steps']==[1e-4,1e-5,1e-6]
    assert manifest['repeat_count']==7 and len(raw)==48*7*5==1680
    source_hashes={p:sha(SOURCE/p) for p in protocol['sources']}
    assert source_hashes==protocol['sources']
    assert all(manifest['sources'][p]==h for p,h in source_hashes.items())
    receipt=manifest['native_library'];assert receipt['source_sha256']==source_hashes['native/boundary.cpp']
    assert sha(SOURCE/'photonsmith/_boundary_native.dylib')==receipt['library_sha256']
    methods=['legacy','native','meent_torch','meent_jax_jit','grcwa_autograd']
    baseline=[m for m in methods if m!='native']
    expected={(c,rep,m) for c in ids for rep in range(7) for m in methods}
    indexed={(r['case'],r['rep'],r['method']):r for r in raw}
    assert len(indexed)==1680 and set(indexed)==expected and not any('error' in r for r in raw)
    for row in raw:
        assert row['wall']>0 and row['cpu']>0
        assert np.isfinite(row['value']) and np.all(np.isfinite(row['gradient'])) and len(row['gradient'])==6
    for cid in ids:
        for rep in range(1,7):
            assert indexed[cid,rep,'native']['gradient']==indexed[cid,0,'native']['gradient']
            assert indexed[cid,rep,'native']['value']==indexed[cid,0,'native']['value']
    shuffle=random.Random(37832);cursor=0
    for c in ids:
        for rep in range(7):
            order=methods.copy();shuffle.shuffle(order)
            assert [(r['case'],r['rep'],r['method']) for r in raw[cursor:cursor+5]]==[(c,rep,m) for m in order]
            cursor+=5
    ratios={m:[] for m in baseline};cpus={m:[] for m in baseline}
    gradient_errors={m:[] for m in methods};value_errors={m:[] for m in methods}
    fd_rows=[];sample_count=0;kind_count=collections.Counter();component_count=0
    maximum_decimal_reconstruction_difference=0.;spotchecks=[]
    lower=np.array([.15,.15,.15,0.,0.,0.]);upper=np.array([.85,.85,.85,1.,1.,1.])
    published_fd={(r['case'],r['step']):r for r in published['finite_differences']}
    for ci,c in enumerate(cases):
        cid=c['case']['id'];assert set(c['observations'])==set(methods)
        reference={rep:indexed[cid,rep,'native'] for rep in range(7)}
        g=np.array(reference[0]['gradient']);p=np.array(c['case']['params'])
        for m in methods:
            obs=c['observations'][m];assert len(obs)==7 and {x['rep'] for x in obs}==set(range(7))
            for row in obs:
                assert row==indexed[cid,row['rep'],m]
                n=reference[row['rep']]
                gradient_errors[m].append(float(np.max(abs(np.array(row['gradient'])-n['gradient'])/np.maximum(1,abs(np.array(n['gradient']))))))
                value_errors[m].append(abs(row['value']-n['value']))
            if m!='native':
                ratios[m].append(statistics.median(indexed[cid,rep,m]['wall']/reference[rep]['wall'] for rep in range(7)))
                cpus[m].append(statistics.median(indexed[cid,rep,m]['cpu']/reference[rep]['cpu'] for rep in range(7)))
        records=c['verification']['finite_differences']
        assert len(records)==3 and [r['step'] for r in records]==[1e-4,1e-5,1e-6]
        for record in records:
            h=record['step'];assert record['scheme']=='dense-fourth-order-v1'
            assert record['oracle']=='photonsmith.physics.solve:dense-expm-forward-only'
            assert len(record['components'])==len(record['gradient'])==len(record['second_order_gradient'])==6
            fourth=[];second=[]
            for k,comp in enumerate(record['components']):
                if p[k]-2*h>=lower[k] and p[k]+2*h<=upper[k]:
                    kind='central';offsets=[-2,-1,1,2];integer_weights=[1,-8,8,-1]
                elif p[k]+4*h<=upper[k]:kind='forward';offsets=[0,1,2,3,4];integer_weights=[-25,48,-36,16,-3]
                else:
                    assert p[k]-4*h>=lower[k]
                    kind='backward';offsets=[0,-1,-2,-3,-4];integer_weights=[25,-48,36,-16,3]
                weights=[w/12 for w in integer_weights]
                assert comp['component']==k and comp['kind']==kind and comp['offsets']==offsets and comp['weights']==weights
                assert len(comp['samples'])==len(offsets)
                values={}
                for offset,sample in zip(offsets,comp['samples']):
                    point=p.copy();point[k]+=offset*h
                    assert sample['offset']==offset and sample['params']==point.tolist()
                    assert np.all(point>=lower) and np.all(point<=upper) and math.isfinite(sample['value'])
                    values[offset]=sample['value'];sample_count+=1
                    if ci in (0,12,24,36) and k in (0,3) and h==1e-4:
                        spotchecks.append({'case':cid,'component':k,'step':h,'point':point.tolist(),'value':sample['value'],'case_definition':c['case']})
                calc=math.fsum(w*values[o] for o,w in zip(offsets,weights))/h
                close(calc,record['gradient'][k])
                with localcontext() as context:
                    context.prec=60
                    ideal=sum((Decimal(w)*Decimal.from_float(values[o]) for o,w in zip(offsets,integer_weights)),Decimal(0))/(Decimal(12)*Decimal.from_float(h))
                maximum_decimal_reconstruction_difference=max(maximum_decimal_reconstruction_difference,abs(calc-float(ideal)))
                if kind=='central':diagnostic=(values[1]-values[-1])/(2*h)
                else:
                    sign=1 if kind=='forward' else -1
                    diagnostic=sign*math.fsum([-3*values[0],4*values[sign],-values[2*sign]])/(2*h)
                close(diagnostic,record['second_order_gradient'][k]);fourth.append(calc);second.append(diagnostic)
                kind_count[kind]+=1;component_count+=1
            fourth=np.array(fourth);second=np.array(second)
            analytic_error=float(np.max(abs(g-fourth)/np.maximum(1,abs(g))))
            fd_error=float(np.max(abs(g-fourth)/np.maximum(1,abs(fourth))))
            second_error=float(np.max(abs(g-second)/np.maximum(1,abs(g))))
            row={'case':cid,'step':h,'analytic_normalized_error':analytic_error,'fd_normalized_error':fd_error,
                 'worst_normalized_error':max(analytic_error,fd_error),'second_order_diagnostic_error':second_error}
            for key,actual in [('normalized_by_analytic_error',analytic_error),('normalized_by_fd_error',fd_error),
                               ('max_normalized_error',max(analytic_error,fd_error)),('second_order_max_error',second_error)]:
                close(actual,published_fd[cid,h][key])
            fd_rows.append(row)
    assert len(fd_rows)==48*3==144 and component_count==48*3*6==864
    recomputed={};bootstrap=np.random.default_rng(37833)
    for m in baseline:
        logs=np.log(ratios[m]);samples=bootstrap.integers(0,48,size=(10000,48));boots=np.exp(np.mean(logs[samples],axis=1))
        row={'wall_geomean':math.exp(math.fsum(math.log(x) for x in ratios[m])/48),
             'cpu_geomean':math.exp(math.fsum(math.log(x) for x in cpus[m])/48),
             'interval':np.percentile(boots,[2.5,97.5]).tolist(),
             'native_case_wins':sum(x>1 for x in ratios[m]),
             'by_harmonics':{str(order):math.exp(statistics.mean(math.log(x) for x,c in zip(ratios[m],inputs) if c['harmonics']==order)) for order in (3,5,9,15)}}
        reported=published['summary'][m]
        close(row['wall_geomean'],reported['geometric_mean_wall_speedup']);close(row['cpu_geomean'],reported['geometric_mean_cpu_speedup'])
        for x,y in zip(row['interval'],reported['paired_case_bootstrap_95pct']):close(x,y)
        for order,value in row['by_harmonics'].items():close(value,reported['by_harmonics'][order])
        recomputed[m]=row
    fastest=min(recomputed,key=lambda m:recomputed[m]['wall_geomean'] if m!='legacy' else math.inf)
    maxima={key:max(c['verification'][field] for c in cases) for key,field in [('power','max_power_error'),('balance','balance_error'),('residual','residual')]}
    gates={'legacy_speed':recomputed['legacy']['wall_geomean']>=5,
           'external_speed':recomputed[fastest]['wall_geomean']>=2,
           'external_interval':recomputed[fastest]['interval'][0]>1,
           'power':maxima['power']<=1e-7,'balance':maxima['balance']<=1e-8,
           'gradients':max(max(v) for v in gradient_errors.values())<=2e-5,
           'finite_difference':max(r['worst_normalized_error'] for r in fd_rows)<=2e-5}
    assert gates==published['local_gates'] and all(gates.values()) and published['all_local_gates_pass'] is True
    assert fastest==published['fastest_tested_external_aggregate']
    for m in methods:
        close(max(gradient_errors[m]),published['max_normalized_gradient_error_by_method'][m])
        close(max(value_errors[m]),published['max_value_error_by_method'][m])
    with threadpool_limits(1):
        for sample in spotchecks:
            c=sample['case_definition'];sol=physics.solve(sample['point'],wavelength=c['wavelength'],harmonics=c['harmonics'])
            fresh=float(sol['T'][c['harmonics']+c['target_order']]);sample['fresh_dense_value']=fresh
            sample['absolute_error']=abs(fresh-sample['value']);assert sample['absolute_error']<1e-10
    old=json.loads((SOURCE/'benchmarks/boundary/confirmation-v3/summary.json').read_text())
    assert old['local_gates']['finite_difference'] is False
    threads=manifest['thread_observations'];assert len(threads)==50
    result={'scope':'independent coding-agent raw/statistics/FD audit, not human external validation',
            'input_sha256':input_hash,'seed_regeneration_exact':True,'case_count':48,'raw_count':1680,'raw_failures':0,
            'exact_raw_id_cartesian_set':True,'randomized_method_order_reproduced':True,'source_hashes':source_hashes,
            'native_value_and_gradient_identical_across_all_seven_repeats':True,
            'native_library_sha256':receipt['library_sha256'],'sources_and_library_match_frozen_plan':True,
            'fd_case_step_records':len(fd_rows),'fd_coordinate_estimates':component_count,'forward_samples':sample_count,
            'stencil_counts':dict(kind_count),'max_fourth_order_worst_normalized_error':max(r['worst_normalized_error'] for r in fd_rows),
            'max_fourth_order_error_by_step':{str(h):max(r['worst_normalized_error'] for r in fd_rows if r['step']==h) for h in (1e-4,1e-5,1e-6)},
            'max_retained_second_order_error':max(r['second_order_diagnostic_error'] for r in fd_rows),
            'max_binary64_vs_60_digit_weighted_reconstruction_difference':maximum_decimal_reconstruction_difference,
            'statistics':recomputed,'fastest_external':fastest,'numerical_maxima':maxima,
            'max_external_gradient_errors':{m:max(v) for m,v in gradient_errors.items()},'gates':gates,'all_seven_gates_pass':all(gates.values()),
            'old_v3_gate_still_false':True,'thread_observations':len(threads),
            'total_threads_observed':sorted({x.get('total_process_threads') for x in threads}),
            'thread_limit':'total includes idle workers, not strict single-thread JAX',
            'fresh_dense_spotcheck_count':len(spotchecks),'fresh_dense_spotcheck_max_error':max(s['absolute_error'] for s in spotchecks),
            'fd_records':fd_rows,'spotchecks':spotchecks,
            'routing_failure':{'scope':'parent transcription of observed tool output, not an original process logfile',
                               'path':str(ROOT/'evidence/confirmation-v4-summary-routing-failure.json'),
                               'sha256':sha(ROOT/'evidence/confirmation-v4-summary-routing-failure.json'),
                               'effect':'direct script import failed; supported module invocation succeeded without source/input/threshold changes'}}
    (OUT/'confirmation-v4-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('fd_records','spotchecks')},indent=2))

if __name__=='__main__':main()
