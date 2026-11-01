"""Independent v3 raw evidence audit; does not import the production summarizer."""
import json,hashlib,math,statistics,random,collections
from pathlib import Path
import numpy as np
ROOT=Path('/Users/nazeeh/Documents/ChatGPT/Continue/engineering-accomplishment-2026-10-29')
REV=ROOT/'correction-v2/PhotonSmith';RUN=REV/'benchmarks/boundary/confirmation-v3';PLAN=ROOT/'evidence/confirmation-v3-plan';OUT=Path(__file__).parent
def read(p):return json.loads(p.read_text())
def main():
    plan=read(PLAN/'protocol.json');inputs=read(RUN/'inputs.json');manifest=read(RUN/'manifest.json');cases=read(RUN/'cases.json');raw=[json.loads(l) for l in (RUN/'raw.jsonl').read_text().splitlines()];published=read(RUN/'summary.json')
    input_sha=hashlib.sha256((RUN/'inputs.json').read_bytes()).hexdigest()
    assert (RUN/'inputs.json').read_bytes()==(PLAN/'inputs.json').read_bytes()
    assert input_sha==plan['input_sha256']==manifest['cases_sha256']
    assert len(inputs)==len(cases)==plan['cases']==48
    assert collections.Counter(c['harmonics'] for c in inputs)=={3:12,5:12,9:12,15:12}
    assert [c['case'] for c in cases]==inputs
    assert manifest['expected_case_ids']==[c['id'] for c in inputs]
    methods=['legacy','native','meent_torch','meent_jax_jit','grcwa_autograd'];reps=7
    assert manifest['repeat_count']==reps
    keys=[(r['case'],r['rep'],r['method']) for r in raw];expected={(c['id'],rep,m) for c in inputs for rep in range(reps) for m in methods}
    assert len(raw)==1680 and len(set(keys))==1680 and set(keys)==expected
    assert not any('error' in r for r in raw)
    for r in raw:
        assert len(r['gradient'])==6 and np.isfinite(r['gradient']).all() and math.isfinite(r['value'])
        assert all(math.isfinite(r[k]) and r[k]>0 for k in ['wall','cpu'])
    idx=dict(zip(keys,raw));rng=random.Random(37832);expected_order=[]
    for c in inputs:
        for rep in range(reps):
            perm=methods.copy();rng.shuffle(perm);expected_order += [(c['id'],rep,m) for m in perm]
    assert expected_order==keys
    source_checks={}
    for path,sha in plan['sources'].items():
        actual=hashlib.sha256((REV/path).read_bytes()).hexdigest();assert actual==sha,path
        if path in manifest['sources']:assert manifest['sources'][path]==sha,path
        source_checks[path]=actual
    assert manifest['native_library']['source_sha256']==source_checks['native/boundary.cpp']
    assert hashlib.sha256((REV/'photonsmith/_boundary_native.dylib').read_bytes()).hexdigest()==manifest['native_library']['library_sha256']
    gen=np.random.default_rng(plan['seed']);regenerated=[]
    for m in [3,5,9,15]:
        for j in range(12):
            p=np.r_[gen.uniform(.15,.85,3),gen.uniform(0,1,3)];wl=float(gen.uniform(1.25,1.9))
            regenerated.append({'id':f'v3-m{m}-{j:02d}','harmonics':m,'params':p.tolist(),'wavelength':wl,'period':2.4,'layer_height':.3,'material_index':2.,'target_order':1})
    gen.shuffle(regenerated);assert regenerated==inputs
    gradient_errors=collections.defaultdict(list);value_errors=collections.defaultdict(list);fd=[];ratios=collections.defaultdict(list);cpuratios=collections.defaultdict(list);by_harm=collections.defaultdict(lambda:collections.defaultdict(list));wins=collections.Counter()
    for ci,c in enumerate(cases):
        name=c['case']['id'];n={rep:idx[name,rep,'native'] for rep in range(reps)}
        for m in methods:
            saved=c['observations'][m];assert len(saved)==reps
            for row in saved:assert row==idx[name,row['rep'],m]
            for rep in range(reps):
                row=idx[name,rep,m];g=np.array(n[rep]['gradient']);gradient_errors[m].append(float(np.max(abs(np.array(row['gradient'])-g)/np.maximum(1,abs(g)))));value_errors[m].append(abs(row['value']-n[rep]['value']))
            if m!='native':
                ratio=statistics.median(idx[name,rep,m]['wall']/n[rep]['wall'] for rep in range(reps));ratios[m].append(ratio);by_harm[m][c['case']['harmonics']].append(ratio);cpuratios[m].append(statistics.median(idx[name,rep,m]['cpu']/n[rep]['cpu'] for rep in range(reps)));wins[m]+=ratio>1
        if ci<8:
            assert [v['step'] for v in c['verification']['finite_differences']]==[1e-4,1e-5,1e-6]
            for v in c['verification']['finite_differences']:
                g=np.array(n[0]['gradient']);error=abs(np.array(v['gradient'])-g)/np.maximum(1,abs(g));fd.append({'case':name,'step':v['step'],'max_normalized_error':float(error.max()),'component':int(error.argmax())})
        else:assert 'finite_differences' not in c['verification']
    bootstrap=np.random.default_rng(37833);summary={}
    for m in ['legacy','meent_torch','meent_jax_jit','grcwa_autograd']:
        logs=np.array([math.log(v) for v in ratios[m]]);samples=bootstrap.integers(0,48,(10000,48));boot=np.exp(logs[samples].sum(axis=1)/48)
        summary[m]={'wall_geomean':math.exp(sum(logs)/48),'cpu_geomean':math.exp(sum(math.log(v) for v in cpuratios[m])/48),'bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),'native_case_median_wins':int(wins[m]),'by_harmonics':{str(h):math.exp(sum(math.log(v) for v in vs)/len(vs)) for h,vs in by_harm[m].items()}}
        p=published['summary'][m];assert math.isclose(summary[m]['wall_geomean'],p['geometric_mean_wall_speedup'],rel_tol=1e-13);assert math.isclose(summary[m]['cpu_geomean'],p['geometric_mean_cpu_speedup'],rel_tol=1e-13);np.testing.assert_allclose(summary[m]['bootstrap_95pct'],p['paired_case_bootstrap_95pct'],rtol=1e-13)
    max_power=max(c['verification']['max_power_error'] for c in cases);max_balance=max(c['verification']['balance_error'] for c in cases);maxfd=max(r['max_normalized_error'] for r in fd)
    assert maxfd>2e-5 and published['local_gates']['finite_difference'] is False
    assert published['max_power_error']==max_power and published['max_balance_error']==max_balance
    external=min((m for m in summary if m!='legacy'),key=lambda m:summary[m]['wall_geomean']);assert external==published['fastest_tested_external_aggregate']=='meent_jax_jit'
    threads=manifest['thread_observations'];assert len(threads)==50 and [t['case'] for t in threads if t['stage']=='before_case']==[c['id'] for c in inputs]
    result={'input_sha256':input_sha,'cases':48,'raw_attempts':1680,'failed_attempts':0,'unique_complete_raw_cartesian_product':True,'cases_equal_frozen_plan':True,'seed_regeneration_matches':True,'randomized_method_order_matches':True,'source_sha256':source_checks,'native_library_sha256':manifest['native_library']['library_sha256'],'summary_from_raw':summary,'fastest_external':external,'max_gradient_errors':{m:max(vs) for m,vs in gradient_errors.items()},'max_value_errors':{m:max(vs) for m,vs in value_errors.items()},'max_power_error':max_power,'max_balance_error':max_balance,'finite_differences':fd,'original_finite_difference_gate':False,'full_confirmation_pass':False,'thread_observation_count':len(threads),'observed_total_threads':sorted(set(t['total_process_threads'] for t in threads)),'observed_running_threads':sorted(set(t['running_process_threads'] for t in threads))}
    (OUT/'confirmation-v3-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['finite_differences','source_sha256']},indent=2))
if __name__=='__main__':main()
