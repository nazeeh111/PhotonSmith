"""Recompute every summary from paired raw evidence; reject incomplete runs."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np

METHODS = ('native','legacy','meent_torch','meent_jax_jit','grcwa_autograd')


def validate_evidence(folder, cases, raw, manifest):
    """Score only a complete, internally consistent frozen experiment."""
    inputs_path = folder/'inputs.json'
    if not inputs_path.is_file():
        raise ValueError('Frozen inputs.json is required to establish planned cases')
    encoded = inputs_path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != manifest['cases_sha256']:
        raise ValueError('Frozen input hash differs from manifest')
    planned = json.loads(encoded)
    case_ids = [c['id'] for c in planned]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError('Planned case IDs must be nonempty and unique')
    if manifest.get('expected_case_ids', case_ids) != case_ids:
        raise ValueError('Manifest case IDs differ from frozen inputs')
    if manifest.get('methods', list(METHODS)) != list(METHODS):
        raise ValueError('Manifest method set differs from this experiment')
    reps = manifest['repeat_count']
    scheme = manifest.get('finite_difference_scheme')
    independent_declared = (any(k in manifest for k in ('finite_difference_scheme','finite_difference_case_ids',
                             'finite_difference_steps','finite_difference_normalization'))
                            or 'photonsmith/gradient_check.py' in manifest.get('sources',{})
                            or any(v.get('scheme') for c in cases for v in c['verification'].get('finite_differences',[])))
    if independent_declared:
        from photonsmith.gradient_check import SCHEME,STEPS
        if scheme!=SCHEME or manifest.get('finite_difference_case_ids')!=case_ids or manifest.get('finite_difference_steps')!=list(STEPS):
            raise ValueError('Independent finite-difference plan differs from complete frozen inputs')
    if type(reps) is not int or reps < 3:
        raise ValueError('At least three paired repeats required')
    if [c['case'] for c in cases] != planned:
        raise ValueError('Completed cases differ from the complete frozen input plan')
    expected = {(c, rep, method) for c in case_ids for rep in range(reps) for method in METHODS}
    if any('error' in r for r in raw):
        raise ValueError('Run contains failed attempts; no successful-only speed summary')
    indexed = {(r['case'], r['rep'], r['method']): r for r in raw}
    if len(indexed) != len(raw) or set(indexed) != expected:
        raise ValueError('Missing, duplicate or unexpected raw observations')
    for row in raw:
        gradient = np.asarray(row['gradient'], dtype=float)
        if (gradient.shape != (6,) or not np.all(np.isfinite(gradient))
                or not np.isfinite(row['value'])
                or any(not np.isfinite(row[k]) or row[k] <= 0 for k in ('wall','cpu'))):
            raise ValueError('Nonfinite or invalid timing/value/gradient observation')
    for ci, case in enumerate(cases):
        observations = case['observations']
        if set(observations) != set(METHODS):
            raise ValueError('Unexpected method set')
        for method, rows in observations.items():
            if len(rows) != reps or {r['rep'] for r in rows} != set(range(reps)):
                raise ValueError('Missing or duplicate method repeats')
            for row in rows:
                key = (case['case']['id'], row['rep'], method)
                if row != indexed[key]:
                    raise ValueError('Case observations differ from raw evidence')
        verification = case['verification']
        for key in ('max_power_error','balance_error','residual'):
            if not np.isfinite(verification[key]) or verification[key] < 0:
                raise ValueError('Invalid numerical verification')
        if scheme is not None or ci < 8:
            fd = verification.get('finite_differences', [])
            if [v['step'] for v in fd] != [1e-4,1e-5,1e-6]:
                raise ValueError('Missing required finite-difference step sweep')
            for check in fd:
                g = np.asarray(check['gradient'], dtype=float)
                if g.shape != (6,) or not np.all(np.isfinite(g)):
                    raise ValueError('Invalid finite-difference gradient')
                if scheme is not None:
                    from photonsmith.gradient_check import validate_record
                    validate_record(case['case']['params'],check)


def summarize(folder):
    cases=json.loads((folder/'cases.json').read_text())
    raw=[json.loads(x) for x in (folder/'raw.jsonl').read_text().splitlines()]
    manifest=json.loads((folder/'manifest.json').read_text())
    validate_evidence(folder, cases, raw, manifest)
    rows=[];fd=[];gradient={};value={};reps=manifest['repeat_count']
    for c in cases:
        observations=c['observations'];reference={r['rep']:r for r in observations['native']}
        if set(observations)!=set(['native','legacy','meent_torch','meent_jax_jit','grcwa_autograd']):raise ValueError('Unexpected method set')
        ratios={};cpu_ratios={}
        for name,obs in observations.items():
            if len(obs)!=reps:raise ValueError('Missing method repeats')
            for x in obs:
                n=reference[x['rep']]
                gradient.setdefault(name,[]).append(float(np.max(abs(np.array(x['gradient'])-n['gradient'])/np.maximum(1,abs(np.array(n['gradient']))))))
                value.setdefault(name,[]).append(abs(x['value']-n['value']))
            if name!='native':
                ratios[name]=statistics.median(x['wall']/reference[x['rep']]['wall'] for x in obs)
                cpu_ratios[name]=statistics.median(x['cpu']/reference[x['rep']]['cpu'] for x in obs)
        rows.append({'case':c['case']['id'],'harmonics':c['case']['harmonics'],'speedup':ratios,'cpu_speedup':cpu_ratios})
        if 'finite_differences' in c['verification']:
            g=np.array(next(iter(reference.values()))['gradient'])
            for test in c['verification']['finite_differences']:
                estimate=np.asarray(test['gradient'])
                analytic_error=float(np.max(abs(g-estimate)/np.maximum(1,abs(g))))
                row={'case':c['case']['id'],'step':test['step'],'max_normalized_error':analytic_error}
                if manifest.get('finite_difference_scheme'):
                    fd_error=float(np.max(abs(g-estimate)/np.maximum(1,abs(estimate))))
                    row.update(normalized_by_analytic_error=analytic_error,normalized_by_fd_error=fd_error,
                               max_normalized_error=max(analytic_error,fd_error),scheme=test['scheme'],
                               second_order_max_error=float(np.max(abs(g-test['second_order_gradient'])/np.maximum(1,abs(g)))))
                fd.append(row)
    summary={};rng=np.random.default_rng(37833)
    for method in rows[0]['speedup']:
        logs=np.log([r['speedup'][method] for r in rows]);boot=np.exp(logs[rng.integers(0,len(logs),(10000,len(logs)))].mean(axis=1))
        summary[method]={'geometric_mean_wall_speedup':float(np.exp(logs.mean())),'paired_case_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),'geometric_mean_cpu_speedup':float(np.exp(np.log([r['cpu_speedup'][method] for r in rows]).mean())),'by_harmonics':{str(m):float(np.exp(np.log([r['speedup'][method] for r in rows if r['harmonics']==m]).mean())) for m in (3,5,9,15)}}
    external=min((m for m in summary if m!='legacy'),key=lambda m:summary[m]['geometric_mean_wall_speedup'])
    result={'cases':len(cases),'raw_rows':len(raw),'failed_attempts':0,'summary':summary,'fastest_tested_external_aggregate':external,'max_power_error':max(c['verification']['max_power_error'] for c in cases),'max_balance_error':max(c['verification']['balance_error'] for c in cases),'max_residual':max(c['verification']['residual'] for c in cases),'max_normalized_gradient_error_by_method':{k:max(v) for k,v in gradient.items()},'max_value_error_by_method':{k:max(v) for k,v in value.items()},'finite_differences':fd,'case_ratios':rows}
    result['local_gates']={'legacy_speed':summary['legacy']['geometric_mean_wall_speedup']>=5,'external_speed':summary[external]['geometric_mean_wall_speedup']>=2,'external_interval':summary[external]['paired_case_bootstrap_95pct'][0]>1,'power':result['max_power_error']<=1e-7,'balance':result['max_balance_error']<=1e-8,'gradients':max(result['max_normalized_gradient_error_by_method'].values())<=2e-5,'finite_difference':max(x['max_normalized_error'] for x in fd)<=2e-5}
    if manifest.get('finite_difference_scheme'):
        result['finite_difference_protocol']={'scheme':manifest['finite_difference_scheme'],'cases':len(cases),
                                              'components_per_case':6,'steps':manifest['finite_difference_steps'],
                                              'limit':2e-5,'normalization':'both denominators; worst controls gate'}
        result['all_local_gates_pass']=all(result['local_gates'].values())
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('folder',type=Path);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    r=summarize(args.folder)
    with args.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('finite_differences','case_ratios')},indent=2))
