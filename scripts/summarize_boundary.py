"""Recompute every summary from paired raw evidence; reject incomplete runs."""
import argparse,json,math,statistics
from pathlib import Path
import numpy as np

def summarize(folder):
    cases=json.loads((folder/'cases.json').read_text())
    raw=[json.loads(x) for x in (folder/'raw.jsonl').read_text().splitlines()]
    manifest=json.loads((folder/'manifest.json').read_text())
    expected=len(cases)*manifest['repeat_count']*5
    if len(raw)!=expected:raise ValueError('Incomplete raw observations')
    if any('error' in r for r in raw):raise ValueError('Run contains failed attempts; no successful-only speed summary')
    ids={(r['case'],r['rep'],r['method']) for r in raw}
    if len(ids)!=len(raw):raise ValueError('Duplicate raw observation')
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
                fd.append({'case':c['case']['id'],'step':test['step'],'max_normalized_error':float(np.max(abs(g-test['gradient'])/np.maximum(1,abs(g))))})
    summary={};rng=np.random.default_rng(37833)
    for method in rows[0]['speedup']:
        logs=np.log([r['speedup'][method] for r in rows]);boot=np.exp(logs[rng.integers(0,len(logs),(10000,len(logs)))].mean(axis=1))
        summary[method]={'geometric_mean_wall_speedup':float(np.exp(logs.mean())),'paired_case_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),'geometric_mean_cpu_speedup':float(np.exp(np.log([r['cpu_speedup'][method] for r in rows]).mean())),'by_harmonics':{str(m):float(np.exp(np.log([r['speedup'][method] for r in rows if r['harmonics']==m]).mean())) for m in (3,5,9,15)}}
    external=min((m for m in summary if m!='legacy'),key=lambda m:summary[m]['geometric_mean_wall_speedup'])
    result={'cases':len(cases),'raw_rows':len(raw),'failed_attempts':0,'summary':summary,'fastest_tested_external_aggregate':external,'max_power_error':max(c['verification']['max_power_error'] for c in cases),'max_balance_error':max(c['verification']['balance_error'] for c in cases),'max_residual':max(c['verification']['residual'] for c in cases),'max_normalized_gradient_error_by_method':{k:max(v) for k,v in gradient.items()},'max_value_error_by_method':{k:max(v) for k,v in value.items()},'finite_differences':fd,'case_ratios':rows}
    result['local_gates']={'legacy_speed':summary['legacy']['geometric_mean_wall_speedup']>=5,'external_speed':summary[external]['geometric_mean_wall_speedup']>=2,'external_interval':summary[external]['paired_case_bootstrap_95pct'][0]>1,'power':result['max_power_error']<=1e-7,'balance':result['max_balance_error']<=1e-8,'gradients':max(result['max_normalized_gradient_error_by_method'].values())<=2e-5,'finite_difference':max(x['max_normalized_error'] for x in fd)<=2e-5}
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('folder',type=Path);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    r=summarize(args.folder)
    with args.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('finite_differences','case_ratios')},indent=2))
