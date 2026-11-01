"""Independent recomputation of saved benchmark rows; executes no solver."""
from pathlib import Path
import hashlib,json,random,collections
import numpy as np
base=Path(__file__).resolve().parent.parent
review=base/'review';repo=base/'PhotonSmith';ev=base/'evidence/confirmation-v2'
protocol=json.loads((base/'protocol/PROTOCOL-v2.json').read_text())
inputs=json.loads((base/'protocol/confirmation-cases-v2.json').read_text())
manifest=json.loads((ev/'manifest.json').read_text());cases=json.loads((ev/'cases.json').read_text())
raw=[json.loads(s) for s in (ev/'raw.jsonl').read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
methods=['legacy','native','meent_torch','meent_jax_jit','grcwa_autograd']
checks={};checks['input_hash_matches']=sha(base/'protocol/confirmation-cases-v2.json')==protocol['input_sha256']==manifest['cases_sha256']
checks['case_count_48']=len(inputs)==len(cases)==48
checks['ids_unique']=len({c['id'] for c in inputs})==48
checks['cases_match_inputs_in_order']=[c['case'] for c in cases]==inputs
checks['raw_count_expected']=len(raw)==48*7*5
checks['all_method_case_rep_keys_exact']=collections.Counter((r['case'],r['rep'],r['method']) for r in raw)==collections.Counter((c['id'],rep,method) for c in inputs for rep in range(7) for method in methods)
checks['repeats_match']=manifest['repeat_count']==7==protocol['timing']['repetitions']
checks['fixed_problem_family_matches_runner']=all(c['period']==2.4 and c['layer_height']==.3 and c['material_index']==2. and c['target_order']==1 for c in inputs)
checks['mode_strata_exact']=dict(collections.Counter(str(c['harmonics']) for c in inputs))==protocol['harmonics']
checks['raw_rows_match_case_observations']=collections.Counter(json.dumps(r,sort_keys=True) for r in raw)==collections.Counter(json.dumps(r,sort_keys=True) for c in cases for vv in c['observations'].values() for r in vv)
rng=random.Random(37832);expected_order=[]
for c in inputs:
 for rep in range(7):
  order=methods.copy();rng.shuffle(order);expected_order.extend((c['id'],rep,m) for m in order)
checks['exact_preregistered_method_shuffle']=[(r['case'],r['rep'],r['method']) for r in raw]==expected_order
baseline=json.loads((base/'baseline-files.json').read_text())
checks['legacy_physics_equals_original_snapshot']=baseline['photonsmith/physics.py']==manifest['sources']['photonsmith/physics.py']
source_checks={p:{'recorded':digest,'current':sha(repo/p),'matches':digest==sha(repo/p)} for p,digest in manifest['sources'].items()}
# Preserve the exact runner if still present; a changed current runner is not a
# historical-run failure if it has deliberately become a new v2 candidate.
for p,item in source_checks.items():
 if item['matches'] and p.endswith('.py'):
  q=review/(Path(p).stem+'-'+item['recorded']+'.py')
  if not q.exists():q.write_bytes((repo/p).read_bytes())
checks['native_receipt_source_matches_run']=manifest['native_library']['source_sha256']==manifest['sources']['native/boundary.cpp']
checks['current_library_matches_run_receipt']=sha(repo/'photonsmith/_boundary_native.dylib')==manifest['native_library']['library_sha256']
errors=[r for r in raw if 'error' in r]
checks['no_failed_attempts']=not errors
checks['all_recorded_outputs_finite']=all(np.isfinite(r.get('value',np.nan)) and np.all(np.isfinite(r.get('gradient',[np.nan]))) for r in raw)
checks['all_times_positive']=all(r.get('wall',0)>0 and r.get('cpu',0)>0 for r in raw)
max_value={m:0. for m in methods};max_gradient={m:0. for m in methods};normalized_gradient={m:0. for m in methods}
ratios={m:[] for m in methods if m!='native'};cpu_ratios={m:[] for m in methods if m!='native'};strata={};per_case=[]
fd_results=[]
bykey={(r['case'],r['rep'],r['method']):r for r in raw}
for index,c in enumerate(inputs):
 ident=c['id'];native=bykey[(ident,0,'native')];ng=np.asarray(native['gradient'])
 row={'case':ident,'harmonics':c['harmonics'],'paired_medians':{}}
 for method in methods:
  observations=[bykey[(ident,rep,method)] for rep in range(7)]
  for r in observations:
   max_value[method]=max(max_value[method],abs(r['value']-native['value']))
   gg=np.asarray(r['gradient']);max_gradient[method]=max(max_gradient[method],float(np.max(abs(gg-ng))))
   normalized_gradient[method]=max(normalized_gradient[method],float(np.max(abs(gg-ng)/np.maximum(1,abs(ng)))))
  if method!='native':
   pair=[bykey[(ident,rep,method)]['wall']/bykey[(ident,rep,'native')]['wall'] for rep in range(7)]
   cp=[bykey[(ident,rep,method)]['cpu']/bykey[(ident,rep,'native')]['cpu'] for rep in range(7)]
   ratio=float(np.median(pair));ratios[method].append(ratio);cpu_ratios[method].append(float(np.median(cp)));row['paired_medians'][method]=ratio
 if index<8:
  p=np.array(c['params']);safe=bool(np.all(p[:3]-1e-4>=.15) and np.all(p[:3]+1e-4<=.85) and np.all(p[3:]-1e-4>=0) and np.all(p[3:]+1e-4<=1))
  for fd in cases[index]['verification']['finite_differences']:
   gg=np.asarray(fd['gradient']);fd_results.append({'case':ident,'step':fd['step'],'boundary_safe_at_max_step':safe,'max_abs_error':float(np.max(abs(gg-ng))),'max_componentwise_normalized_error':float(np.max(abs(gg-ng)/np.maximum(1,abs(ng)))),'normwise_relative_error':float(np.linalg.norm(gg-ng)/max(1,np.linalg.norm(ng)))})
 per_case.append(row)

rg=np.random.default_rng(37833);sample_indices=rg.integers(0,48,size=(10000,48))
summary={}
for method,values in ratios.items():
 x=np.log(values);boot=np.exp(x[sample_indices].mean(axis=1))
 by_m={}
 for m in [3,5,9,15]:
  subset=[v for c,v in zip(inputs,values) if c['harmonics']==m];by_m[str(m)]=float(np.exp(np.mean(np.log(subset))))
 summary[method]={'geomean_speedup':float(np.exp(x.mean())),'bootstrap95_interval':np.quantile(boot,[.025,.975]).tolist(),'case_count':48,'min_case_median_ratio':min(values),'max_case_median_ratio':max(values),'geomean_cpu_ratio':float(np.exp(np.mean(np.log(cpu_ratios[method])))),'by_harmonics':by_m}
external=[m for m in ratios if m!='legacy'];fastest=min(external,key=lambda m:summary[m]['geomean_speedup'])
oracle=np.min(np.array([ratios[m] for m in external]),axis=0)
oracle_boot=np.exp(np.log(oracle)[sample_indices].mean(axis=1))
verification={'max_all_order_power_error_meent_numpy':max(c['verification']['max_power_error'] for c in cases),'max_balance_error':max(c['verification']['balance_error'] for c in cases),'max_residual':max(c['verification']['residual'] for c in cases),'max_value_disagreement_vs_native_by_method':max_value,'max_absolute_gradient_disagreement_by_method':max_gradient,'max_componentwise_normalized_gradient_disagreement_by_method':normalized_gradient,'finite_difference_records':fd_results}
result={'checks':checks,'source_checks':source_checks,'evidence_sha256':{p:sha(ev/p) for p in ['manifest.json','raw.jsonl','cases.json']},'input_sha256':sha(base/'protocol/confirmation-cases-v2.json'),'protocol_sha256':sha(base/'protocol/PROTOCOL-v2.json'),'raw_count':len(raw),'failure_count':len(errors),'summary':summary,'fastest_tested_external':fastest,'fastest_tested_external_speedup':summary[fastest],'per_case_fastest_external_oracle':{'geomean_ratio':float(np.exp(np.mean(np.log(oracle)))),'bootstrap95_interval':np.quantile(oracle_boot,[.025,.975]).tolist()},'verification':verification,'gates_for_these_rows':{'legacy_geomean_ge_5':summary['legacy']['geomean_speedup']>=5,'fastest_tested_external_geomean_ge_2':summary[fastest]['geomean_speedup']>=2,'fastest_tested_external_lower_gt_1':summary[fastest]['bootstrap95_interval'][0]>1},'bootstrap_algorithm':'NumPy default_rng(37833), 10000 shared paired resamples of48case log-medians, percentile2.5/97.5','limits':'Saved-row analysis only; no solver or benchmark rerun. Local synthetic six-parameter family, shared host, finite comparators and local predeclaration; not external-human validation.'}
(review/'benchmark-v2-independent-recomputation.json').write_text(json.dumps(result,indent=2)+'\n')
(review/'benchmark-v2-case-ratios.json').write_text(json.dumps(per_case,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['verification','source_checks']},indent=2))
print('verification maxima',json.dumps({k:v for k,v in verification.items() if k!='finite_difference_records'},indent=2))
print('max FD normalized errors by step', {step:max(r['max_componentwise_normalized_error'] for r in fd_results if r['step']==step) for step in [1e-4,1e-5,1e-6]})
