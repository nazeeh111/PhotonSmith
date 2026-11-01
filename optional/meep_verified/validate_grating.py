"""Run a frozen, bounded grating protocol; keep failures and refuse stale reuse."""
import sys
sys.dont_write_bytecode = True
import argparse,cmath,datetime,platform,fcntl,hashlib,json,math,os,resource,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meep_grating import load_config

DEFAULTS={'power_control':.005,'phase_control':.05,'vacuum_spurious':.001,'resolution':.005,'balance':.005,'negative_power':-.001,'time':.002,'pml':.002,'air':.003,'flux':.005,'translation_power':.005,'translation_phase':.05,'phase_power_floor':.01,'native_power':.01,'native_phase':.10,'native_amplitude':.05}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def pair(v): return [complex(x,y) for x,y in zip(v['real'],v['imag'])]
def diff(a,b): return max(abs(x-y) for x,y in zip(a,b))
def phase(a,b): return abs(cmath.phase(a/b))
def powers(d): return d['T']+d['R']
def runtime_files():
    prefix=Path(sys.prefix).resolve(); names=set()
    for metadata in sorted((prefix/'conda-meta').glob('*.json')):
        names.add(str(metadata.relative_to(prefix)))
        names.update(json.loads(metadata.read_text())['files'])
    if not names: raise ValueError('This helper requires the pinned conda runtime with package metadata.')
    values={}
    names.update(str(x.relative_to(prefix)) for x in prefix.rglob('*') if 'codex-provenance' not in x.relative_to(prefix).parts)
    for name in sorted(names):
        path=prefix/name
        if not path.resolve().is_relative_to(prefix): raise ValueError('Runtime file escapes pinned prefix: '+name)
        if path.is_symlink():
            values[name]={'link':os.readlink(path),'target_sha256':sha(path) if path.is_file() else None}
        elif path.is_file():values[name]=sha(path)
        elif path.is_dir():values[name]='directory'
        else:raise ValueError('Missing pinned runtime file: '+name)
    digest=hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'file_count':len(values),'digest':digest,'files':values}

def native_data(path,cfg):
    d=json.loads(path.read_text()); config=d.get('config',d.get('design'))
    if config is None: raise ValueError('Native reference needs config or design identifying its physical input.')
    if any(config.get(k)!=v for k,v in cfg.items()): raise ValueError('Native reference physical configuration differs from this input.')
    if 'results' in d:
        matches=[x for x in d['results'] if x.get('harmonics')==13 and x.get('status')=='executed']
        if len(matches)!=1: raise ValueError('Expected one executed native M13 reference.')
        n=matches[0]
    else: n=d
    for k in ['orders','T','R','amplitudes_t']:
        if k not in n: raise ValueError('Missing native reference field '+k)
    size=len(n['orders'])
    if not size or len(set(n['orders']))!=size or any(isinstance(x,bool) or not isinstance(x,int) for x in n['orders']): raise ValueError('Native orders must be unique integers.')
    if any(len(n[k])!=size for k in ['T','R','amplitudes_t']): raise ValueError('Native array lengths differ.')
    numbers=n['T']+n['R']
    for pair_value in n['amplitudes_t']:
        if not isinstance(pair_value,list) or len(pair_value)!=2:raise ValueError('Native complex amplitudes need real/imag pairs.')
        numbers+=pair_value
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in numbers):raise ValueError('Native values must be finite real numbers.')
    open_orders=[m for m in range(-math.floor(cfg['period']/cfg['wavelength']),math.floor(cfg['period']/cfg['wavelength'])+1) if abs(m*cfg['wavelength']/cfg['period'])<1-1e-12]
    if not set(open_orders).issubset(n['orders']):raise ValueError('Native reference is missing open diffraction orders.')
    return n

def evaluate(data,cfg,t,native):
    g={}; h=3*cfg['layer_height']; lam=cfg['wavelength']
    for case,n in [('vacuum',1.),('slab',cfg['material_index'])]:
        d=data[case]; delta=2*math.pi*n*h/lam; den=math.cos(delta)-1j*(n+1/n)*math.sin(delta)/2
        at=1/den; ar=1j*(n-1/n)*math.sin(delta)/2/den; z=d['orders'].index(0)
        errors=[abs(d[s][j]-(abs(a)**2 if m==0 else 0)) for s,a in [('T',at),('R',ar)] for j,m in enumerate(d['orders'])]
        ph=[phase(pair(d['complex_T_at_z_device_end_incident_at_z0'])[z],at)]
        if abs(ar)>1e-12: ph.append(phase(pair(d['complex_R_at_z0_incident_at_z0'])[z],ar))
        spur=max((abs(d[s][j]) for s in ['T','R'] for j,m in enumerate(d['orders']) if m),default=0)
        g[case]={'pass':max(errors)<=t['power_control'] and max(ph)<=t['phase_control'] and spur<=t['vacuum_spurious'] and d['reference_projected_incident_power']>0,'power_error':max(errors),'phase_errors':ph,'spurious_power':spur,'analytic_t':[at.real,at.imag],'analytic_r':[ar.real,ar.imag]}
    for name,a,b,key in [('resolution','medium','fine','resolution'),('time','medium','long-time','time'),('pml','medium','thick-pml','pml'),('air','medium','wide-air','air')]:
        delta=diff(powers(data[a]),powers(data[b]));g[name]={'pass':delta<=t[key],'max_channel_difference':delta,'limit':t[key]}
    d=data['fine'];g['balance']={'pass':abs(d['balance']-1)<=t['balance'] and min(powers(d))>=t['negative_power'],'balance_error':abs(d['balance']-1),'minimum_channel':min(powers(d))}
    flux={name:max(abs(v['transmission']-v['integrated_flux_T']),abs(v['reflection']-v['integrated_flux_R'])) for name,v in data.items()}
    g['flux']={'pass':max(flux.values())<=t['flux'],'per_run_error':flux}
    a,b=data['medium'],data['translated'];delta=diff(powers(a),powers(b));ph=[]
    for side,key in [('T','complex_T_at_z_device_end_incident_at_z0'),('R','complex_R_at_z0_incident_at_z0')]:
        ac,bc=pair(a[key]),pair(b[key])
        for j,m in enumerate(a['orders']):
            if a[side][j]>=t['phase_power_floor']: ph.append({'side':side,'order':m,'error':phase(bc[j],ac[j]*cmath.exp(-2j*math.pi*m*b['translation_fraction']))})
    g['translation']={'pass':delta<=t['translation_power'] and all(x['error']<=t['translation_phase'] for x in ph),'max_power_difference':delta,'phase_errors':ph}
    if native:
        vals=[];ph=[];tc=pair(d['complex_T_at_z_device_end_incident_at_z0'])
        for j,m in enumerate(d['orders']):
            k=native['orders'].index(m)
            for side in ['T','R']: vals.append({'side':side,'order':m,'external':d[side][j],'native':native[side][k],'difference':abs(d[side][j]-native[side][k])})
            nt=complex(*native['amplitudes_t'][k])
            ph.append({'order':m,'phase_error':phase(tc[j],nt) if abs(nt)>1e-14 else None,'amplitude_error':abs(tc[j]-nt),'required':native['T'][k]>=t['phase_power_floor']})
        g['native']={'pass':max(x['difference'] for x in vals)<=t['native_power'] and all(x['phase_error'] is not None and x['phase_error']<=t['native_phase'] and x['amplitude_error']<=t['native_amplitude'] for x in ph if x['required']),'powers':vals,'complex_transmission':ph}
    trusted=all(x['pass'] for k,x in g.items() if k!='native')
    decision='inconclusive' if not trusted else ('external_checks_passed_without_native_comparison' if not native else ('supported' if g['native']['pass'] else 'contradicted'))
    return {'decision':decision,'gates':g,'limits':'Bounded 2D numerical tolerance only; no rigorous error bound, fabrication or 3D certification.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--native-reference',type=Path);p.add_argument('--thresholds',type=Path);p.add_argument('--resolutions',type=int,nargs=3,default=[20,40,80]);p.add_argument('--timeout',type=float,default=3600);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not sys.flags.isolated:p.error('Run with the pinned Python executable and -I for isolated imports.')
    if any(k.startswith('DYLD_') or k in ['LD_LIBRARY_PATH','LD_PRELOAD'] for k in os.environ):p.error('Dynamic-loader environment overrides are unsupported; run in a clean process environment.')
    cfg=load_config(a.config)
    if not (10<=a.resolutions[0]<a.resolutions[1]<a.resolutions[2]<=400):p.error('Provide three strictly increasing resolutions in 10..400.')
    if not math.isfinite(a.timeout) or not 1<=a.timeout<=14400:p.error('Timeout must be finite, 1..14400 seconds.')
    thresholds=dict(DEFAULTS)
    if a.thresholds:
        custom=json.loads(a.thresholds.read_text());unknown=set(custom)-set(thresholds)
        if unknown:p.error('Unknown thresholds: '+','.join(sorted(unknown)))
        thresholds.update(custom)
    if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in thresholds.values()):p.error('Thresholds must be finite real numbers.')
    if any(v<0 for k,v in thresholds.items() if k!='negative_power') or thresholds['negative_power']>0:p.error('Thresholds must be nonnegative except negative_power <=0.')
    if any(abs(cfg['period']*res-round(cfg['period']*res))>1e-8 for res in a.resolutions):p.error('Every period × resolution must be an integer; choose commensurate resolutions.')
    native=native_data(a.native_reference,cfg) if a.native_reference else None
    root=a.output_dir.resolve();helper=Path(__file__).with_name('meep_grating.py').resolve()
    runtime=runtime_files()
    plan={'schema':2,'runtime_files_digest':runtime['digest'],'runtime_file_count':runtime['file_count'],'isolated_python':True,'platform':platform.platform(),'kernel':platform.uname().version,'config_sha256':sha(a.config),'normalized_config':cfg,'native_sha256':sha(a.native_reference) if a.native_reference else None,'helper_sha256':sha(helper),'validator_sha256':sha(__file__),'python':sys.executable,'python_sha256':sha(sys.executable),'runtime_metadata_sha256':{str(x.relative_to(Path(sys.prefix))):sha(x) for x in sorted((Path(sys.prefix)/'conda-meta').glob('*.json'))},'resolutions':a.resolutions,'thresholds':thresholds,'timeout_seconds':a.timeout,'base':{'pml':1.,'air':1.5,'after_sources':250.},'translation_fraction':.137}
    if a.resume:
        if not (root/'plan.json').exists():p.error('No frozen plan exists to resume.')
        if json.loads((root/'plan.json').read_text())!=plan:p.error('Input, code, thresholds, runtime or protocol differs from frozen plan; use a new directory.')
    else:
        root.mkdir(parents=True,exist_ok=False);save(root/'runtime-files.json',runtime);save(root/'plan.json',plan);(root/'input.json').write_bytes(a.config.read_bytes())
        if a.native_reference:(root/'native-reference.json').write_bytes(a.native_reference.read_bytes())
    lock=(root/'.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('This output directory has an active validation process.')
    if sha(root/'input.json')!=plan['config_sha256']:p.error('Frozen input bytes changed.')
    if native and sha(root/'native-reference.json')!=plan['native_sha256']:p.error('Frozen native reference bytes changed.')
    journal=json.loads((root/'journal.json').read_text()) if (root/'journal.json').exists() else []
    lo,mid,hi=a.resolutions
    runs=[('vacuum',hi,['--case','vacuum']),('slab',hi,['--case','slab']),('coarse',lo,[]),('medium',mid,[]),('fine',hi,[]),('long-time',mid,['--after-sources','500']),('thick-pml',mid,['--pml','2']),('wide-air',mid,['--air','3']),('translated',mid,['--translation','.137'])]
    env={k:v for k,v in os.environ.items() if not (k.startswith('DYLD_') or k.startswith('PYTHON') or k in ['LD_LIBRARY_PATH','LD_PRELOAD'])};env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(root/'mpl-cache'))
    for row in journal:
        if row.get('returncode') is None and row.get('pid'):
            try: os.kill(row['pid'],0)
            except ProcessLookupError: pass
            except PermissionError: p.error('Cannot verify the prior owned process has exited; do not resume yet.')
            else: p.error('A prior attempt process may still be active; wait or stop only that verified owned process before resuming.')
    if any(row.get('returncode') is None and 'elapsed_seconds' not in row for row in journal):
        p.error('Interrupted attempt elapsed time is unknown. Outputs are preserved; a new directory requires an explicit new budget decision.')
    spent=sum(x.get('elapsed_seconds',0) for x in journal)
    start=time.monotonic();data={};accepted={} 
    for name,res,extra in runs:
        old=[x for x in journal if x['name']==name and x.get('returncode')==0]
        if old:
            row=old[-1];out=root/row['output']
            if not all((root/k).is_file() and sha(root/k)==v for k,v in row['sha256'].items()):p.error('Previously completed output changed; preserve it and use a new directory.')
            data[name]=json.loads(out.read_text());accepted[name]=row;continue
        attempt=sum(x['name']==name for x in journal)+1;folder=root/'runs'/name/f'attempt-{attempt:03d}';folder.mkdir(parents=True,exist_ok=False);out=folder/'result.json'
        cmd=[sys.executable,'-I','-B',str(helper),str(root/'input.json'),'--output',str(out),'--resolution',str(res),*extra]
        row={'name':name,'attempt':attempt,'command':cmd,'started_utc':stamp(),'output':str(out.relative_to(root)),'returncode':None};journal.append(row);save(root/'journal.json',journal)
        remaining=a.timeout-spent-(time.monotonic()-start)
        if remaining<=0:row['returncode']='budget_exhausted';save(root/'journal.json',journal);break
        ts=time.monotonic()
        with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env)
            row['pid']=proc.pid;save(root/'journal.json',journal)
            try:row['returncode']=proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait();row['returncode']='timeout'
        row.update(elapsed_seconds=time.monotonic()-ts,child_maxrss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        row['sha256']={str(f.relative_to(root)):sha(f) for f in folder.iterdir() if f.is_file()};save(root/'journal.json',journal)
        if row['returncode']!=0:break
        data[name]=json.loads(out.read_text());accepted[name]=row
    if len(data)!=len(runs):
        save(root/'summary.json',{'decision':'incomplete','completed':list(data),'missing':[n for n,_,_ in runs if n not in data],'note':'Failures retained; --resume retries failed/incomplete stages into fresh attempt folders.'});return 2
    # Confirm every accepted file, command setting, physical input and runtime before qualification.
    for name,res,extra in runs:
        d=data[name];row=accepted[name];actual=d.get('original_config',d['config'])
        if actual!=cfg or d['resolution_pixels_per_um']!=res or row['returncode']!=0:raise RuntimeError('Run identity mismatch: '+name)
        if not all((root/k).is_file() and sha(root/k)==v for k,v in row['sha256'].items()):raise RuntimeError('Output identity changed: '+name)
        for flag,key,default in [('--pml','pml_um',1.),('--air','air_um',1.5),('--after-sources','time_after_sources_um_per_c',250.)]:
            expected=float(extra[extra.index(flag)+1]) if flag in extra else default
            if d[key]!=expected:raise RuntimeError('Run setting mismatch: '+name+' '+key)
        expected_case=extra[extra.index('--case')+1] if '--case' in extra else 'design'
        expected_shift=float(extra[extra.index('--translation')+1]) if '--translation' in extra else 0.
        if d['case']!=expected_case or d['translation_fraction']!=expected_shift:raise RuntimeError('Control/transformation identity mismatch: '+name)
        if d['meep_version']!='1.35.0':raise RuntimeError('Unexpected solver version; requalify runtime and update frozen protocol explicitly.')
    if runtime_files()['digest']!=plan['runtime_files_digest']:raise RuntimeError('Installed runtime files changed during execution.')
    if sha(a.config)!=plan['config_sha256'] or sha(helper)!=plan['helper_sha256'] or sha(__file__)!=plan['validator_sha256']:raise RuntimeError('Source/input changed during execution.')
    result=evaluate(data,cfg,thresholds,native);result.update(config=cfg,plan_sha256=sha(root/'plan.json'),accepted_runs=accepted,completed_utc=stamp())
    save(root/'summary.json',result);print(json.dumps({'decision':result['decision'],'output_dir':str(root)},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
