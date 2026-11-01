#!/usr/bin/env python3
"""Independent array and gate review; no imports of producer or Meep, no simulation."""
from pathlib import Path
import json,hashlib,datetime,cmath,math
import numpy as np
TASK=Path(__file__).resolve().parents[2]
ROOT=TASK/'validation-v4/meep/run-40-80-160'
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
plan=load(ROOT/'plan.json'); cfg=plan['normalized_config']; th=plan['thresholds']; journal_blob=(ROOT/'journal.json').read_bytes();journal=json.loads(journal_blob)
checks=[]
def check(name,ok,**info):
    checks.append(dict(name=name,pass_=bool(ok),**info))
def close(name,a,b,tol=1e-11):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    check(name,err<=tol,error=err,tolerance=tol)
def cp(x):return np.array(x['real'])+1j*np.array(x['imag'])
P=cfg['period']; h=3*cfg['layer_height']; wl=cfg['wavelength']; k0=2*np.pi/wl
orders=np.arange(-math.floor(P/wl),math.floor(P/wl)+1); kz=np.sqrt(k0*k0-(2*np.pi*orders/P)**2)
expected={'vacuum':(160,'vacuum',0,1,1.5,250),'slab':(160,'slab',0,1,1.5,250),'coarse':(40,'design',0,1,1.5,250),'medium':(80,'design',0,1,1.5,250),'fine':(160,'design',0,1,1.5,250),'long-time':(80,'design',0,1,1.5,500),'thick-pml':(80,'design',0,2,1.5,250),'wide-air':(80,'design',0,1,3,250),'translated':(80,'design',.137,1,1.5,250)}
for key,path in [('config_sha256',ROOT/'input.json'),('native_sha256',ROOT/'native-reference.json'),('helper_sha256',Path('/Users/nazeeh/.codex/skills/meep-local/scripts/meep_grating.py')),('validator_sha256',Path('/Users/nazeeh/.codex/skills/meep-local/scripts/validate_grating.py'))]:
    check('frozen_'+key,sha(path)==plan[key],observed=sha(path),expected=plan[key])
check('frozen_protocol',plan['resolutions']==[40,80,160] and th['native_power']==.005 and th['resolution']==.0025 and plan['translation_fraction']==.137)
expected_limits={'power_control':.005,'phase_control':.05,'vacuum_spurious':.001,'resolution':.0025,'balance':.005,'negative_power':-.001,'time':.002,'pml':.002,'air':.003,'flux':.005,'translation_power':.005,'translation_phase':.05,'phase_power_floor':.01,'native_power':.005,'native_phase':.1,'native_amplitude':.05}
check('all_frozen_limits_match_requested_protocol',th==expected_limits)
runtime=load(ROOT/'runtime-files.json')
runtime_digest=hashlib.sha256(json.dumps(runtime['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
check('runtime_manifest_self_consistency',runtime_digest==runtime['digest']==plan['runtime_files_digest'] and len(runtime['files'])==runtime['file_count']==plan['runtime_file_count'])
public=Path('/private/tmp/photon-corrected-final-public-20261101')
for rel in ['native/boundary.cpp','photonsmith/native.py','photonsmith/physics.py','photonsmith/boundary.py']:
    check('unchanged_public_source_'+rel,sha(public/rel)==sha(TASK/'validation-v4/PhotonSmith'/rel),sha256=sha(public/rel))
native=load(ROOT/'native-reference.json')
check('native_configuration',native['config']==cfg and load(ROOT/'input.json')['target_order']==-1 and load(ROOT/'input.json')['polarization']=='TE')
end_to_end=load(TASK/'evidence/corrected-end-to-end-v1.json')
origin=[a['result'] for c in end_to_end['results'] if c['case']['id']=='new-index-1730' for a in c['attempts'] if a['method']=='native' and a['rep']==0]
check('actual_optimized_device_provenance',len(origin)==1 and all(origin[0][k]==v for k,v in cfg.items()) and origin[0]['target_order']==-1,source_sha256=sha(TASK/'evidence/corrected-end-to-end-v1.json'))
for kind,path,key in [('source',TASK/'validation-v4/PhotonSmith/native/boundary.cpp','source_sha256'),('library',TASK/'validation-v4/PhotonSmith/photonsmith/_boundary_native.dylib','library_sha256')]:
    check('native_'+kind+'_receipt',sha(path)==native['native_library_receipt'][key],observed=sha(path))
fc=native['fourier_convergence']; stored=fc['all_results'];check('native_convergence_orders',[x['harmonics'] for x in stored]==[21,31,45,61])
ns=[]
for item in stored:
    idx=[item['orders'].index(int(m)) for m in orders]
    ns.append(np.r_[np.array(item['T'])[idx],np.array(item['R'])[idx]])
conv=float(np.max(np.abs(ns[-1]-ns[-2])));close('native_stored_convergence_metric',conv,fc['max_open_power_difference_45_vs_61']);check('native_fourier_limit',conv<=.0001,error=conv)
for key in ['orders','T','R','amplitudes_t']:close('native_final_matches_M61_'+key,native[key],stored[-1][key])
ni=[native['orders'].index(int(m)) for m in orders];ntc=np.array(native['amplitudes_t'])[ni]@np.array([1,1j])
close('native_power_amplitude_consistency',np.array(native['T'])[ni],abs(ntc)**2*kz/k0)
completed={}; projections={}; run_metrics={}
for name in expected:
    rows=[x for x in journal if x['name']==name and x.get('returncode')==0]
    if not rows:continue
    row=rows[-1]
    for rel,digest in row['sha256'].items():check(name+'_hash_'+Path(rel).name,(ROOT/rel).is_file() and sha(ROOT/rel)==digest)
    path=ROOT/row['output'];d=load(path)
    check(name+'_required_artifact_hashes',all(str(path.with_name(n).relative_to(ROOT)) in row['sha256'] for n in ['result.json','result.npz','stdout.log','stderr.log']))
    res,case,shift,pml,air,after=expected[name]
    extras=[]
    if case!='design':extras=['--case',case]
    elif after!=250:extras=['--after-sources',str(after)]
    elif pml!=1:extras=['--pml',str(pml)]
    elif air!=1.5:extras=['--air',str(air)]
    elif shift:extras=['--translation','.137']
    command=[plan['python'],'-I','-B','/Users/nazeeh/.codex/skills/meep-local/scripts/meep_grating.py',str(ROOT/'input.json'),'--output',str(path),'--resolution',str(res),*extras]
    check(name+'_journal_command',row['command']==command)
    check(name+'_identity',d['original_config']==cfg and d['orders']==orders.tolist() and d['meep_version']=='1.35.0' and d['resolution_pixels_per_um']==res and d['case']==case and d['translation_fraction']==shift and d['pml_um']==pml and d['air_um']==air and d['time_after_sources_um_per_c']==after)
    check(name+'_waveform',d['source_waveform']==dict(kind='GaussianSource',frequency=1/wl,fwidth=.15*(1/wl),cutoff=6))
    physical=dict(cfg);physical['params']=cfg['params'][:3]+[(s+shift)%1 for s in cfg['params'][3:]]
    check(name+'_actual_geometry',d['config']==physical)
    coords=d['coordinates'];close(name+'_physical_planes',[coords['source_z_um'],coords['reflection_z_um'],coords['transmission_z_um'],coords['transmission_reference_z_um'],coords['incident_reference_z_um']],[-.75*air,-.35*air,h+.35*air,h,0])
    a=np.load(path.with_suffix('.npz'),allow_pickle=False);pr={}
    for side in ['reference','device']:
        for typ in ['re','rh','te','th']:
            tag=side+'_'+typ; x=a[tag+'_x']+P/2;w=a[tag+'_weights'];f=a[tag+'_field']
            check(name+'_'+tag+'_layout',f.ndim==1 and f.shape==x.shape==w.shape and np.isfinite(f).all() and np.all(w>=0))
            close(name+'_'+tag+'_weight_sum',w.sum(),P)
            # Integrate each Fourier coefficient directly, using returned DFT coordinates.
            val=np.array([np.vdot(np.exp(2j*np.pi*m*x/P),w*f)/P for m in orders]);pr[tag]=val
            close(name+'_'+tag+'_projected',val,a[side+'_projected_'+typ])
        for et,ht in [('re','rh'),('te','th')]:
            close(name+'_'+side+'_'+et+'_common_EH_coordinates',a[side+'_'+et+'_x'],a[side+'_'+ht+'_x'])
            close(name+'_'+side+'_'+et+'_common_EH_weights',a[side+'_'+et+'_weights'],a[side+'_'+ht+'_weights'])
    zero=orders.tolist().index(0);inc=float((pr['reference_re'][zero]*np.conj(pr['reference_rh'][zero])).real)
    check(name+'_positive_incident',inc>0);close(name+'_incident_power',inc,d['reference_projected_incident_power'])
    er=pr['device_re']-pr['reference_re'];hr=pr['device_rh']-pr['reference_rh']
    T=(pr['device_te']*np.conj(pr['device_th'])).real/inc;R=-(er*np.conj(hr)).real/inc
    A0=pr['reference_re'][zero]*np.exp(-1j*k0*coords['reflection_z_um'])
    tc=pr['device_te']*np.exp(-1j*kz*(coords['transmission_z_um']-h))/A0
    rc=er*np.exp(1j*kz*coords['reflection_z_um'])/A0
    for key,val in [('T',T),('R',R),('transmission',T.sum()),('reflection',R.sum()),('balance',T.sum()+R.sum())]:close(name+'_reconstructed_'+key,val,d[key])
    close(name+'_deembedded_t',tc,cp(d['complex_T_at_z_device_end_incident_at_z0']));close(name+'_deembedded_r',rc,cp(d['complex_R_at_z0_incident_at_z0']));close(name+'_deembedded_incident',A0,cp(d['complex_incident_at_z0']))
    # Independent real-space Poynting integration from unprojected complex arrays.
    raw_inc=float(np.real(np.sum(a['reference_re_weights']*a['reference_re_field']*np.conj(a['reference_rh_field']))))
    raw_t=float(np.real(np.sum(a['device_te_weights']*a['device_te_field']*np.conj(a['device_th_field']))))/raw_inc
    raw_r=-float(np.real(np.sum(a['device_re_weights']*(a['device_re_field']-a['reference_re_field'])*np.conj(a['device_rh_field']-a['reference_rh_field']))))/raw_inc
    fluxerr=max(abs(T.sum()-d['integrated_flux_T']),abs(R.sum()-d['integrated_flux_R']))
    check(name+'_projection_flux_gate',fluxerr<=th['flux'],error=fluxerr)
    check(name+'_realspace_flux_gate',max(abs(raw_t-d['integrated_flux_T']),abs(raw_r-d['integrated_flux_R']))<=th['flux'],raw_T=raw_t,raw_R=raw_r)
    completed[name]=d;projections[name]=(T,R,tc,rc);run_metrics[name]=dict(T=T.tolist(),R=R.tolist(),balance=float(T.sum()+R.sum()),flux_error=fluxerr,empty_flux_ratio=d['empty_integrated_flux_ratio'],elapsed_seconds=row['elapsed_seconds'])
    if name in ['vacuum','slab']:
        n=1 if name=='vacuum' else cfg['material_index'];delta=k0*n*h;q=(1-n)/(1+n);u=np.exp(2j*delta)
        at=4*n*np.exp(1j*delta)/((n+1)**2-(n-1)**2*u);ar=q*(1-u)/(1-q*q*u)
        ep=max(np.max(abs(T-np.where(orders==0,abs(at)**2,0))),np.max(abs(R-np.where(orders==0,abs(ar)**2,0))))
        pe=[abs(cmath.phase(tc[zero]/at))]+([abs(cmath.phase(rc[zero]/ar))] if abs(ar)>1e-12 else [])
        check(name+'_analytic_control',ep<=th['power_control'] and max(pe)<=th['phase_control'],power_error=float(ep),phase_errors=pe,analytic_t=[at.real,at.imag],analytic_r=[ar.real,ar.imag])
        spur=max(abs(T[orders!=0]).max(),abs(R[orders!=0]).max());check(name+'_spurious_order_gate',spur<=th['vacuum_spurious'],error=float(spur))
for gate,other in [('resolution','fine'),('time','long-time'),('pml','thick-pml'),('air','wide-air')]:
    if 'medium' in projections and other in projections:
        a=projections['medium'];b=projections[other];v=float(np.max(np.abs(np.r_[a[0],a[1]]-np.r_[b[0],b[1]])))
        check(gate+'_gate',v<=th[gate],error=v,limit=th[gate])
if 'fine' in projections:
    T,R,tc,rc=projections['fine'];check('fine_energy_balance',abs(T.sum()+R.sum()-1)<=th['balance'] and min(np.r_[T,R])>=th['negative_power'],error=float(abs(T.sum()+R.sum()-1)))
    idx=[native['orders'].index(int(m)) for m in orders];nt=np.array(native['T'])[idx];nr=np.array(native['R'])[idx];nc=np.array(native['amplitudes_t'])[idx]@np.array([1,1j])
    err=float(np.max(np.abs(np.r_[T,R]-np.r_[nt,nr])));check('native_power_gate',err<=th['native_power'],error=err,limit=th['native_power'])
    for i,m in enumerate(orders):
        if nt[i]>=th['phase_power_floor']:
            pe=abs(cmath.phase(tc[i]/nc[i]));ae=abs(tc[i]-nc[i]);check('native_complex_transmission_'+str(m),pe<=th['native_phase'] and ae<=th['native_amplitude'],phase_error=pe,amplitude_error=ae)
if 'medium' in projections and 'translated' in projections:
    a=projections['medium'];b=projections['translated'];err=float(np.max(np.abs(np.r_[a[0],a[1]]-np.r_[b[0],b[1]])));check('translation_power_gate',err<=th['translation_power'],error=err)
    for side in [0,1]:
        for i,m in enumerate(orders):
            if a[side][i]>=th['phase_power_floor']:
                pe=abs(cmath.phase(b[side+2][i]/(a[side+2][i]*np.exp(-2j*np.pi*m*.137))));check('translation_phase_'+str(side)+'_'+str(m),pe<=th['translation_phase'],error=pe)
missing=[n for n in expected if n not in completed]
if not missing:
    check('producer_final_summary_exists',(ROOT/'summary.json').is_file())
    if (ROOT/'summary.json').is_file():
        summary=load(ROOT/'summary.json')
        check('producer_final_supported_decision',summary.get('decision')=='supported',decision=summary.get('decision'))
        check('producer_final_plan_identity',summary.get('plan_sha256')==sha(ROOT/'plan.json'))
        check('producer_final_nine_stage_identities',set(summary.get('accepted_runs',{}))==set(expected) and all(summary['accepted_runs'][n]==[x for x in journal if x['name']==n and x.get('returncode')==0][-1] for n in expected))
    # Recheck all recorded installed-runtime bytes after successful completion.
    prefix=Path(plan['python']).parent.parent
    runtime_fail=[]
    for rel,identity in runtime['files'].items():
        p=prefix/rel
        if isinstance(identity,dict):
            import os
            ok=p.is_symlink() and os.readlink(p)==identity['link'] and (sha(p) if p.is_file() else None)==identity['target_sha256']
        elif identity=='directory':ok=p.is_dir() and not p.is_symlink()
        else:ok=p.is_file() and not p.is_symlink() and sha(p)==identity
        if not ok:runtime_fail.append(rel)
    check('final_live_runtime_file_identities',not runtime_fail,file_count=len(runtime['files']),mismatches=runtime_fail)
    additions=sorted(str(p.relative_to(prefix)) for p in prefix.rglob('*') if 'codex-provenance' not in p.relative_to(prefix).parts and str(p.relative_to(prefix)) not in runtime['files'])
    check('final_live_runtime_no_additions',not additions,additions=additions)
failed=[c for c in checks if not c['pass_']]
report=dict(status='incomplete' if missing else ('pass' if not failed else 'fail'),reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),journal_sha256=hashlib.sha256(journal_blob).hexdigest(),plan_sha256=sha(ROOT/'plan.json'),completed=list(completed),missing=missing,checks_count=len(checks),failed=failed,run_metrics=run_metrics,checks=checks,limitations=['Coding-agent numerical review, not external human validation.','No physical device, fabrication, finite-3D, absorption or dispersion validation.','No new simulation launched by review.'])
report.update(review_script_sha256=sha(__file__),producer_summary_sha256=sha(ROOT/'summary.json') if (ROOT/'summary.json').is_file() else None)
(OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','completed','missing','checks_count','failed']},indent=2))
