"""Use the optional exact native solver without changing legacy experiments."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from . import native
from .boundary import Layer
from .physics import DEFAULT_PARAMS,validate_params,epsilon_matrix
from .design import bounds_for,geometry_audit


def simulate(params=DEFAULT_PARAMS,wavelength=1.55,harmonics=5,material_index=2.):
    p=validate_params(params);m=native._int(harmonics,1,128,'harmonics')
    if not np.isfinite(material_index) or material_index<1:raise ValueError('Invalid material index')
    orders=np.arange(-m,m+1)
    layers=[Layer(epsilon_matrix(p[j],p[j+3],orders,material_index),.3) for j in range(3)]
    s=native.solve(layers,wavelength,2.4)
    return {'solver':'native-scattering-v1','params':p.tolist(),'wavelength':float(wavelength),
            'harmonics':m,'orders':orders.tolist(),'T':s.T.tolist(),'R':s.R.tolist(),
            'total_T':float(s.T.sum()),'total_R':float(s.R.sum()),'power_balance':float(s.T.sum()+s.R.sum()),
            'residual':s.residual,'material_index':float(material_index),'period':2.4,'layer_height':.3}


def optimize(params=DEFAULT_PARAMS,wavelength=1.55,target_order=1,iterations=20,
             harmonics=5,min_feature=.36,material_index=2.):
    m=native._int(harmonics,1,128,'harmonics');target=native._int(target_order,-m,m,'target_order')
    steps=native._int(iterations,1,10000,'iterations')
    bounds=bounds_for(min_feature);lo,hi=np.array(bounds).T
    p=np.clip(validate_params(params),lo,hi);start=perf_counter();history=[]
    first=simulate(p,wavelength,m,material_index);initial=first['T'][m+target]
    best=p.copy();best_value=initial
    def objective(p):
        nonlocal best,best_value
        v,g=native.objective_gradient(p,wavelength,target,m,material_index=material_index)
        history.append({'iteration':len(history)+1,'params':p.tolist(),'efficiency':float(v),'gradient':g.tolist(),'seconds':perf_counter()-start})
        if v>best_value:best=p.copy();best_value=v
        return -v,-g
    result=minimize(objective,best,jac=True,bounds=bounds,method='L-BFGS-B',
                    options={'maxiter':steps,'ftol':1e-8,'gtol':1e-5,'maxls':12})
    return {'solver':'native-scattering-v1','method':'adjoint','params':best.tolist(),
            'wavelength':float(wavelength),'target_order':target,'harmonics':m,
            'initial_efficiency':float(initial),'efficiency':float(best_value),
            'runtime_seconds':perf_counter()-start,'forward_evaluations':len(history)+1,
            'gradient_evaluations':len(history),'history':history,'optimizer_status':int(result.status),
            'optimizer_message':str(result.message),'constraint_audit':geometry_audit(best,min_feature)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['simulate','optimize']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--params',type=float,nargs=6,default=DEFAULT_PARAMS.tolist())
    p.add_argument('--wavelength',type=float,default=1.55);p.add_argument('--harmonics',type=int,default=5)
    p.add_argument('--target-order',type=int,default=1);p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--material-index',type=float,default=2.)
    args=p.parse_args(argv)
    if args.output.exists():p.error('Output exists; choose a fresh path')
    kwargs=dict(params=args.params,wavelength=args.wavelength,harmonics=args.harmonics,material_index=args.material_index)
    if args.action=='optimize':kwargs.update(target_order=args.target_order,iterations=args.iterations)
    result=(optimize if args.action=='optimize' else simulate)(**kwargs)
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(encoded)
    print(args.output)

if __name__=='__main__':main()
