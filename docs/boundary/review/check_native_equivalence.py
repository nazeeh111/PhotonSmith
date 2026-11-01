from pathlib import Path
import json,sys
import numpy as np
base=Path(__file__).parent
sys.path.insert(0,str(base.parent/'PhotonSmith'))
from photonsmith import native,boundary,physics
records=[]
for m in [3,15,31]:
 p=np.array([.23,.64,.38,.13,.72,.44])
 ref=boundary.bar_solution(p,harmonics=m)
 n=2*m+1
 layers=[boundary.Layer(physics.epsilon_matrix(p[j],p[j+3],ref.orders),.3) for j in range(3)]
 actual=native.solve(layers,1.55,2.4,target_order=1)
 value,gradient=ref.transmission_gradient(1)
 records.append(dict(kind='bar_high_order',m=m,power_error=float(max(np.max(abs(actual.T-ref.T)),np.max(abs(actual.R-ref.R)))),amplitude_error=float(max(np.max(abs(actual.t-ref.t)),np.max(abs(actual.r-ref.r)))),gradient_error=float(np.max(abs(actual.gradient-np.array(gradient)))),state_error=float(np.max(abs(actual.state-ref.state)))))
for h in [.2,100.,1e4,1e6]:
 layers=[boundary.Layer(4*np.eye(7),h)]
 py=boundary.Solution(layers,1.55,2.4);pv,pg=py.transmission_gradient(0)
 c=native.solve(layers,1.55,2.4,target_order=0)
 records.append(dict(kind='thick_slab',h=h,power_error=float(max(np.max(abs(c.T-py.T)),np.max(abs(c.R-py.R)))),gradient_error=float(np.max(abs(c.gradient-np.array(pg))))))
# Invalid dimensions, loss, nonpositive operators and numerical phase guard.
refusals=[]
for name,layers,args in [
 ('even_order',[boundary.Layer(np.eye(4),.3)],{}),
 ('lossy_operator',[boundary.Layer(np.eye(3)*(1+1j),.3)],{}),
 ('indefinite_operator',[boundary.Layer(-np.eye(3),.3)],{}),
 ('nonfinite_operator',[boundary.Layer(np.eye(3)*np.nan,.3)],{}),
 ('oversized_phase',[boundary.Layer(4*np.eye(3),1e9)],{}),
 ('unsupported_target',[boundary.Layer(4*np.eye(3),.3)],{'target_order':5}),
 ('mismatched_shape',[boundary.Layer(np.eye(3),.3),boundary.Layer(np.eye(5),.3)],{})]:
 try:native.solve(layers,1.55,2.4,**args)
 except Exception as e:refusals.append({'case':name,'refused':True,'exception':type(e).__name__,'message':str(e)})
 else:refusals.append({'case':name,'refused':False})
result={'records':records,'refusals':refusals,'source_manifest':'native-reviewed-files.json','limits':'Numerical equivalence of native source/library to Python v3, not external-human validation or a speed measurement.'}
(base/'native-equivalence-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
