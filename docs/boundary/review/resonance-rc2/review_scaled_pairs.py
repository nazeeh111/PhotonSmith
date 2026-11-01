"""Check nondimensionalization/roundoff at nonunit layer heights."""
import sys,ctypes as ct,json,math
from pathlib import Path
import numpy as np
import mpmath as mp
from review_actual_revision import boundary,OUT
from reproduce import mquot
mp.mp.dps=70
def main():
 lib=ct.CDLL(str(OUT/'review-pair.dylib'));lib.review_pair.argtypes=[ct.c_double]*4+[ct.c_void_p];lib.review_pair.restype=None
 rows=[]
 for h in [1e-4,.3,1.,3.3,1e4]:
  for a in [1e-8,1.,9.99999e7]:
   eta=a/h
   for z in [0.,1.,-1.,math.pi**2,(10*math.pi)**2]:
    for gaprel in [2e-16,1e-12,1e-8,8e-6]:
     gap=gaprel*max(1,abs(z));x=(z-gap/2)/(h*h);y=(z+gap/2)/(h*h)
     if x==y:continue
     ref=[complex(v) for v in mquot(x,y,h,eta)];r,t=boundary._loewner(np.array([x,y]),h,eta);vals=[r[0,1],t[0,1]];out=np.empty(4);lib.review_pair(x,y,h,eta,out.ctypes.data)
     for name,v in [('python',vals),('native',[complex(*out[:2]),complex(*out[2:])])]:
      for k in (0,1):
       error=abs(v[k]-ref[k]);rows.append({'backend':name,'h':h,'a':a,'z':z,'gaprel':gaprel,'component':k,'normalized_error':float(error/max(1,abs(ref[k]))),'relative_error':float(error/max(1e-300,abs(ref[k]))),'reference':[ref[k].real,ref[k].imag],'actual':[v[k].real,v[k].imag]})
 result={'count':len(rows),'worst_normalized':sorted(rows,key=lambda r:r['normalized_error'],reverse=True)[:15]}
 (OUT/'scaled-pair-review-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
