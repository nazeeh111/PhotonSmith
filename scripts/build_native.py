"""Build the optional exact CPU kernel using an existing C++ compiler/LAPACK.

No downloads or global installation. Output is local to this source checkout.
"""
import argparse,hashlib,json,os,platform,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path)
p.add_argument('--sanitize',action='store_true')
a=p.parse_args();root=Path(__file__).resolve().parents[1]
compiler=shutil.which(os.environ.get('CXX','clang++')) or shutil.which('c++')
if not compiler:raise SystemExit('A C++17 compiler is required; no compiler was installed')
suffix='.dylib' if platform.system()=='Darwin' else '.so'
out=a.output or root/'photonsmith'/('_boundary_native'+suffix)
if out.exists():raise SystemExit('Existing library preserved; choose a fresh output')
out.parent.mkdir(parents=True,exist_ok=True)
flags=['-std=c++17','-O3','-DNDEBUG','-fPIC','-shared']
if a.sanitize:flags=['-std=c++17','-O1','-g','-fPIC','-shared','-fsanitize=address,undefined','-fno-omit-frame-pointer']
cmd=[compiler,*flags,str(root/'native/boundary.cpp'),'-o',str(out)]
cmd+=['-framework','Accelerate'] if platform.system()=='Darwin' else ['-llapack','-lblas']
r=subprocess.run(cmd,text=True,capture_output=True)
print(r.stdout,end='');print(r.stderr,end='')
if r.returncode:raise SystemExit(r.returncode)
receipt={'command':cmd,'compiler':subprocess.check_output([compiler,'--version'],text=True),'source_sha256':hashlib.sha256((root/'native/boundary.cpp').read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'platform':platform.platform()}
out.with_suffix(out.suffix+'.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(out)
