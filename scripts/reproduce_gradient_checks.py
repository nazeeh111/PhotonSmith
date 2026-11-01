"""Recompute independent gradient checks without installing JAX/Torch/grcwa.

Run from a source checkout after installing PhotonSmith and explicitly building
its optional native kernel. No saved finite-difference values are reused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from photonsmith import native
from photonsmith.gradient_check import dense_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-id',action='append',default=[])
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a fresh path')
    encoded=args.cases.read_bytes();planned=json.loads(encoded)
    ids=[c['id'] for c in planned]
    if not ids or len(set(ids))!=len(ids):parser.error('Case IDs must be nonempty and unique')
    if set(args.case_id)-set(ids):parser.error('Requested case is absent from the input plan')
    selected=[c for c in planned if not args.case_id or c['id'] in args.case_id]
    records=[]
    for c in selected:
        if (c['period']!=2.4 or c['layer_height']!=.3 or c['material_index']!=2.
                or c['target_order']!=1):
            parser.error('This command checks the declared fixed benchmark physical model')
        value,g=native.objective_gradient(c['params'],wavelength=c['wavelength'],harmonics=c['harmonics'])
        checks=dense_checks(c['params'],c['wavelength'],c['harmonics'])
        errors=[]
        for check in checks:
            fd=np.asarray(check['gradient']);difference=abs(g-fd)
            errors.append(max(float(np.max(difference/np.maximum(1,abs(g)))),
                              float(np.max(difference/np.maximum(1,abs(fd))))))
        records.append({'case':c,'native_value':float(value),'native_gradient':g.tolist(),
                        'independent_checks':checks,'max_normalized_errors':errors,
                        'pass':max(errors)<=2e-5})
    result={'input_sha256':hashlib.sha256(encoded).hexdigest(),'planned_cases':len(planned),
            'selected_cases':len(selected),'complete_input_plan_checked':len(selected)==len(planned),
            'scope':'New actual native gradients vs dense forward-only fourth-order estimates; no performance or external-human-validation claim',
            'source_sha256':{name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                             for name in ['photonsmith/gradient_check.py','photonsmith/physics.py',
                                          'photonsmith/native.py','native/boundary.cpp']},
            'limit':2e-5,'all_selected_cases_pass':all(x['pass'] for x in records),'records':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    return 0 if result['all_selected_cases_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
