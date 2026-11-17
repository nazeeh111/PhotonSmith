from pathlib import Path
import json,hashlib,sys,re
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
R=Path(__file__).resolve().parents[2]
MOBILE='--mobile' in sys.argv
plt.rcParams.update({'font.family':['DejaVu Sans','Arial','sans-serif'],'font.size':15,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':13,'svg.fonttype':'none','figure.facecolor':'#ffffff','axes.facecolor':'#ffffff','text.color':'#172b3a','axes.labelcolor':'#172b3a','xtick.color':'#334a5b','ytick.color':'#334a5b','axes.edgecolor':'#8a9aa7'})
manifest={'method':'Matplotlib source-derived figures from exact retained JSON; no solver, inference, new experiment, image enhancement or smoothing','runtime':{'python':sys.version,'matplotlib':matplotlib.__version__,'numpy':np.__version__},'font_policy':['DejaVu Sans','Arial','sans-serif'],'figures':[]}
def save(n,fig,stem,inputs,transforms):
 fig.get_layout_engine().set(rect=(0,0 if MOBILE else .13,1,.94 if MOBILE else .80))
 if MOBILE:stem+='-mobile'
 d=R/'docs/presentation';d.mkdir(parents=True,exist_ok=True);outputs=[]
 for ext in ['png','svg']:
  p=d/(stem+'.'+ext);fig.savefig(p,dpi=160,facecolor='white');outputs.append({'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 if n in ['InstrumentZero','PhotonSmith']:
  series=signal if n=='InstrumentZero' else spectrum
  gid='source-signal' if n=='InstrumentZero' else 'source-spectrum'
  xml=ET.fromstring((d/(stem+'.svg')).read_text());path=xml.find('.//s:g[@id="'+gid+'"]/s:path',{'s':'http://www.w3.org/2000/svg'})
  numbers=[float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',path.attrib['d'])]
  expected=series.get_transform().transform(series.get_xydata());scale=72/fig.dpi
  expected=[(x*scale,(fig.bbox.height-y)*scale) for x,y in expected]
  required=769 if n=='InstrumentZero' else 21
  assert len(expected)==required and len(numbers)==required*2,(n,len(numbers))
  error=max(abs(numbers[2*i+j]-point[j]) for i,point in enumerate(expected) for j in [0,1])
  assert error<0.00001,(n,error)
  manifest['source_to_svg']={'series':gid,'source_samples':required,'exported_vertices':len(numbers)//2,'maximum_svg_coordinate_error':error,'path_simplify':matplotlib.rcParams['path.simplify']}
 manifest['figures'].append({'repository':n,'inputs':[{'path':str(p.relative_to(R)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],'transformations':transforms,'outputs':outputs});plt.close(fig)
# Actual stored solved field and wavelength spectrum, same geometry, exact arrays.
p=R/'experiments/designs/interactive-refined-1550nm.json';v=json.loads(p.read_text());a=v['result'];field=a['field'];z=np.array(field['z']);x=np.array(field['x']);real=np.array(field['real']);eps=np.array(field['epsilon']);wl=np.array(a['spectrum']['wavelength']);power=np.array(a['spectrum']['efficiency'])*100
assert real.shape==(len(z),len(x)) and len(wl)==len(power)
fig,axs=plt.subplots(2 if MOBILE else 1,1 if MOBILE else 2,figsize=(6.4,8.8) if MOBILE else (13.2,5.1),gridspec_kw={'height_ratios':[1.3,1]} if MOBILE else {'width_ratios':[1.4,1]},layout='constrained');fig.suptitle('A solved field. A spectrum.' if MOBILE else 'A grating, its solved field, and its spectrum.',fontsize=19 if MOBILE else 22,fontweight='bold')
vmax=float(abs(real).max());im=axs[0].pcolormesh(x,z,real,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='nearest',rasterized=True);axs[0].contour(x,z,eps,levels=[2.5],colors=['#172b3a'],linewidths=1);axs[0].set_aspect('equal');axs[0].invert_yaxis();axs[0].set(xlabel='x (µm)',ylabel='z (µm)',title='1550 nm field, one periodic cell');c=fig.colorbar(im,ax=axs[0],shrink=.78);c.set_label('Re(Ey), unit incident amplitude',fontsize=12)
spectrum=axs[1].plot(wl*1000,power,'o-',color='#145b81',lw=2,markersize=4,gid='source-spectrum')[0];axs[1].set(xlabel='Wavelength (nm)',ylabel='Incident power into order +1 (%)',ylim=(0,100),title='21 stored wavelength solves');axs[1].grid(axis='y',alpha=.18); (fig.text if not MOBILE else lambda *a,**k:None)(.5,.025,'Stored M9 Maxwell simulation; search used M5. Numerical 2D model, not a fabricated-device measurement.',ha='center',fontsize=11)
save('PhotonSmith',fig,'field-and-spectrum',[p],['Stored real field displayed without filtering, normalization or interpolation.','Symmetric diverging color limits around zero; equal x/z physical scale.','epsilon=2.5 contour marks binary air/dielectric boundary.','Spectrum efficiency multiplied by 100; all 21 samples retained, lines connect sampled solves.','M13 independent validation remains separately labeled in README.'])

(R/('docs/presentation/manifest-mobile.json' if MOBILE else 'docs/presentation/manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print('Rendered figure and manifest from exact retained source arrays.')
