#include "/Users/nazeeh/Documents/ChatGPT/Continue/engineering-accomplishment-2026-10-29/correction-v2/PhotonSmith/native/boundary.cpp"
extern "C" void review_pair(double left,double right,double h,double eta,double*out) {
  double d=left-right,gap=std::abs(d)*h*h;
  Modal f=modal(left,h,eta),g=modal(right,h,eta);
  C r,t;
  if(d==0) {r=f.dr;t=f.dt;}
  else if(gap<1e-5 || gap<1e-3*(std::sqrt(std::abs(left))*h+std::sqrt(std::abs(right))*h)) {
    Divided p=close_divided(left,right,h,eta);r=p.r;t=p.t;
  } else {r=(f.r-g.r)/d;t=(f.t-g.t)/d;}
  out[0]=r.real();out[1]=r.imag();out[2]=t.real();out[3]=t.imag();
}
