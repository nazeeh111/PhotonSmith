// Exact lossless scalar-TE scattering and matrix-function adjoint.
// Independently implemented for PhotonSmith. No eigenvector derivatives.
// Column-major BLAS/LAPACK internally; public complex arrays are row-major.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
using C = std::complex<double>;
using M = std::vector<C>;
constexpr double pi = 3.141592653589793238462643383279502884;
const C I(0,1);
extern "C" {
void zheev_(const char*,const char*,const int*,C*,const int*,double*,C*,const int*,double*,int*);
void zpotrf_(const char*,const int*,C*,const int*,int*);
void zgemm_(const char*,const char*,const int*,const int*,const int*,const C*,const C*,const int*,const C*,const int*,const C*,C*,const int*);
void zgbtrf_(const int*,const int*,const int*,const int*,C*,const int*,int*,int*);
void zgbtrs_(const char*,const int*,const int*,const int*,const int*,const C*,const int*,const int*,C*,const int*,int*);
}
struct Modal { C r,t,dr,dt; };
static bool finite(C v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }
static Modal modal(double lam,double h,double eta) {
    double z=lam*h*h, a=eta*h;
    if (!std::isfinite(z)||!std::isfinite(a)||std::max(std::abs(a),std::sqrt(std::abs(z)))>1e8)
        throw std::invalid_argument("Optical thickness exceeds supported floating-point phase range");
    double co,si,cp,sp,scale=1;
    if(std::abs(z)<1e-3) {
        co=1-z/2+z*z/24-z*z*z/720+z*z*z*z/40320;
        si=1-z/6+z*z/120-z*z*z/5040+z*z*z*z/362880;
        cp=-.5+z/12-z*z/240+z*z*z/10080;
        sp=-1./6+z/60-z*z/1680+z*z*z/90720;
    } else {
        if(z>0) {double x=std::sqrt(z);co=std::cos(x);si=std::sin(x)/x;}
        else {double x=std::sqrt(-z);scale=std::exp(-x);co=(1+std::exp(-2*x))/2;si=-std::expm1(-2*x)/(2*x);}
        cp=-si/2;sp=(co-si)/(2*z);
    }
    C den=co-I*((a*a+z)*si/(2*a)), num=-I*((a*a-z)*si/(2*a));
    C dd=cp-I*((si+(a*a+z)*sp)/(2*a)), dn=-I*((-si+(a*a-z)*sp)/(2*a));
    return {num/den,scale/den,h*h*(dn*den-num*dd)/(den*den),-h*h*scale*dd/(den*den)};
}
static M product(const M&a,const M&b,int n,char transb='N') {
    M c(n*n);C one(1),zero(0);char normal='N';
    zgemm_(&normal,&transb,&n,&n,&n,&one,a.data(),&n,b.data(),&n,&zero,c.data(),&n);
    return c;
}
static M matrix_function(const M&v,const std::vector<Modal>&f,int n,bool reflection) {
    M a=v;
    for(int j=0;j<n;j++)for(int i=0;i<n;i++)a[i+j*n]*=reflection?f[j].r:f[j].t;
    return product(a,v,n,'C');
}
static M transform_vector(const M&v,const C*x,int n) {
    M r(n);
    for(int j=0;j<n;j++)for(int i=0;i<n;i++)r[j]+=std::conj(v[i+j*n])*x[i];
    return r;
}
static M matvec(const M&a,const C*x,int n) {
    M r(n);for(int j=0;j<n;j++)for(int i=0;i<n;i++)r[i]+=a[i+j*n]*x[j];return r;
}
struct Layer {M v,r,t;std::vector<double>values;std::vector<Modal>f;double h;};
static int solve_core(int n,int count,double wl,double period,double nt,double nb,
                     const C*epsilon,const double*heights,int target,int gradient,
                     double*R,double*T,C*ar,C*at,C*state,C*geps,double*residual) {
    if(n<3||n>257||n%2==0||count<1||count>64||target < -n/2||target>n/2)
        throw std::invalid_argument("Unsupported order, layer count or target");
    for(double x:{wl,period,nt,nb})if(!std::isfinite(x)||x<=0)throw std::invalid_argument("Finite positive wavelength, period and exterior indices required");
    int m=n/2,blocks=2*(count+1),total=blocks*n,bw=3*n-1,ld=3*bw+1;
    if(static_cast<long long>(ld)*total>8000000)throw std::invalid_argument("Band allocation limit exceeded");
    double k0=2*pi/wl;
    std::vector<double>kx(n);M kt(n),kb(n),rho0(n),rhoN(n);
    for(int i=0;i<n;i++) {
        kx[i]=2*pi*(i-m)/period;
        kt[i]=std::sqrt(C((k0*nt)*(k0*nt)-kx[i]*kx[i]));
        kb[i]=std::sqrt(C((k0*nb)*(k0*nb)-kx[i]*kx[i]));
        if(!finite(kt[i])||!finite(kb[i])||std::min(std::abs(kt[i]),std::abs(kb[i]))<1e-10*k0)
            throw std::invalid_argument("Nonfinite wave number or exterior diffraction cutoff");
        rho0[i]=(k0-kt[i])/(k0+kt[i]);rhoN[i]=(k0-kb[i])/(k0+kb[i]);
    }
    std::vector<Layer>layers;layers.reserve(count);
    for(int l=0;l<count;l++) {
        double h=heights[l];if(!std::isfinite(h)||h<=0)throw std::invalid_argument("Finite positive layer thickness required");
        M eps(n*n);
        for(int j=0;j<n;j++)for(int i=0;i<n;i++) {
            C a=epsilon[l*n*n+i*n+j],b=std::conj(epsilon[l*n*n+j*n+i]);
            if(!finite(a)||std::abs(a-b)>1e-12*(1+std::abs(b)))throw std::invalid_argument("Finite Hermitian permittivity required");
            eps[i+j*n]=(a+b)*.5;
        }
        M q=eps;int info;char upper='U',vectors='V';
        zpotrf_(&upper,&n,q.data(),&n,&info);
        if(info!=0)throw std::invalid_argument("Positive definite permittivity required");
        q=eps;for(C& x:q)x*=k0*k0;
        for(int i=0;i<n;i++)q[i+i*n]-=kx[i]*kx[i];
        for(C x:q)if(!finite(x))throw std::invalid_argument("Nonfinite Maxwell operator");
        std::vector<double>values(n),rwork(std::max(1,3*n-2));C query;int querysize=-1;
        zheev_(&vectors,&upper,&n,q.data(),&n,values.data(),&query,&querysize,rwork.data(),&info);
        if(info!=0)throw std::runtime_error("LAPACK eigensolver workspace query failed");
        int lwork=std::max(2*n-1,static_cast<int>(query.real()));M work(lwork);
        zheev_(&vectors,&upper,&n,q.data(),&n,values.data(),work.data(),&lwork,rwork.data(),&info);
        if(info!=0)throw std::runtime_error("LAPACK Hermitian eigensolver failed");
        Layer layer;layer.v=std::move(q);layer.values=std::move(values);layer.h=h;
        for(double v:layer.values)layer.f.push_back(modal(v,h,k0));
        layer.r=matrix_function(layer.v,layer.f,n,true);layer.t=matrix_function(layer.v,layer.f,n,false);
        layers.push_back(std::move(layer));
    }
    M band(static_cast<size_t>(ld)*total);
    auto put=[&](int row,int col,C x){band[2*bw+row-col+static_cast<size_t>(col)*ld]=x;};
    auto block=[&](const M&a,int r,int c,C sign){for(int j=0;j<n;j++)for(int i=0;i<n;i++)put(r*n+i,c*n+j,sign*a[i+j*n]);};
    for(int i=0;i<n;i++){put(i,i,1);put(i,n+i,-rho0[i]);}
    for(int j=0;j<count;j++) {
        auto&l=layers[j];
        for(int i=0;i<n;i++){put((2*j+1)*n+i,(2*j+1)*n+i,1);put((2*j+2)*n+i,(2*j+2)*n+i,1);}
        block(l.r,2*j+1,2*j,-1);block(l.t,2*j+1,2*j+3,-1);
        block(l.t,2*j+2,2*j,-1);block(l.r,2*j+2,2*j+3,-1);
    }
    for(int i=0;i<n;i++){put((blocks-1)*n+i,(blocks-1)*n+i,1);put((blocks-1)*n+i,(blocks-2)*n+i,-rhoN[i]);}
    std::vector<int>piv(total);int info,one=1;char normal='N';
    zgbtrf_(&total,&total,&bw,&bw,band.data(),&ld,piv.data(),&info);
    if(info!=0)throw std::runtime_error("LAPACK Maxwell band factorization failed");
    M rhs(total);rhs[m]=2.*kt[m]/(k0+kt[m]);C excitation=rhs[m];
    zgbtrs_(&normal,&total,&bw,&bw,&one,band.data(),&ld,piv.data(),rhs.data(),&total,&info);
    if(info!=0)throw std::runtime_error("LAPACK Maxwell band solve failed");
    for(C x:rhs)if(!finite(x))throw std::runtime_error("Nonfinite Maxwell solution");
    std::copy(rhs.begin(),rhs.end(),state);
    double residual2=0;
    for(int i=0;i<n;i++) {
        C a=rhs[i]-rho0[i]*rhs[n+i]-(i==m?excitation:C(0));
        C b=rhs[(blocks-1)*n+i]-rhoN[i]*rhs[(blocks-2)*n+i];
        residual2+=std::norm(a)+std::norm(b);
    }
    for(int j=0;j<count;j++) {
        const auto&l=layers[j];const C*u=&rhs[2*j*n],*w=&rhs[(2*j+3)*n];
        M ru=matvec(l.r,u,n),tw=matvec(l.t,w,n),tu=matvec(l.t,u,n),rw=matvec(l.r,w,n);
        for(int i=0;i<n;i++)residual2+=std::norm(rhs[(2*j+1)*n+i]-ru[i]-tw[i])+std::norm(rhs[(2*j+2)*n+i]-tu[i]-rw[i]);
    }
    *residual=std::sqrt(residual2/std::norm(excitation));
    for(int i=0;i<n;i++) {
        ar[i]=rhs[i]+rhs[n+i]-(i==m?C(1):C(0));at[i]=rhs[(blocks-2)*n+i]+rhs[(blocks-1)*n+i];
        R[i]=kt[i].real()/kt[m].real()*std::norm(ar[i]);T[i]=kb[i].real()/kt[m].real()*std::norm(at[i]);
    }
    if(!gradient)return 0;
    M adj(total);int q=m+target;C seed=kb[q].real()/kt[m].real()*at[q];
    adj[(blocks-2)*n+q]=adj[(blocks-1)*n+q]=seed;char conjugate='C';
    zgbtrs_(&conjugate,&total,&bw,&bw,&one,band.data(),&ld,piv.data(),adj.data(),&total,&info);
    if(info!=0)throw std::runtime_error("LAPACK adjoint band solve failed");
    for(int j=0;j<count;j++) {
        const auto&l=layers[j];M a=transform_vector(l.v,&adj[(2*j+1)*n],n),b=transform_vector(l.v,&adj[(2*j+2)*n],n);
        M u=transform_vector(l.v,&rhs[2*j*n],n),w=transform_vector(l.v,&rhs[(2*j+3)*n],n),g(n*n);
        for(int c=0;c<n;c++)for(int r=0;r<n;r++) {
            double delta=l.values[r]-l.values[c];C lr,lt;
            if(std::abs(delta)*l.h*l.h<1e-5){Modal f=modal((l.values[r]+l.values[c])*.5,l.h,k0);lr=f.dr;lt=f.dt;}
            else {lr=(l.f[r].r-l.f[c].r)/delta;lt=(l.f[r].t-l.f[c].t)/delta;}
            C mr=std::conj(a[r])*u[c]+std::conj(b[r])*w[c],mt=std::conj(a[r])*w[c]+std::conj(b[r])*u[c];
            g[r+c*n]=std::conj(mr*lr+mt*lt);
        }
        M full=product(product(l.v,g,n),l.v,n,'C');
        for(int r=0;r<n;r++)for(int c=0;c<n;c++)geps[j*n*n+r*n+c]=k0*k0*.5*(full[r+c*n]+std::conj(full[c+r*n]));
    }
    return 0;
}
extern "C" int ps_solve(int n,int count,double wl,double period,double nt,double nb,
                        const C*eps,const double*h,int target,int gradient,
                        double*R,double*T,C*ar,C*at,C*state,C*g,double*residual,char*error,int capacity) noexcept {
    try{return solve_core(n,count,wl,period,nt,nb,eps,h,target,gradient,R,T,ar,at,state,g,residual);}
    catch(const std::exception&e){if(capacity>0)std::snprintf(error,capacity,"%s",e.what());return 1;}
    catch(...){if(capacity>0)std::snprintf(error,capacity,"Unknown native solver error");return 2;}
}
extern "C" int ps_bar(int m,const double*p,double wl,double period,double height,double index,int target,
                      double*value,double*gradient,char*error,int capacity) noexcept {
    try {
        if(m<1||m>128||!std::isfinite(index)||index<1)throw std::invalid_argument("Invalid harmonic count or material index");
        for(int i=0;i<6;i++)if(!std::isfinite(p[i])||(i<3?(p[i]<.15-1e-12||p[i]>.85+1e-12):(p[i]<0||p[i]>1)))throw std::invalid_argument("Invalid six-bar parameters");
        int n=2*m+1;M eps(3*n*n),geps(3*n*n),ar(n),at(n),state(8*n);
        std::vector<double>R(n),T(n),h(3,height);double residual;
        for(int j=0;j<3;j++)for(int r=0;r<n;r++)for(int c=0;c<n;c++) {
            int d=r-c;double coeff=d==0?p[j]:std::sin(pi*d*p[j])/(pi*d);
            eps[j*n*n+r*n+c]=(r==c?1.:0.)+(index*index-1)*coeff*std::exp(-2.*I*pi*double(d)*p[j+3]);
        }
        solve_core(n,3,wl,period,1,1,eps.data(),h.data(),target,1,R.data(),T.data(),ar.data(),at.data(),state.data(),geps.data(),&residual);
        *value=T[m+target];std::fill(gradient,gradient+6,0.);
        for(int j=0;j<3;j++)for(int r=0;r<n;r++)for(int c=0;c<n;c++) {
            int d=r-c;C phase=std::exp(-2.*I*pi*double(d)*p[j+3]);
            C dw=(index*index-1)*std::cos(pi*d*p[j])*phase;
            double coeff=d==0?p[j]:std::sin(pi*d*p[j])/(pi*d);
            C ds=(index*index-1)*coeff*(-2.*I*pi*double(d))*phase;
            C g=std::conj(geps[j*n*n+r*n+c]);gradient[j]+=2.*(g*dw).real();gradient[j+3]+=2.*(g*ds).real();
        }
        return 0;
    } catch(const std::exception&e){if(capacity>0)std::snprintf(error,capacity,"%s",e.what());return 1;}
    catch(...){if(capacity>0)std::snprintf(error,capacity,"Unknown native bar solver error");return 2;}
}
