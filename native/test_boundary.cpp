#include "boundary.cpp"
#include <iostream>
#include <limits>

static constexpr double dg = 1234567.75;
static const C cg(1234567.75, -9876543.25);
static void require(bool yes,const char*message) { if(!yes)throw std::runtime_error(message); }
static void check(int n,int count,int seed,bool zeros=false) {
    int blocks=2*(count+1),m=n/2;
    M eps(count*n*n), heights_dummy;
    std::vector<double> heights(count);
    double wl=zeros?1.:1.37,period=zeros?.5:2.91,nt=zeros?1.3:1.2,nb=zeros?1.1:1.6;
    for(int l=0;l<count;l++) {
        heights[l]=zeros?.21:.1+.017*((seed+l)%11);
        double w=.2+.03*((seed+l)%17),c=.03*((seed+3*l)%29),index=1.3+.1*((seed+7*l)%13);
        for(int r=0;r<n;r++)for(int col=0;col<n;col++) {
            int d=r-col;
            double coeff=d==0?w:std::sin(pi*d*w)/(pi*d);
            eps[l*n*n+r*n+col]=zeros?(r==col?C(4):C(0)):(r==col?C(1):C(0))+(index*index-1)*coeff*std::exp(-2.*I*pi*double(d)*c);
        }
    }
    M saved=eps;
    std::vector<double> r(n+2,dg),t(n+2,dg),residual(3,dg);
    M ar(n+2,cg),at(n+2,cg),state(blocks*n+2,cg),gradient(count*n*n+2,cg);
    std::vector<char> err(514,'X');int target=seed%(2*m+1)-m;
    int status=ps_solve(n,count,wl,period,nt,nb,eps.data(),heights.data(),target,1,r.data()+1,t.data()+1,ar.data()+1,at.data()+1,state.data()+1,gradient.data()+1,residual.data()+1,err.data()+1,512);
    require(status==0,"native numerical case failed");require(eps==saved,"native input mutation");
    for(auto*x:{&r,&t})require(x->front()==dg&&x->back()==dg,"real output buffer overrun");
    for(auto*x:{&ar,&at,&state,&gradient})require(x->front()==cg&&x->back()==cg,"complex output buffer overrun");
    require(residual.front()==dg&&residual.back()==dg,"residual output overrun");
    require(err.front()=='X'&&err.back()=='X',"error buffer overrun");
    double power=0;for(int j=1;j<=n;j++){require(std::isfinite(r[j])&&std::isfinite(t[j]),"nonfinite power");power+=r[j]+t[j];}
    require(std::abs(power-1)<1e-7,"lossless power mismatch");
    require(std::isfinite(residual[1])&&residual[1]<1e-8,"large residual");
    for(int j=1;j<=count*n*n;j++)require(finite(gradient[j]),"nonfinite gradient");
}
int main() {
    try {
        int cases=0;
        for(int n:{3,7,17,33,65})for(int count:{1,3,5})for(int seed:{1,4,7}){check(n,count,seed);++cases;}
        check(7,1,0,true);++cases;
        std::vector<char> error(10,'X');
        int code=ps_solve(0,1,1,1,1,1,nullptr,nullptr,0,0,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,error.data()+1,8);
        require(code!=0&&error.front()=='X'&&error.back()=='X'&&error[8]=='\0',"bounded error message refusal failed");
        code=ps_solve(257,64,1,1,1,1,nullptr,nullptr,0,0,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,error.data()+1,8);
        require(code!=0,"projected allocation cap not refused");
        double p[6]={.45,.45,.45,.15,.4,.65};double value=0;double g[6];char err[512];
        code=ps_bar(5,p,1.55,2.4,.3,2,1,&value,g,err,sizeof(err));require(code==0&&value>0&&value<1,"bar adapter failed");
        p[0]=std::numeric_limits<double>::quiet_NaN();
        require(ps_bar(5,p,1.55,2.4,.3,2,1,&value,g,err,sizeof(err))!=0,"nonfinite geometry not refused");
        std::cout<<"PASS "<<cases<<" guarded native numerical cases, allocation and bounded-error refusals, bar adapter checks\n";
    } catch(const std::exception&e) { std::cerr<<"FAIL "<<e.what()<<"\n";return 1; }
}
