#
#   solv_call_bump : use Monte Carlo to approximate vanilla call with bump functions
#
import math
from solv import iter_solve_grad
from call_put_bs import vanilla_call_price


def sign(x):
    return math.copysign(1, x)

def step(x): 
    return sign(x) + 1


def smooth_step(x):                 ## smooth cubic sigmoid on [0,1] : (0,0) .. (1,1)
    return x*x*(3 - 2*x);     

def bump(x):
    if (x<-1 or x>1):
        return 0;
    if (x<0):
        return smooth_step(1+x)
    return smooth_step(1-x)


def test_bump_01():
    for i in range(400):
        s = i/100 - 2.0     # -2 .. -1 .. 0 .. +1 .. +2
        h = bump(s)
        #t = step(s)
        print("bump : {:3.4f} = {:3.4f}".format(s,h));
        #print("step : {:3.4f} = {:3.4f}".format(s,t));

#test_bump_01();


## N-dim bump and planar cutoff

## bump3( H, x,y,z, dx,dy,dz )
## plan3( K, a,b,c )            // zero on one side of plan, 1 on other side : aka cutoff func [ rotation built in ]


def bump3(H, ox,oy,oz, w,h,d):       # make a function : R3->R 
    def fxyz(x,y,z):
        dx=x-ox
        dy=y-oy
        dz=z-oz
        return H*bump(w*dx)*bump(h*dy)*bump(d*dz)
    return fxyz


def plan3(K, s,t,u):
    def fxyz(x,y,z):
        v = s*x + t*y + u*z + K;
        if (v>0):
            return 1
        return 0
    return fxyz


def bump2(H, ox,oy, iw,ih):             # gaussian like 2D bump with limited range : R2->[0,1]
    def fxy(x,y):
        dx=x-ox
        dy=y-oy
        return H*bump(iw*dx)*bump(ih*dy)
    return fxy

def plan2(K, s,t):                      # step function in R2 : 0 one side of line, 1 other side
    def fxy(x,y):
        v = s*x + t*y + K;
        if (v>0):
            return 1
        return 0
    return fxy


# the vanilla call fn we want to approximate 

def van_call(S,K):
    #S=100
    #K=100
    r=0.05
    v=0.20
    T=1.0
    cal = vanilla_call_price(S, K, r, v, T)
    return cal


print("van_call : {:3.4f}".format(van_call(100,105)))
print("van_call : {:3.4f}".format(van_call(100,100)))
print("van_call : {:3.4f}".format(van_call(100, 95)))
    
ftarg = van_call                    # target function to approximate : vanilla call
    

fsamp = [];
DG=5                            # grid spacing for samples
DLO=70
DHI=120
print("xy range : ",DLO,"..",DHI," : grid step ",DG)
for s in range(DLO,DHI,DG):
    for k in range(DLO,DHI,DG):
        v = ftarg(s,k)
        fsamp.append([s,k,v]);

def log_batch_samples():
    for [s,k,v] in fsamp:
        print("s {:3d} {:3d} : {:7.4f}".format(s,k,v))
    print("ftarg samples : ",len(fsamp))

## log_batch_samples();

# compute error over batch samples : rms and max


def err_rms_batch(func):
    es=0;
    n=0;
    for [s,k,v] in fsamp:
        vf = func(s,k)
        dv = v-vf;
        es += dv*dv
        # print("  dv : {:6.3f}".format(dv));
        n+=1
    # print("  es : {:6.3f}".format(es));
    # print("  n  : ",n)
    rms = math.sqrt(es/n);
    return rms

def err_max_batch(func):
    mdv = 0.0 
    mav = 0.0 
    ex = 0
    ey = 0
    for [s,k,v] in fsamp:
        vf = func(s,k)
        dv = v-vf;
        av = abs(dv);
        if av > mav:
            mav = av
            mdv = dv
            ex = s
            ey = k
        
    print(" err max batch : {:d} {:d} : {:6.3f}".format(ex,ey,mdv));
    return [ex,ey,mdv]

def fconst(C):
    def fc(x,y):
        return C;
    return fc

def test_batch_errs():
    fcon10 = fconst(10)
    rms_con = err_rms_batch(fcon10)
    print("rms const : {:6.3f}".format(rms_con))
    [mx,my,mz] = err_max_batch(fcon10)



def fzero2(s,t):
    return 0.0

def fadd2(fa,fb):
    def fs2(s,t):
        return fa(s,t)+fb(s,t)
    return fs2


all_bumps = []                                  ## record all our bump fn slots

ETA     = 0.000000001;
NITER   = 10_000;
NROUNDS = 10;


def fit_single_bump(_fbase):

    _slots = [5, 80,93, 0.04,0.03];             ## slots to adjust : H x y w h

    # target new bump at worst area

    [xe, ye, err_h] = err_max_batch(_fbase);
    _slots[0] = err_h;
    _slots[1] = xe
    _slots[2] = ye
    fbase = _fbase;

    def err_scor(slots):
        [H, ox,oy, iw,ih] = slots; 
        if (iw<ETA) or (ih<ETA):
            return 10_000_000
        fbump = bump2(H, ox,oy, iw,ih)
        func = fadd2(fbase, fbump)
        er = err_rms_batch(func)
        return er
        
    [rscr,rslots] = iter_solve_grad(_slots, err_scor, NITER)

    oscr = err_scor(_slots);
    #print(" oscr : {:8.4f}".format(oscr));
    #print(" rscr : {:8.4f}".format(rscr));
    #for sl in rslots :
    #    print("  slot : {:8.4f}".format(sl))

    all_bumps.append(rslots);

    [H, ox,oy, iw,ih] = rslots
    fbump = bump2(H, ox,oy, iw,ih)      # new adjustment bump
    fapprox = fadd2(fbase, fbump)       # combined 

    print("compare at 100,100 :")
    cal_tru = van_call(100,100)
    cal_est = fapprox(100,100)
    print("  van_call : {:8.4f}".format(cal_tru))
    print("  fapprox  : {:8.4f}".format(cal_est))
    print("     diff  : {:8.4f}".format(abs(cal_tru-cal_est)))

    print("rms and max errs : ")
    [xe, ye, emax] = err_max_batch(fapprox)
    erms = err_rms_batch(fapprox)
    print("  rms err  : {:8.4f}".format(erms))
    print("  max err  : {:8.4f}".format(emax))

    return fapprox


## repeat many rounds of fitting bups to approximate / converge toward ftarg


all_bumps.append([5, 80,93, 0.04,0.03]);
f00 = bump2(5, 80,93, 0.04,0.03)

funk = f00                          # combined sum of bump funcs
for k in range(NROUNDS):
    print("\nround : ",k)
    funk = fit_single_bump(funk);


def log_all_slots():
    for slts in all_bumps:
        sss = "bump2 slots : "
        for sl in slts :
            sss += " {:12.8f}".format(sl)
        print(sss)

log_all_slots()

# remake funk by combinings bump2s from all the stored slots 

def bump2_from_slots(slts):
    [H, ox,oy, iw,ih] = slts
    return bump2(H, ox,oy, iw,ih)

def make_all_bumps_func():
    func = fzero2
    for slts in all_bumps:
        fbump = bump2_from_slots(slts)
        func = fadd2(func, fbump)
    return func


fxx = make_all_bumps_func()
print("\nfxx from all_slots")
[xe, ye, emax] = err_max_batch(fxx)
erms = err_rms_batch(fxx)
print("  rms err  : {:8.4f}".format(erms))
print("  max err  : {:8.4f}".format(emax))
