##
## iter_solve_grad 
##
##      iterating stochastic solver uses random moves and sampled gradient descent
##
##      copyright (c) gordon anderson
##      released under MIT licence
##
import random
import copy
import math


def rand():
    return random.random()

def srand():
    return 2*(0.5-rand());

def randn(N):
    return random.randint(0,N-1);


def iter_solve_grad(_slots, err_scor, NITER=5000):
    
    nsl = len(_slots);
    slots = _slots.copy();

    jigl_args = [ 
        [1.10,0.90], [1.02,0.98], [1.002,0.998], 
        [0.200,-0.200], [0.020,-0.020], [0.002,-0.002], [0.0001,-0.0001] 
    ]

    jigl_n = len(jigl_args);

    def rand_grad(nsl) : 
        arr = slots.copy();
        for i in range(nsl):
          arr[i] = 0.005*srand()
        return arr

    grds = rand_grad(nsl);                      ## nonlocal var

    def slots_plus_grad(ugrds) :
        cslots = slots.copy();
        for i,dv in enumerate(ugrds) :
            cslots[i] += dv
        return cslots;

    def jiggl_grad(nscr) : 
        nonlocal grds
        ngrad = grds.copy()
        for k in range(2*nsl) :
            s = randn(nsl);
            v = ngrad[s];

            optn = randn(jigl_n);
            varn = randn(2);

            dv = jigl_args[optn][varn];
            if (optn<3):
                v *= dv;
            else:
                v += dv;

            ngrad[s]=v;

        nuslots = slots_plus_grad(ngrad)
        escr = err_scor(nuslots);
        if (escr<nscr):
            #use_grads(ngrad)
            grds = ngrad

    nloop = NITER;
    bscr = err_scor(slots);
    bslots = slots;
    nimprov=0;
    while(nloop>0):
        mslots = slots_plus_grad(grds); 
        nscr = err_scor(mslots);
        if (nscr<bscr):
            # print("imprv  : {:.4f}  ->  {:.4f} ".format(bscr,nscr))
            # for sl in slots :
            #    print("  slot : {:6.4f}".format(sl))

            if (rand()>0.90):
                jiggl_grad(nscr);

            bscr = nscr;    
            bslots = mslots;
            nimprov+=1;
        else:                         
            jiggl_grad(nscr);

        nloop-=1;

    print("  solver : improved {:d} of {:d} rounds".format(nimprov,NITER));
    
    return [bscr, bslots];              # return best error score and best slots values



# sample problem : find xn or nearest points u, v on two segs [va,vb] [vc,vd]

def dist2(p,v):
    [x,y] = p;
    [a,b] = v;
    dx = x-a;
    dy = y-b;
    return math.sqrt(dx*dx+dy*dy);
    

def test_find_pt_near(p):
 
    def err_scor(slots):
        x = slots[0];
        y = slots[1];
        dd = dist2([x,y],p);
        return dd;

    _slots = [1,3];

    [rscr,rslots] = iter_solve_grad(_slots, err_scor, NITER=10000)

    print("find_pt_near using iter_solv_grad :");
    print("  rscr   : {:.8f}".format(rscr))
    print("  rslots : {:.6f},{:.6f}".format(rslots[0],rslots[1]))


## test_find_pt_near([4,5])


# sample problem : find closest points on 2 line segments in 3D

def dist3(p,v):
    [x,y,z] = p
    [a,b,c] = v
    dx = x-a
    dy = y-b
    dz = z-c
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def lerp(a,b,s):
    t = 1-s
    return s*a + t*b

def clamp(lo,hi,s):
    return max(min(s,hi),lo)

def lerp3(v,p,s):
    [x,y,z] = v
    [a,b,c] = p
    return [lerp(x,a,s), lerp(y,b,s), lerp(z,c,s)]


def test_find_near_pts_on_segs3(sg1,sg2):
    [va,vb] = sg1
    [pa,pb] = sg2

    ## slots : s and t slide points v and p along sg1 and sg2 respectively
    
    _slots=[rand(),rand()]
    #_slots=[0.33,0.77]

    def err_scor(slots):
        s = slots[0]
        t = slots[1]
        if s<0 or s>1 or t<0 or t>1 :           # harsh penalty for s,t out of bounds :]
            return 10_000_000;
        v = lerp3(va,vb,s)
        p = lerp3(pa,pb,t)
        dvp = dist3(v,p)
        return dvp

    [rscr,rslots] = iter_solve_grad(_slots, err_scor, NITER=1000)

    [s,t] = rslots
    v = lerp3(va,vb,s)
    p = lerp3(pa,pb,t)

    print("find_near_pts_on_segs3 using iter_solv_grad :")
    print("  rscr   : {:.8f}".format(rscr))
    print("  rslots : {:.6f},{:.6f}".format(s,t))
    print("  v_near : {:.4f},{:.4f},{:.4f}".format(v[0],v[1],v[2]))
    print("  p_near : {:.4f},{:.4f},{:.4f}".format(p[0],p[1],p[2]))


def test_pts_seg3():
    test_find_near_pts_on_segs3([[0,0,0],[4,0,0]], [[2,2,-2],[2,2,2]])   # expect [2,0,0] & [2,2,0] d=2.0
    test_find_near_pts_on_segs3([[0,0,0],[4,1,1]], [[2,2,-2],[2,2,2]])   
    test_find_near_pts_on_segs3([[0,3,0],[1,0,0]], [[0,0,0],[3,2,0]])
    test_find_near_pts_on_segs3([[4,0,0],[0,4,0]], [[0,0,0],[4,4,0]])    # cross in plane, expect v=p=[2,2] & d=0

## test_pts_seg3();

