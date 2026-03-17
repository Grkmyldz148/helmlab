"""GPU-accelerated production test for GenSpace.
Aligned with CPU production_test.py thresholds and methodology.
Usage: python production_test_gpu.py --json gen_v14.json
"""
import json, time, argparse, math, numpy as np
import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D65 = np.array([0.95047, 1.0, 1.08883])
D65_T = torch.tensor(D65, device=device)
M_S = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
M_Si = np.linalg.inv(M_S)
M_S_T = torch.tensor(M_S, device=device)
M_Si_T = torch.tensor(M_Si, device=device)

OK_M1s = np.array([[0.4122214708,0.5363325363,0.0514459929],[0.2119034982,0.6806995451,0.1073969566],[0.0883024619,0.2817188376,0.6299787005]])
OK_M1 = OK_M1s @ M_Si
OK_M2 = np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]])

def scbrt(x): return np.sign(x)*np.abs(x)**(1/3)
def s2l(c): return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s(c): return np.where(c<=0.0031308,c*12.92,1.055*np.maximum(c,1e-10)**(1/2.4)-0.055)
def fwd(M1,M2,xyz): return M2@scbrt(M1@xyz)
def inv_f(M1i,M2i,lab): lc=M2i@lab; return M1i@(np.sign(lc)*np.abs(lc)**3)
def srgb2xyz(rgb): return M_S@s2l(np.array(rgb,dtype=float))
def xyz2cl(xyz):
    r=xyz/D65; f=np.where(r>0.008856,r**(1/3),7.787*r+16/116)
    return np.array([116*f[1]-16,500*(f[0]-f[1]),200*(f[1]-f[2])])
def de00(l1,l2):
    dL=l2[0]-l1[0];C1=np.sqrt(l1[1]**2+l1[2]**2);C2=np.sqrt(l2[1]**2+l2[2]**2)
    dC=C2-C1;dH=np.sqrt(max(0,(l2[1]-l1[1])**2+(l2[2]-l1[2])**2-dC**2))
    SL=1+0.015*(l1[0]-50)**2/np.sqrt(20+(l1[0]-50)**2);SC=1+0.045*C1;SH=1+0.015*C1
    return np.sqrt((dL/SL)**2+(dC/SC)**2+(dH/SH)**2)

# ── GPU CUSP SCAN ──
def gpu_cusp_scan(M1_np, M2_np, n_hues=360, n_L=150, n_C=120):
    M1=torch.tensor(M1_np,device=device);M2=torch.tensor(M2_np,device=device)
    M1i=torch.linalg.inv(M1);M2i=torch.linalg.inv(M2)
    hues=torch.linspace(0,2*3.14159265*(1-1/n_hues),n_hues,device=device)
    Ls=torch.linspace(0.02,0.998,n_L,device=device)
    Cs=torch.linspace(0.001,0.5,n_C,device=device)
    cusp_L=torch.zeros(n_hues,device=device);cusp_C=torch.zeros(n_hues,device=device)
    max_c_all=torch.zeros(n_hues,n_L,device=device)
    B=18
    for hs in range(0,n_hues,B):
        he=min(hs+B,n_hues);nh=he-hs
        Le=Ls.view(1,n_L,1).expand(nh,n_L,n_C)
        Ce=Cs.view(1,1,n_C).expand(nh,n_L,n_C)
        ch=torch.cos(hues[hs:he]).view(nh,1,1);sh=torch.sin(hues[hs:he]).view(nh,1,1)
        lab=torch.stack([Le,Ce*ch,Ce*sh],dim=-1).reshape(-1,3)
        lc=lab@M2i.T;lm=torch.sign(lc)*torch.abs(lc).pow(3.)
        lin=(lm@M1i.T)@M_Si_T.T
        ok=((lin>=-0.002).all(dim=1)&(lin<=1.002).all(dim=1)).reshape(nh,n_L,n_C)
        cv=Cs.view(1,1,n_C).expand(nh,n_L,n_C)
        mc,_=torch.where(ok,cv,torch.zeros_like(cv)).max(dim=2)
        max_c_all[hs:he]=mc
        ci=mc.argmax(dim=1)
        for i in range(nh):
            cusp_L[hs+i]=Ls[ci[i]];cusp_C[hs+i]=mc[i,ci[i]]
    return cusp_L.cpu().numpy(),cusp_C.cpu().numpy(),max_c_all.cpu().numpy(),Ls.cpu().numpy()

# ── TESTS ──
class R:
    def __init__(s,n,v,p,d="",t=""):
        s.name=n;s.value=v;s.passed=p;s.detail=d;s.tier=t
    def __str__(s):
        st="PASS" if s.passed else "FAIL"
        return f"  [{st}] [{s.tier}] {s.name}: {s.value} {s.detail}"

def run_all(M1,M2,label,ok_cusp_L,ok_cusp_C):
    M1i,M2i=np.linalg.inv(M1),np.linalg.inv(M2)
    tests=[]

    # ── 1. GAMUT GEOMETRY (GPU) ──
    t0=time.time()
    cL,cC,mc_all,Ls=gpu_cusp_scan(M1,M2)
    ok_cL,ok_cC,_,_=gpu_cusp_scan(OK_M1,OK_M2) if ok_cusp_L is None else (ok_cusp_L,ok_cusp_C,None,None)
    geo_time=time.time()-t0

    # 1a. Cusp position vs OKLab (direct space-hue comparison, CPU-aligned threshold)
    diffs=np.abs(cL-ok_cL);mx_diff=np.max(diffs);wi=np.argmax(diffs)
    tests.append(R("Cusp Position vs OKLab",f"max_diff={mx_diff:.3f}",mx_diff<0.05,
                   f"worst: h={wi}deg diff={mx_diff:.3f}",t="T4"))

    # 1b. Yellow cusp L (h=80-95)
    yd=[]
    for h in range(80,96):
        yd.append(abs(cL[h%360]-ok_cL[h%360]))
    ym,ymx=np.mean(yd),np.max(yd)
    tests.append(R("Yellow Cusp L Accuracy",f"mean={ym:.3f} max={ymx:.3f}",
                   ym<0.03 and ymx<0.05,t="T4"))

    # 1c. Boundary monotonicity (tolerance=0.001, CPU-aligned)
    M1t=torch.tensor(M1,device=device);M2t=torch.tensor(M2,device=device)
    M1it=torch.linalg.inv(M1t);M2it=torch.linalg.inv(M2t)
    Lf=torch.linspace(0.3,0.998,100,device=device)
    Cf=torch.linspace(0.001,0.4,80,device=device)
    non_uni=0
    for hd in range(0,360,1):
        hr=hd*3.14159265/180
        ch_t,sh_t=np.cos(hr),np.sin(hr)
        Le=Lf.view(100,1).expand(100,80)
        Ce=Cf.view(1,80).expand(100,80)
        lab=torch.stack([Le,Ce*ch_t,Ce*sh_t],dim=-1).reshape(-1,3)
        lc=lab@M2it.T;lm=torch.sign(lc)*torch.abs(lc).pow(3.)
        lin=(lm@M1it.T)@M_Si_T.T
        ok=((lin>=-0.002).all(dim=1)&(lin<=1.002).all(dim=1)).reshape(100,80)
        cv2=Cf.view(1,80).expand(100,80)
        mc2,_=torch.where(ok,cv2,torch.zeros_like(cv2)).max(dim=1)
        mc_np=mc2.cpu().numpy()
        ci2=np.argmax(mc_np)
        for i in range(ci2+1,len(mc_np)-1):
            if mc_np[i]>mc_np[i-1]+0.001:  # CPU uses 0.001
                non_uni+=1;break
    tests.append(R("Gamut Boundary Monotonicity",f"non_unimodal_hues={non_uni}",non_uni==0,t="T1"))

    # 1d. Cusp smoothness (absolute threshold, CPU-aligned)
    jumps=[abs(cL[i]-cL[(i+1)%360]) for i in range(360)]
    mj=max(jumps)
    ok_jumps=[abs(ok_cL[i]-ok_cL[(i+1)%360]) for i in range(360)]
    ok_mj=max(ok_jumps)
    tests.append(R("Cusp Shape Smoothness",f"max_jump={mj:.5f} (OKLab={ok_mj:.5f})",
                   mj<0.02,t="T3"))

    # 1e. Cusp cliff-edge (max chroma drop ratio from cusp)
    max_drop=0
    ok_max_drop=0
    for hd in range(0,360,5):
        hr=hd*3.14159265/180;ch_v,sh_v=np.cos(hr),np.sin(hr)
        for m1n,m2n,target in [(M1,M2,'gen'),(OK_M1,OK_M2,'ok')]:
            m1i_n,m2i_n=np.linalg.inv(m1n),np.linalg.inv(m2n)
            cusp_c=0;cusp_l=0
            Ls_scan=np.linspace(0.3,0.998,100)
            mc_arr=[]
            for L_val in Ls_scan:
                lo,hi=0.0,0.5
                for _ in range(40):
                    mid=(lo+hi)/2
                    lab_v=np.array([L_val,mid*ch_v,mid*sh_v])
                    xyz=inv_f(m2i_n,m2i_n,lab_v)  # use correct inv
                    lc_v=m2i_n@lab_v;xyz=m1i_n@(np.sign(lc_v)*np.abs(lc_v)**3)
                    rgb=M_Si@xyz
                    if np.all(rgb>=-0.001) and np.all(rgb<=1.001): lo=mid
                    else: hi=mid
                mc_arr.append(lo)
                if lo>cusp_c: cusp_c=lo;cusp_l=L_val
            mc_arr=np.array(mc_arr)
            ci=np.argmax(mc_arr)
            if ci<len(mc_arr)-2 and cusp_c>0.01:
                drop=(cusp_c-mc_arr[min(ci+2,len(mc_arr)-1)])/cusp_c
                if target=='gen': max_drop=max(max_drop,drop)
                else: ok_max_drop=max(ok_max_drop,drop)
    tests.append(R("Cusp Cliff-Edge",f"max_drop={max_drop*100:.1f}% (OKLab={ok_max_drop*100:.1f}%)",
                   max_drop<ok_max_drop*2.0,t="T2"))

    print(f"  Gamut Geometry: {geo_time:.1f}s")

    # ── 2. GRADIENT QUALITY ──
    t0=time.time()
    # 2a. Blue->White midpoint
    bx=srgb2xyz([0,0,1]);wx=srgb2xyz([1,1,1])
    bl=fwd(M1,M2,bx);wl=fwd(M1,M2,wx);ml=(bl+wl)/2
    mx2=inv_f(M1i,M2i,ml);ms=l2s(np.clip(M_Si@mx2,0,1))
    gr=ms[1]/max(ms[0],1e-10)
    tests.append(R("Blue->White Midpoint G/R",f"G/R={gr:.3f}",gr>=1.20,t="T2"))

    # 2b. Red->White midpoint
    rx=srgb2xyz([1,0,0])
    rl=fwd(M1,M2,rx);ml2=(rl+wl)/2
    mx3=inv_f(M1i,M2i,ml2);ms2=l2s(np.clip(M_Si@mx3,0,1))
    gb=ms2[1]-ms2[2]
    tests.append(R("Red->White Midpoint G-B",f"G-B={gb:+.4f}",gb<=0.08,t="T3"))

    # 2c. Gradient CV (compute OKLab reference, not hardcoded)
    pairs=[];prims=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(prims)):
        for j in range(i+1,len(prims)):pairs.append((prims[i],prims[j]))
    rng=np.random.RandomState(42)
    for _ in range(60):pairs.append((rng.rand(3).tolist(),rng.rand(3).tolist()))

    def compute_cv_for(m1,m2,m1i,m2i):
        cvs=[]
        for c1,c2 in pairs:
            x1,x2=srgb2xyz(c1),srgb2xyz(c2)
            l1,l2=fwd(m1,m2,x1),fwd(m1,m2,x2)
            ds=[];prev=None
            for t in np.linspace(0,1,26):
                lab=l1+t*(l2-l1);xyz=inv_f(m1i,m2i,lab)
                s8=np.round(l2s(np.clip(M_Si@xyz,0,1))*255)/255
                cl=xyz2cl(np.maximum(M_S@s2l(s8),1e-10))
                if prev is not None:ds.append(de00(prev,cl))
                prev=cl
            if ds:
                a=np.array(ds);m=np.mean(a)
                if m>0.001:cvs.append(np.std(a)/m)
        return np.array(cvs)

    gen_cvs=compute_cv_for(M1,M2,M1i,M2i)
    ok_M1i,ok_M2i=np.linalg.inv(OK_M1),np.linalg.inv(OK_M2)
    ok_cvs=compute_cv_for(OK_M1,OK_M2,ok_M1i,ok_M2i)
    mcv=gen_cvs.mean() if len(gen_cvs)>0 else 999
    ok_mcv=ok_cvs.mean() if len(ok_cvs)>0 else 999
    p95=np.percentile(gen_cvs,95) if len(gen_cvs)>0 else 999
    ok_p95=np.percentile(ok_cvs,95) if len(ok_cvs)>0 else 999
    tests.append(R("Gradient CV",f"mean={mcv*100:.2f}% p95={p95*100:.2f}% (OKLab: {ok_mcv*100:.2f}%/{ok_p95*100:.2f}%)",
                   mcv<ok_mcv*1.2 and p95<ok_p95*1.2,t="T2"))

    # 2d. Hue drift (chroma threshold=3.0, mean AND max, CPU-aligned)
    gen_drifts=[]
    for c1,c2 in pairs:
        x1,x2=srgb2xyz(c1),srgb2xyz(c2)
        l1,l2=fwd(M1,M2,x1),fwd(M1,M2,x2)
        max_drift_pair=0.0
        prev_h=None
        for t in np.linspace(0,1,26):
            lab=l1+t*(l2-l1);xyz=inv_f(M1i,M2i,lab)
            s8=np.round(l2s(np.clip(M_Si@xyz,0,1))*255)/255
            cl=xyz2cl(np.maximum(M_S@s2l(s8),1e-10))
            C_val=np.sqrt(cl[1]**2+cl[2]**2)
            if C_val<3.0:  # CPU uses 3.0 (CIELab C*)
                prev_h=None;continue
            h=math.atan2(cl[2],cl[1])
            if prev_h is not None:
                dh=abs(math.atan2(math.sin(h-prev_h),math.cos(h-prev_h)))
                dh_deg=dh*180.0/math.pi
                max_drift_pair=max(max_drift_pair,dh_deg)
            prev_h=h
        gen_drifts.append(max_drift_pair)
    gen_drifts=np.array(gen_drifts)
    mean_drift=gen_drifts.mean()
    max_drift=gen_drifts.max()
    tests.append(R("Hue Drift",f"mean={mean_drift:.1f}deg max={max_drift:.1f}deg",
                   mean_drift<15.0 and max_drift<45.0,t="T3"))

    # 2e. Lightness monotonicity (check L increases along black→white gradient)
    bk_xyz=srgb2xyz([0,0,0]);wh_xyz=srgb2xyz([1,1,1])
    bk_lab=fwd(M1,M2,bk_xyz);wh_lab=fwd(M1,M2,wh_xyz)
    l_reversals=0
    prev_L=None
    for t in np.linspace(0,1,51):
        lab=bk_lab+t*(wh_lab-bk_lab)
        if prev_L is not None and lab[0]<prev_L-1e-10:
            l_reversals+=1
        prev_L=lab[0]
    tests.append(R("Lightness Monotonicity",f"{l_reversals} reversals",l_reversals==0,t="T1"))
    print(f"  Gradient: {time.time()-t0:.1f}s")

    # ── 3. ACHROMATIC ──
    t0=time.time()
    max_ab=0
    for i in range(257):
        g=i/256;xyz=srgb2xyz([g,g,g]);lab=fwd(M1,M2,xyz)
        c=np.sqrt(lab[1]**2+lab[2]**2)
        if c>max_ab:max_ab=c
    tests.append(R("Gray Ramp Chrominance",f"max={max_ab:.2e}",max_ab<1e-6,t="T1"))

    wlab=fwd(M1,M2,D65)
    w_err=max(abs(wlab[0]-1),abs(wlab[1]),abs(wlab[2]))
    tests.append(R("D65 White Mapping",f"|L-1|={abs(wlab[0]-1):.2e} |a|={abs(wlab[1]):.2e} |b|={abs(wlab[2]):.2e}",
                   w_err<1e-6,t="T1"))
    print(f"  Achromatic: {time.time()-t0:.1f}s")

    # ── 4. ROUND-TRIP ──
    t0=time.time()
    max_err=0
    has_nan=False
    rng2=np.random.RandomState(42)
    for _ in range(1000):
        rgb=rng2.rand(3);xyz=M_S@s2l(rgb);lab=fwd(M1,M2,xyz)
        if np.any(np.isnan(lab)) or np.any(np.isinf(lab)):
            has_nan=True;continue
        xyz2=inv_f(M1i,M2i,lab);rgb2=l2s(np.clip(M_Si@xyz2,0,1))
        if np.any(np.isnan(rgb2)) or np.any(np.isinf(rgb2)):
            has_nan=True;continue
        e=np.max(np.abs(rgb-rgb2))
        if e>max_err:max_err=e
    tests.append(R("Full sRGB Round-Trip",f"max_err={max_err:.2e}",max_err<1e-12,t="T1"))
    tests.append(R("NaN/Inf Check",f"{'FOUND' if has_nan else 'none'}",not has_nan,t="T1"))
    print(f"  Round-trip: {time.time()-t0:.1f}s")

    # ── 5. CONDITION ──
    c1n,c2n=np.linalg.cond(M1),np.linalg.cond(M2)
    ok_c1,ok_c2=np.linalg.cond(OK_M1),np.linalg.cond(OK_M2)
    tests.append(R("Condition M1",f"cond={c1n:.1f} (OKLab={ok_c1:.1f})",c1n<ok_c1*1.5,t="T2"))
    tests.append(R("Condition M2",f"cond={c2n:.1f} (OKLab={ok_c2:.1f})",c2n<ok_c2*1.5,t="T2"))

    # ── 6. HUE ──
    prims_h=[([1,0,0],0),([1,1,0],60),([0,1,0],120),([0,1,1],180),([0,0,1],240),([1,0,1],300)]
    errs=[]
    for rgb,eh in prims_h:
        lab=fwd(M1,M2,srgb2xyz(rgb));h=np.degrees(np.arctan2(lab[2],lab[1]))%360
        dh=h-eh
        if dh>180:dh-=360
        if dh<-180:dh+=360
        errs.append(dh**2)
    rms=np.sqrt(np.mean(errs))
    tests.append(R("Hue Linearity",f"RMS={rms:.1f}deg",rms<35,t="T3"))

    # Primary L range
    pLs=[fwd(M1,M2,srgb2xyz(rgb))[0] for rgb in [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]]
    plr=max(pLs)-min(pLs)
    tests.append(R("Primary L Range",f"{plr:.3f}",plr>0.4,t="T3"))

    # ── 7. YELLOW / WHITE ──
    yl=fwd(M1,M2,srgb2xyz([1,1,0]))
    yC=np.sqrt(yl[1]**2+yl[2]**2)
    tests.append(R("Yellow Chroma",f"C={yC:.4f}",yC>0.10,t="T4"))
    tests.append(R("White Lightness",f"L={wlab[0]:.6f}",abs(wlab[0]-1)<0.001,t="T4"))

    # ── SUMMARY ──
    np_=sum(1 for t in tests if t.passed);nt=len(tests)
    print(f"\n{'='*60}")
    print(f"  {label}: {np_}/{nt} PASSED")
    print(f"{'='*60}")
    for t in tests:print(t)
    fails=[t for t in tests if not t.passed]
    if fails:
        print(f"\n  FAILED:")
        for t in fails:print(f"    - [{t.tier}] {t.name}: {t.value}")
    return tests,np_,nt

def main():
    p=argparse.ArgumentParser();p.add_argument("--json");args=p.parse_args()
    with open(args.json) as f:d=json.load(f)
    M1,M2=np.array(d["M1"]),np.array(d["M2"])
    print(f"Device: {device}")
    if torch.cuda.is_available():print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_all(M1,M2,f"GenSpace ({args.json})",None,None)

if __name__=="__main__":main()
