import numpy as np, cv2, subprocess, os, shutil
W,H=480,800; FPS=15; N=54; FADE=7
NAVY=np.array([14,18,44],np.float32)   # BGR-agnostic dark blue (applied in RGB order below)
def load(n):
    im=cv2.imread(f"src/{n}.jpg").astype(np.float32)
    m=cv2.imread(f"work/{n}_mask.png",0).astype(np.float32)/255
    m=np.clip((cv2.GaussianBlur(m,(0,0),1.5)-0.5)*5+0.5,0,1)
    hole=cv2.dilate((m>0.05).astype(np.uint8),np.ones((25,25),np.uint8))
    s=4; small=cv2.resize(im,None,fx=1/s,fy=1/s,interpolation=cv2.INTER_AREA).astype(np.uint8)
    hs=cv2.resize(hole,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_NEAREST)
    inp=cv2.inpaint(small,hs,6,cv2.INPAINT_TELEA)
    inp=cv2.GaussianBlur(cv2.resize(inp,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_CUBIC).astype(np.float32),(0,0),6)
    hf=cv2.GaussianBlur(hole.astype(np.float32),(0,0),8)[...,None]
    bg=im*(1-hf)+inp*hf
    return im,m,bg
def aff(z,fx,fy,ox,oy):  # source point (fx,fy) lands at output (ox,oy) with zoom z
    return np.float32([[z,0,ox-z*fx],[0,z,oy-z*fy]])
ease=lambda t: t*t*(3-2*t)
rng=np.random.default_rng(3); P=rng.random((46,5))   # particles shared by both gifs
def particles(frame,gi):
    lay=np.zeros((H,W),np.float32); T=gi*N+frame
    for x,y,sp,sz,ph in P:
        yy=(y*H - T*(0.5+1.6*sp))%H; xx=(x*W+10*np.sin(T*0.07+ph*6.28))%W
        a=0.25+0.55*(0.5+0.5*np.sin(T*0.2+ph*20))
        cv2.circle(lay,(int(xx),int(yy)),int(1+2*sz),float(a),-1,cv2.LINE_AA)
    return cv2.GaussianBlur(lay,(0,0),1.3)
def render(n,gi,cam,head):
    im,m,bg=load(n); h,w=m.shape
    yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
    # sway weight: grows away from the head/torso axis -> long hair moves, face stays
    wgt=np.clip((np.abs(xx-head[0])-170)/380,0,1)*np.clip((yy-head[1])/500,0,1)
    d="work/f"+n; shutil.rmtree(d,ignore_errors=True); os.makedirs(d)
    for f in range(N):
        t=f/(N-1); T=gi*N+f
        ph=2*np.pi*T/36
        dx=wgt*9*np.sin(ph - yy/260); dy=wgt*3*np.cos(ph - yy/260)
        br=1+0.006*np.sin(2*np.pi*T/27)                      # breathing
        my=(yy-h)/br+h
        mx_=(xx-dx).astype(np.float32); my_=(my-dy).astype(np.float32)
        fg=cv2.remap(im,mx_,my_,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        fm=cv2.remap(m,mx_,my_,cv2.INTER_LINEAR,borderValue=0)
        z,fx,fy,ox,oy=cam(t)
        zb=z*(1.06-0.05*t)                                   # background drifts slower -> parallax
        A=cv2.warpAffine(fg,aff(z,fx,fy,ox,oy),(W,H),flags=cv2.INTER_AREA if z<1 else cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        Mk=cv2.warpAffine(fm,aff(z,fx,fy,ox,oy),(W,H),flags=cv2.INTER_AREA,borderValue=0)[...,None]
        Bg=cv2.warpAffine(bg,aff(zb,fx,fy,ox+8*np.sin(t*3.14),oy),(W,H),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REPLICATE)
        out=Bg*(1-Mk)+A*Mk
        out+=particles(f,gi)[...,None]*np.array([255,225,190],np.float32)*0.75   # BGR: cool white-blue
        k=min(1,f/FADE,(N-1-f)/FADE); k=ease(k)
        out=out*k+NAVY[::-1]*0+np.array([44,18,14],np.float32)*(1-k)
        cv2.imwrite(f"{d}/{f:03d}.png",np.clip(out,0,255).astype(np.uint8))
    o=f"work/gif{gi+1}.gif"
    subprocess.run(f'ffmpeg -y -v error -framerate {FPS} -i {d}/%03d.png -vf "split[a][b];[a]palettegen=max_colors=200:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" -loop 0 {o}',shell=True,check=True)
    print(o,os.path.getsize(o)//1024,"KB")
# GIF1: wide seated shot -> slow push in on the face, ends rising upward
def camA(t):
    e=ease(t); z=0.372+0.33*e
    return z,640,1080-430*e,240,400-70*e*0+0
# GIF2: picks up the upward move: tilt from legs up to the face, then settle
def camB(t):
    e=ease(min(1,t/0.72)); z=0.56-0.06*e+0.05*ease(max(0,(t-0.6)/0.4))
    return z,640,1560-860*e,240,430
render("a",0,camA,(640,640)); render("b",1,camB,(640,430))
