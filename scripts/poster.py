import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
S=2000; SS=2          # output size, supersample
NAVY=np.array([38,36,96.]); MINT=np.array([196,228,178.]); CREAM=np.array([253,247,222.])
RED=np.array([214,52,74.]); GOLD=np.array([240,170,60.]); YEL=np.array([248,231,70.]); VIO=np.array([150,140,214.])

def halftone(dark, pitch, ang):
    h,w=dark.shape; y,x=np.mgrid[0:h,0:w].astype(np.float32)
    c,s=np.cos(ang),np.sin(ang); u=(x*c+y*s)/pitch; v=(-x*s+y*c)/pitch
    d=np.hypot(u-np.round(u),v-np.round(v))
    r=0.72*np.sqrt(np.clip(dark,0,1))
    return np.clip((r-d)*pitch/1.5+0.5,0,1)

def make(name, crop, place, texts, tx, ty, align, out):
    im=cv2.imread(f"src/{name}.jpg")[:,:,::-1].astype(np.float32)/255
    m=cv2.imread(f"work/{name}_mask.png",0).astype(np.float32)/255
    x0,y0,x1,y1=crop; im=im[y0:y1,x0:x1]; m=m[y0:y1,x0:x1]
    px,py,pw=place; sc=pw*SS/im.shape[1]; W=int(im.shape[1]*sc); H=int(im.shape[0]*sc)
    im=cv2.resize(im,(W,H),interpolation=cv2.INTER_CUBIC); m=cv2.resize(m,(W,H),interpolation=cv2.INTER_CUBIC)
    m=np.clip((cv2.GaussianBlur(m,(0,0),2)-0.5)*6+0.5,0,1)
    R,G,B=im[...,0],im[...,1],im[...,2]; L=0.299*R+0.587*G+0.114*B
    mx=im.max(2); mn=im.min(2); sat=(mx-mn)/(mx+1e-4)
    sm=lambda a:cv2.GaussianBlur(a.astype(np.float32),(0,0),2.5)
    red=sm((R>0.33)&(R>1.7*G)&(R>1.4*B)&(sat>0.45))
    gold=sm((R>0.6)&(G>0.45)&(B<0.6*R)&(sat>0.4)&(G>0.6*R))
    blue=sm((B>1.25*R)&(B>0.28)&(sat>0.35)&(L<0.6))
    light=sm(L>0.60)
    base=np.ones((H,W,3),np.float32)*MINT
    for wgt,col in((blue,VIO),(light,CREAM),(gold,GOLD),(red,RED)):
        wgt=np.clip((wgt-0.5)*5+0.5,0,1)[...,None]; base=base*(1-wgt)+col*wgt
    Lb=cv2.GaussianBlur(L,(0,0),1.5)
    dark=1-np.power(np.clip(Lb,0,1),0.42); dark=np.clip((dark-0.10)/0.82,0,1)
    dark=np.where(red>0.5,dark*0.75,dark)
    ink=halftone(dark,7.0*SS,np.deg2rad(45))
    solid=np.clip((0.085-cv2.GaussianBlur(L,(0,0),1.0))*40+0.5,0,1)   # line art / deep blacks
    ink=np.maximum(ink,solid)[...,None]
    ch=base*(1-ink)+NAVY*ink
    # sticker outline
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9*SS//2*2+1,)*2)
    ol=cv2.dilate(m,k)
    C=S*SS; canvas=np.ones((C,C,3),np.float32)*YEL
    rng=np.random.default_rng(7)
    canvas+=cv2.GaussianBlur(rng.normal(0,9,(C,C,1)).astype(np.float32),(0,0),1.2)[...,None]*np.array([1,1,.6])
    X=int(px*SS); Y=int(py*SS)
    sx0=max(0,-X); sy0=max(0,-Y); ex=min(W,C-X); ey=min(H,C-Y)
    reg=canvas[Y+sy0:Y+ey, X+sx0:X+ex]
    o=ol[sy0:ey,sx0:ex,None]; a=m[sy0:ey,sx0:ex,None]
    reg[:]=reg*(1-o)+NAVY*o; reg[:]=reg*(1-a)+ch[sy0:ey,sx0:ex]*a
    pil=Image.fromarray(np.clip(canvas,0,255).astype(np.uint8)).resize((S,S),Image.LANCZOS)
    d=ImageDraw.Draw(pil)
    f1=ImageFont.truetype("fonts/Yellowtail-Regular.ttf",250); f2=ImageFont.truetype("/usr/share/fonts/truetype/inter/Inter-Bold.ttf" if False else FB,62)
    b1=d.textbbox((0,0),texts[0],font=f1); w1=b1[2]-b1[0]
    b2=d.textbbox((0,0),texts[1],font=f2); w2=b2[2]-b2[0]
    ax=lambda w: tx-w if align=="r" else tx
    d.text((ax(w1)-b1[0],ty-b1[1]),texts[0],font=f1,fill=tuple(RED.astype(int)))
    d.text((ax(w2)-b2[0],ty+(b1[3]-b1[1])+34-b2[1]),texts[1],font=f2,fill=(40,38,60))
    pil.save(out,quality=95); pil.resize((700,700)).save(out.replace(".png","_prev.jpg"))
import subprocess
FB=subprocess.check_output(["fc-match","-f","%{file}","Inter:bold"]).decode()
import sys
which=sys.argv[1:] or ["1","2"]
if "1" in which: make("a",(60,10,1260,1640),(-170,130,1400),("i'm ok","HAPPY EVERYDAY"),1900,860,"r","work/poster1.png")
if "2" in which: make("b",(0,40,1320,1660),(560,120,1560),("so am i","HAPPY EVERY NIGHT"),110,860,"l","work/poster2.png")
