import numpy as np, cv2, subprocess
from PIL import Image, ImageDraw, ImageFont
SW,SH=2000,3000; SS=2
NAVY=np.array([38,36,96.]); MINT=np.array([196,228,178.]); CREAM=np.array([253,247,222.])
RED=np.array([214,52,74.]); GOLD=np.array([240,170,60.]); YEL=np.array([248,231,70.]); VIO=np.array([150,140,214.])
FB=subprocess.check_output(["fc-match","-f","%{file}","Inter:bold"]).decode()
def halftone(dark,pitch,ang):
    h,w=dark.shape; y,x=np.mgrid[0:h,0:w].astype(np.float32)
    c,s=np.cos(ang),np.sin(ang); u=(x*c+y*s)/pitch; v=(-x*s+y*c)/pitch
    d=np.hypot(u-np.round(u),v-np.round(v)); r=0.72*np.sqrt(np.clip(dark,0,1))
    return np.clip((r-d)*pitch/1.5+0.5,0,1)
def make(name,place,clear,blue_y,texts,tx,ty,align,out):
    im=cv2.imread(f"src/{name}.jpg")[:,:,::-1].astype(np.float32)/255
    m=cv2.imread(f"work/{name}_mask.png",0).astype(np.float32)/255
    px,py,pw=place; sc=pw*SS/im.shape[1]; W=int(im.shape[1]*sc); H=int(im.shape[0]*sc)
    im=cv2.resize(im,(W,H),interpolation=cv2.INTER_LANCZOS4); m=cv2.resize(m,(W,H),interpolation=cv2.INTER_CUBIC)
    m=np.clip((cv2.GaussianBlur(m,(0,0),2)-0.5)*6+0.5,0,1)
    R,G,B=im[...,0],im[...,1],im[...,2]; L=0.299*R+0.587*G+0.114*B
    mx=im.max(2); mn=im.min(2); sat=(mx-mn)/(mx+1e-4)
    sm=lambda a:cv2.GaussianBlur(a.astype(np.float32),(0,0),2.5)
    red=sm((R>0.33)&(R>1.7*G)&(R>1.4*B)&(sat>0.45))
    gold=sm((R>0.6)&(G>0.45)&(B<0.6*R)&(sat>0.4)&(G>0.6*R))
    blue=sm((B>1.25*R)&(B>0.36)&(sat>0.35)&(L<0.6)&(L>0.2)); light=sm(L>0.60)
    blue[:int(blue_y*sc)]=0                                 # bluish hair sheen must stay hair-coloured
    base=np.ones((H,W,3),np.float32)*MINT
    for wgt,col in((blue,VIO),(light,CREAM),(gold,GOLD),(red,RED)):
        wgt=np.clip((wgt-0.5)*5+0.5,0,1)[...,None]; base=base*(1-wgt)+col*wgt
    # focus field: wide gaussian falloffs around each focus point, merged smoothly (no hard edge anywhere)
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32); inv=np.ones((H,W),np.float32)
    for cx,cy,rx,ry,amt in clear:
        d2=((xx-cx*sc)/(rx*sc))**2+((yy-cy*sc)/(ry*sc))**2; inv*=1-amt*np.exp(-d2*0.9)
    c=1-inv
    dark=1-np.power(np.clip(cv2.GaussianBlur(L,(0,0),1.5),0,1),0.42); dark=np.clip((dark-0.10)/0.82,0,1)
    dots=halftone(dark,7.0*SS,np.deg2rad(45))
    dots_soft=cv2.GaussianBlur(dots,(0,0),2.2*SS)              # out-of-focus screen far from the focus
    far=np.clip(1-c*2.2,0,1); dots=dots*(1-0.55*far)+dots_soft*0.55*far
    cont=np.clip(1.63*dark,0,1)                                # same ink coverage as the dots, but continuous
    solid=np.clip((0.085-cv2.GaussianBlur(L,(0,0),1.0))*40+0.5,0,1)
    ink=np.maximum(dots*(1-c)+cont*c,solid)[...,None]          # dots lose contrast gradually -> smooth tone
    ht=base*(1-ink)+NAVY*ink
    o=im*255; o=o+0.6*(o-cv2.GaussianBlur(o,(0,0),3)); o=255*np.power(np.clip(o/255,0,1),0.85)
    cb=(0.8*np.power(c,1.6))[...,None]                         # true colour and sharpness only arrive near the centre
    ch=ht*(1-cb)+o*cb
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9*SS//2*2+1,)*2); ol=cv2.dilate(m,k)
    CW,CH=SW*SS,SH*SS; canvas=np.ones((CH,CW,3),np.float32)*YEL
    rng=np.random.default_rng(7)
    canvas+=cv2.GaussianBlur(rng.normal(0,9,(CH,CW,1)).astype(np.float32),(0,0),1.2)[...,None]*np.array([1,1,.6])
    X=int(px*SS); Y=int(py*SS); sx0=max(0,-X); sy0=max(0,-Y); ex=min(W,CW-X); ey=min(H,CH-Y)
    reg=canvas[Y+sy0:Y+ey,X+sx0:X+ex]; oo=ol[sy0:ey,sx0:ex,None]; a=m[sy0:ey,sx0:ex,None]
    reg[:]=reg*(1-oo)+NAVY*oo; reg[:]=reg*(1-a)+ch[sy0:ey,sx0:ex]*a
    pil=Image.fromarray(np.clip(canvas,0,255).astype(np.uint8)).resize((SW,SH),Image.LANCZOS)
    d=ImageDraw.Draw(pil); f1=ImageFont.truetype("fonts/Yellowtail-Regular.ttf",270); f2=ImageFont.truetype(FB,66)
    b1=d.textbbox((0,0),texts[0],font=f1); b2=d.textbbox((0,0),texts[1],font=f2)
    ax=lambda b: tx-(b[2]-b[0]) if align=="r" else tx
    d.text((ax(b1)-b1[0],ty-b1[1]),texts[0],font=f1,fill=tuple(RED.astype(int)))
    d.text((ax(b2)-b2[0],ty+(b1[3]-b1[1])+36-b2[1]),texts[1],font=f2,fill=(40,38,60))
    pil.save(out); pil.resize((600,900)).save(out.replace(".png","_prev.jpg"))
    x0,y0=clear[0][0]*pw/1320+px-450,clear[0][1]*pw/1320+py-450
    pil.crop((int(x0),int(y0),int(x0)+900,int(y0)+900)).save(out.replace(".png","_zoom.jpg"))
# clear = (cx,cy,rx,ry) in source pixels: face, then hands
make("a",(250,545,1500),[(640,720,430,400,1.0),(1090,1590,300,230,0.75)],1150,("i'm ok","HAPPY EVERYDAY"),1890,150,"r","work/full1.png")
make("b",(430,200,1500),[(640,450,400,400,1.0),(985,1640,260,360,0.75)],600,("so am i","HAPPY EVERY NIGHT"),110,150,"l","work/full2.png")
