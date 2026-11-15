import numpy as np, cv2, subprocess, sys
from PIL import Image, ImageDraw, ImageFont
SW,SH=2000,3000
fm=lambda q: subprocess.check_output(["fc-match","-f","%{file}",q]).decode()
yy,xx=np.mgrid[0:SH,0:SW].astype(np.float32)
REDC=np.array([236,24,48.])
def load(name,px,py,pw):
    im=cv2.imread(f"src/{name}.jpg")[:,:,::-1].astype(np.float32)/255
    m=cv2.imread(f"work/{name}_mask.png",0).astype(np.float32)/255
    sc=pw/im.shape[1]; M=np.float32([[sc,0,px],[0,sc,py]])
    im=cv2.warpAffine(im,M,(SW,SH),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
    m=cv2.warpAffine(m,M,(SW,SH),flags=cv2.INTER_CUBIC,borderValue=0)
    m=np.clip((cv2.GaussianBlur(m,(0,0),1.5*max(1,sc/1.5))-0.5)*6+0.5,0,1)
    R,G,B=im[...,0],im[...,1],im[...,2]; L=0.299*R+0.587*G+0.114*B
    sat=(im.max(2)-im.min(2))/(im.max(2)+1e-4)
    sm=lambda a:np.clip((cv2.GaussianBlur(a.astype(np.float32),(0,0),2.5)-0.5)*5+0.5,0,1)
    red=sm((R>0.33)&(R>1.7*G)&(R>1.4*B)&(sat>0.45)); gold=sm((R>0.6)&(G>0.45)&(B<0.6*R)&(sat>0.4)&(G>0.6*R))
    blue=sm((B>1.25*R)&(B>0.36)&(sat>0.35)&(L<0.6)&(L>0.2))
    return dict(im=im,m=m,L=L,red=red,gold=gold,blue=blue,sc=sc,P=lambda x,y:(x*sc+px,y*sc+py))
def spaced(d,txt,font,y,fill,sp,cx=SW//2,right=None,left=None):
    ws=[d.textlength(ch,font=font) for ch in txt]; tot=sum(ws)+sp*(len(txt)-1)
    x=cx-tot/2 if right is None and left is None else (right-tot if right is not None else left)
    for ch,w in zip(txt,ws): d.text((x,y),ch,font=font,fill=fill); x+=w+sp
def dark_fig(D,fx,fy,fall_len=1500,floor=0.42,side_w=170,side_off=90,keep=(1,0),gamma=1.25):
    fall=np.clip(1.12-np.clip(yy-fy,0,None)/fall_len,0,1)**1.7
    side=floor+(1-floor)/(1+np.exp((xx-fx-side_off)/side_w)); light=fall*side
    t=np.clip((D["L"]-0.04)*1.15,0,1)**gamma
    lo=np.array([5,7,18.]); hi=np.array([196,208,238.])
    fig=(lo+(hi-lo)*t[...,None])*light[...,None]+lo*(1-light[...,None])*0.6
    for msk,col,k in((D["red"],REDC,keep[0]),(D["gold"],np.array([255,196,40.]),keep[1])):
        if k:
            c=col*(0.45+0.9*np.clip(D["L"]*2.2,0,1))[...,None]*(0.35+0.65*fall**0.5)[...,None]
            fig=fig*(1-msk[...,None])+c*msk[...,None]
    mb=cv2.GaussianBlur(D["m"],(0,0),3); rim=np.clip(mb-np.roll(mb,14,axis=1),0,1)*D["m"]*fall
    return fig+rim[...,None]*np.array([90,130,255.])*0.9, fall
def glow(msk,col,k=0.55): return (cv2.GaussianBlur(msk,(0,0),40)*3+cv2.GaussianBlur(msk,(0,0),10)*1.2)[...,None]*col*k
def finish(out,grain=5.5,vig=0.5):
    out=out*(1-vig*np.clip(np.hypot((xx-SW/2)/SW,(yy-SH/2)/SH)*1.5-0.45,0,1))[...,None]
    return out+np.random.default_rng(5).normal(0,grain,(SH,SW,1))
def save(out,name,draw):
    pil=Image.fromarray(np.clip(out,0,255).astype(np.uint8)); draw(ImageDraw.Draw(pil),pil)
    pil.save(f"work/{name}.png"); pil.resize((600,900)).save(f"work/{name}_prev.jpg")

def pressure():                      # v2: closer, half the face swallowed by shadow
    pw=3100; s=pw/1320; fxs,fys=640,440; px=1000-fxs*s; py=1180-fys*s
    D=load("b",px,py,pw); fx,fy=D["P"](fxs,fys)
    fig,fall=dark_fig(D,fx,fy,fall_len=1750,floor=0.05,side_w=95,side_off=-10,gamma=1.45)
    bg=np.ones((SH,SW,3),np.float32)*np.array([3,4,9.])
    bg+=np.exp(-(((xx-fx+700)/800)**2+((yy-fy+200)/1100)**2))[...,None]*np.array([14,26,70.])
    m=D["m"]; out=bg*(1-m[...,None])+fig*m[...,None]
    eye=D["red"]*(yy<fy+420)*(yy>fy-400); out+=glow(eye,np.array([255,30,50.]),0.8)
    # long red streak through the eye
    ey,ex=np.unravel_index(np.argmax(cv2.GaussianBlur(eye,(0,0),25)),eye.shape)
    streak=np.exp(-((yy-ey)/5.0)**2)*np.exp(-((xx-ex)/620)**2); out+=streak[...,None]*np.array([255,40,60.])*0.75
    out*=np.clip(1-np.clip(yy-2150,0,None)/650,0,1)[...,None]**1.5
    out=finish(out,6.5,0.7)
    def draw(d,pil):
        f1=ImageFont.truetype(fm("Inter:black"),150); f2=ImageFont.truetype(fm("Inter:medium"),40)
        spaced(d,"DON'T LOOK AWAY",f1,2560,(232,236,246),6)
        spaced(d,"SHE HAS ALREADY SEEN YOU",f2,2780,(236,24,48),16)
    save(out,"t1_pressure",draw)

def standoff():                      # ink close-up, eye to fingertip
    B=90; BOT=330; iw=SW-2*B; ih=SH-B-BOT
    crop_w=600; s=iw/crop_w; pw=1320*s; px=B-255*s; py=B-330*s
    D=load("a",px,py,pw); L=cv2.GaussianBlur(D["L"],(0,0),2.2); Lf=cv2.GaussianBlur(D["L"],(0,0),1.6)
    line=np.clip((cv2.GaussianBlur(D["L"],(0,0),9)-Lf-0.07)*22,0,1)      # thin drawn outlines survive
    paper=np.array([244,240,230.]); ink=np.array([16,15,22.])
    blk=np.clip((0.15-L)*30+0.5,0,1)
    u=(xx+yy)/13.0; ln=np.abs(u-np.round(u))*2                         # diagonal hatch for mid-tones
    mid=np.clip((0.50-L)/0.35,0,1); hatch=np.clip((mid*0.9-ln)*8+0.5,0,1)*(L<0.62)
    cov=np.maximum(np.maximum(blk,hatch),line)*D["m"]
    # background behind the figure: flat hatched grey so the silhouette reads
    bgk=np.clip((0.38-ln)*8+0.5,0,1)*(1-D["m"])
    cov=np.maximum(cov,bgk)[...,None]
    out=paper*(1-cov)+ink*cov
    r=D["red"]*D["m"]; rc=REDC*(0.7+0.5*np.clip(D["L"]*2,0,1))[...,None]
    out=out*(1-r[...,None])+rc*r[...,None]
    frame=((xx>=B)&(xx<SW-B)&(yy>=B)&(yy<SH-BOT))[...,None]; out=np.where(frame,out,paper)
    out+=np.random.default_rng(2).normal(0,3,(SH,SW,1))
    def draw(d,pil):
        d.rectangle((B,B,SW-B-1,SH-BOT-1),outline=(16,15,22),width=6)
        f1=ImageFont.truetype(fm("Inter:black"),120); f2=ImageFont.truetype(fm("Inter:medium"),38)
        spaced(d,"YOUR MOVE.",f1,SH-BOT+70,(16,15,22),4,left=B)
        spaced(d,"NO ONE BLINKS FIRST",f2,SH-BOT+118,(236,24,48),12,right=SW-B)
        d.rectangle((SW-B-560,SH-BOT+100,SW-B,SH-BOT+104),fill=(236,24,48))
    save(out,"t2_standoff",draw)

def halftone(dark,pitch=7.0,ang=np.pi/4):
    c,s=np.cos(ang),np.sin(ang); u=(xx*c+yy*s)/pitch; v=(-xx*s+yy*c)/pitch
    d=np.hypot(u-np.round(u),v-np.round(v)); return np.clip((0.72*np.sqrt(np.clip(dark,0,1))-d)*pitch/1.5+0.5,0,1)
def twoface():                       # yellow halftone | cold dark, split between the eyes
    pw=2300; s=pw/1320; px=1000-540*s; py=400
    D=load("a",px,py,pw); m=D["m"][...,None]; fx,fy=D["P"](540,560)
    # the second eye is a muted gold half-hidden by hair: pick it up with its own looser test
    (bx0,by0),(bx1,by1)=D["P"](585,445),D["P"](700,525); I=D["im"]
    eg=((I[...,0]>I[...,2]*1.2)&(I[...,1]>I[...,2]*1.08)&(I[...,0]>0.38)&(xx>bx0)&(xx<bx1)&(yy>by0)&(yy<by1)).astype(np.float32)
    eg=np.clip((cv2.GaussianBlur(eg,(0,0),2.5)-0.5)*5+0.5,0,1); D["gold"]=np.maximum(D["gold"],eg)
    NAVY=np.array([38,36,96.]); MINT=np.array([196,228,178.]); CREAM=np.array([253,247,222.]); YEL=np.array([248,231,70.])
    base=np.ones((SH,SW,3),np.float32)*MINT; light=np.clip((cv2.GaussianBlur((D["L"]>0.6).astype(np.float32),(0,0),2.5)-0.5)*5+0.5,0,1)
    blue=D["blue"]*(yy>py+1150*s)
    for w,c in((blue,np.array([150,140,214.])),(light,CREAM),(D["gold"],np.array([240,170,60.])),(D["red"],np.array([214,52,74.]))):
        base=base*(1-w[...,None])+c*w[...,None]
    dark=1-np.power(np.clip(cv2.GaussianBlur(D["L"],(0,0),1.5),0,1),0.42); dark=np.clip((dark-0.10)/0.82,0,1)
    ink=np.maximum(halftone(dark),np.clip((0.085-D["L"])*40+0.5,0,1))[...,None]
    ol=cv2.dilate(D["m"],cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))[...,None]
    A=np.ones((SH,SW,3),np.float32)*YEL; A=A*(1-ol)+NAVY*ol; A=A*(1-m)+(base*(1-ink)+NAVY*ink)*m
    fig,fall=dark_fig(D,fx,fy,fall_len=2100,floor=1.0,keep=(1,1),gamma=1.2)
    Bk=np.ones((SH,SW,3),np.float32)*np.array([4,5,12.])+np.exp(-(((xx-fx-500)/900)**2+((yy-fy)/1100)**2))[...,None]*np.array([16,30,78.])
    Bk=Bk*(1-m)+fig*m
    seam=1000+(yy-1500)*-0.10
    geye=eg*(xx>seam)
    Bk+=glow(geye,np.array([255,190,40.]),0.7)
    Bk=finish(Bk,5.5,0.45)
    w=np.clip((xx-seam)/2+0.5,0,1)[...,None]; out=A*(1-w)+Bk*w
    line=np.clip(1-np.abs(xx-seam)/5,0,1)[...,None]; out=out*(1-line)+np.array([250,248,240.])*line
    def draw(d,pil):
        f1=ImageFont.truetype("fonts/Yellowtail-Regular.ttf",250); f2=ImageFont.truetype(fm("Inter:bold"),54)
        d.text((110,110),"i'm ok",font=f1,fill=(214,52,74)); spaced(d,"HAPPY EVERYDAY",f2,385,(40,38,60),2,left=118)
        w1=d.textlength("am i?",font=f1); d.text((1890-w1,110),"am i?",font=f1,fill=(240,236,226))
        spaced(d,"EVERY NIGHT",f2,385,(255,196,40),2,right=1885)
    save(out,"t3_twoface",draw)
for n in sys.argv[1:] or ["pressure","standoff","twoface"]: globals()[n]()
