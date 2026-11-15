import numpy as np, cv2, subprocess
from PIL import Image, ImageDraw, ImageFont
SW,SH=2000,3000
fm=lambda q: subprocess.check_output(["fc-match","-f","%{file}",q]).decode()
im=cv2.imread("src/b.jpg")[:,:,::-1].astype(np.float32)/255
m=cv2.imread("work/b_mask.png",0).astype(np.float32)/255
px,py,pw=-150,250,2300; sc=pw/im.shape[1]; W=pw; H=int(im.shape[0]*sc)
im=cv2.resize(im,(W,H),interpolation=cv2.INTER_LANCZOS4); m=cv2.resize(m,(W,H),interpolation=cv2.INTER_CUBIC)
m=np.clip((cv2.GaussianBlur(m,(0,0),1.5)-0.5)*6+0.5,0,1)
def place(a,fill=0.):           # figure-space layer -> canvas
    out=np.full((SH,SW)+a.shape[2:],fill,np.float32)
    sx0=max(0,-px); sy0=max(0,-py); ex=min(W,SW-px); ey=min(H,SH-py)
    out[py+sy0:py+ey,px+sx0:px+ex]=a[sy0:ey,sx0:ex]; return out
R,G,B=im[...,0],im[...,1],im[...,2]; L=0.299*R+0.587*G+0.114*B
sat=(im.max(2)-im.min(2))/(im.max(2)+1e-4)
red=cv2.GaussianBlur(((R>0.33)&(R>1.7*G)&(R>1.4*B)&(sat>0.45)).astype(np.float32),(0,0),2)
red=np.clip((red-0.5)*5+0.5,0,1)
im_c,m_c,L_c,red_c=place(im),place(m),place(L),place(red)
yy,xx=np.mgrid[0:SH,0:SW].astype(np.float32)
fx,fy=640*sc+px,440*sc+py                                   # face centre on canvas
# light: a hard top-left key that dies quickly down the body
fall=np.clip(1.12-np.clip(yy-fy,0,None)/1500,0,1)**1.7
side=0.42+0.58/(1+np.exp((xx-fx-90)/170))
light=fall*side
t=np.clip((L_c-0.04)*1.15,0,1)**1.25
lo=np.array([5,7,18.]); hi=np.array([196,208,238.])
fig=(lo+(hi-lo)*t[...,None])*light[...,None]+lo*(1-light[...,None])*0.6
# selective colour: red stays alive, only slightly dimmed lower down
rcol=np.array([236,24,48.])*(0.45+0.9*np.clip(L_c*2.2,0,1))[...,None]*(0.35+0.65*fall**0.5)[...,None]
fig=fig*(1-red_c[...,None])+rcol*red_c[...,None]
# cold rim light on the lit side
mb=cv2.GaussianBlur(m_c,(0,0),3); rim=np.clip(mb-np.roll(mb,14,axis=1),0,1)*m_c*fall
fig+=rim[...,None]*np.array([90,130,255.])*0.9
# background: near-black, faint blue bloom behind the head, giant looming shadow of the figure
bg=np.ones((SH,SW,3),np.float32)*np.array([4,5,12.])
bloom=np.exp(-(((xx-fx+250)/900)**2+((yy-fy+150)/1000)**2))
bg+=bloom[...,None]*np.array([16,30,78.])
big=cv2.warpAffine(m_c,np.float32([[1.55,0,fx-1.55*fx+260],[0,1.55,fy-1.55*fy-120]]),(SW,SH))
bg*=(1-0.8*cv2.GaussianBlur(big,(0,0),18))[...,None]
out=bg*(1-m_c[...,None])+fig*m_c[...,None]
# eye glow
eye=red_c*(yy<fy+200)*(yy>fy-250); glow=cv2.GaussianBlur(eye,(0,0),40)*3+cv2.GaussianBlur(eye,(0,0),10)*1.2
out+=glow[...,None]*np.array([255,30,50.])*0.55
# sink the bottom into black for the title, vignette, grain
out*=np.clip(1-np.clip(yy-2050,0,None)/700,0.0,1)[...,None]**1.4
out*=(1-0.5*np.clip(np.hypot((xx-SW/2)/SW,(yy-SH/2)/SH)*1.5-0.45,0,1))[...,None]
out+=np.random.default_rng(5).normal(0,5.5,(SH,SW,1))
pil=Image.fromarray(np.clip(out,0,255).astype(np.uint8)); d=ImageDraw.Draw(pil)
def spaced(txt,font,y,fill,sp,cx=SW//2):
    ws=[d.textlength(ch,font=font) for ch in txt]; tot=sum(ws)+sp*(len(txt)-1); x=cx-tot/2
    for ch,w in zip(txt,ws): d.text((x,y),ch,font=font,fill=fill); x+=w+sp
f1=ImageFont.truetype(fm("Inter:black"),176); f2=ImageFont.truetype(fm("Inter:medium"),44)
spaced("DON'T LOOK",f1,2380,(232,236,246),10); spaced("AWAY",f1,2570,(232,236,246),10)
d.rectangle((SW//2-60,2800,SW//2+60,2805),fill=(236,24,48))
spaced("SHE HAS ALREADY SEEN YOU",f2,2835,(236,24,48),14)
pil.save("work/dark.png"); pil.resize((600,900)).save("work/dark_prev.jpg")
