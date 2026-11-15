import sys
from rembg import remove, new_session
from PIL import Image
s=new_session("isnet-anime")
for n in "ab":
    im=Image.open(f"src/{n}.jpg").convert("RGB")
    remove(im,session=s,only_mask=True).save(f"work/{n}_mask.png"); print(n,im.size)
