source = "archive/img_align_celeba/img_align_celeba"
downscaled = "formatted/downscaled"
target = "formatted/target"

import os
import face_recognition
import numpy as np
from PIL import Image

def downscale(img, sf):
    w, h = img.shape[:2]
    nw, nh = w // sf, h // sf
    dtype = img.dtype.type
    nimg = np.zeros((nw, nh, img.shape[2]), dtype=dtype)
    for i in range(nw):
        for j in range(nh):
            for c in range(3):
                nimg[i,j,c] = dtype(img[sf*i:sf*(i+1),sf*j:sf*(j+1),c].mean() + 0.5)
    return nimg

imgs = os.listdir(source)
idx = 0
for filename in imgs:
    im = face_recognition.load_image_file(os.path.join(source, filename))
    faces = face_recognition.face_locations(im)
    if len(faces) == 0:
        continue
    face = faces[0]
    top, right, bot, left = face
    v = (top + bot) // 2
    h = (right + left) // 2
    crop = im[v - 64 : v + 64, h - 64 : h + 64]
    if crop.shape != (128, 128, 3):
        continue
    
    x32 = downscale(crop, 4)
    x16 = downscale(x32, 2)
    
    x32 = Image.fromarray(x32)
    x16 = Image.fromarray(x16)
    crop = Image.fromarray(crop)

    x16.save(os.path.join(downscaled, "%03d.png" % idx))
    x32.save(os.path.join(target, "%03d.png" % idx))
    crop.save("here.png")
    
    idx += 1
    break
