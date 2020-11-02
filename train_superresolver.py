import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
from PIL import Image

downscaled = "formatted/downscaled"
target = "formatted/target"

import os
img_filenames = os.listdir(downscaled)

def display_tensor(x):
    x = np.transpose(x.numpy(), (1, 2, 0))
    img = Image.fromarray((255*x+0.5).astype(np.uint8))
    img.show()

def generate_checkpoint_image(x, y, sfs, device=None):
    nn_upsamp = nn.Upsample(scale_factor=2, mode="nearest")
    bc_upsamp = nn.Upsample(scale_factor=2, mode="bicubic")
    
    if device is not None:
        nn_upsamp = nn_upsamp.to(device)
        bc_upsamp = bc_upsamp.to(device)
        sfs = sfs.to(device)
        x = x.to(device)
        y = y.to(device)
        
    rows = x.shape[0]
    canvas = np.zeros((rows * 32, 32 * 5, 3), dtype=np.uint8)
    with torch.no_grad():
        h = np.transpose(sfs(x).cpu().numpy(), (0, 2, 3, 1))
        n = np.transpose(nn_upsamp(x).cpu().numpy(), (0, 2, 3, 1))
        bc = np.transpose(bc_upsamp(x).cpu().numpy(), (0, 2, 3, 1))
        x = np.transpose(x.cpu().numpy(), (0, 2, 3, 1))
        y = np.transpose(y.cpu().numpy(), (0, 2, 3, 1))

    rescale = lambda x : np.clip(255 * x + 0.5, 0, 255).astype(np.uint8)
        
    for i in range(rows):
        canvas[32*i+8:32*(i+1)-8, 32*0+8:32*1-8, :] = rescale(x[i])
        canvas[32*i:32*(i+1), 32*1:32*2, :] = rescale(n[i])
        canvas[32*i:32*(i+1), 32*2:32*3, :] = rescale(y[i])
        canvas[32*i:32*(i+1), 32*3:32*4, :] = rescale(bc[i])
        canvas[32*i:32*(i+1), 32*4:32*5, :] = rescale(h[i])

    return Image.fromarray(canvas)

def get_batch(batch_size=64):
    filenames = [random.choice(img_filenames) for _ in range(batch_size)]
    x = []
    y = []
    for fname in filenames:
        x_im = np.array(Image.open(os.path.join(downscaled, fname)))
        y_im = np.array(Image.open(os.path.join(target, fname)))
        x.append(x_im)
        y.append(y_im)
    x = np.transpose(np.array(x), (0, 3, 1, 2)) / 255.
    y = np.transpose(np.array(y), (0, 3, 1, 2)) / 255.
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    return x, y


# takes in 16x16 images, produces 32x32
class SmallFaceSuperresolver(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.upsamp = nn.Upsample(scale_factor=2)
        self.conv1 = nn.ConvTranspose2d(3, 32, (5, 5))
        self.conv2 = nn.ConvTranspose2d(32, 64, (5, 5))
        self.conv3 = nn.ConvTranspose2d(64, 128, (5, 5))
        self.conv4 = nn.ConvTranspose2d(128, 256, (5, 5))
        self.conv5 = nn.Conv2d(64 + 256, 256, (1, 1))
        self.conv6 = nn.Conv2d(256, 3, (1, 1))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.05)
        x = F.leaky_relu(self.conv2(x), 0.05)
        c2_skip = self.upsamp(x)
        c2_crop = c2_skip[:,:,8:40,8:40]
        x = F.leaky_relu(self.conv3(x), 0.05)
        x = F.leaky_relu(self.conv4(x), 0.05)
        x = torch.cat((x, c2_crop), dim=1)
        x = F.leaky_relu(self.conv5(x), 0.05)
        x = torch.sigmoid(self.conv6(x))
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sfs = SmallFaceSuperresolver().to(device)

opt = optim.Adam(sfs.parameters(), lr=0.001)

loss_func = nn.L1Loss()

NUM_EPOCHS = 200
MB_PER_EPOCH = 64

cx, cy = get_batch(batch_size=8)

baseline = nn.Upsample(scale_factor=2, mode="bicubic")
baseline_loss = 0.0

for mb in range(MB_PER_EPOCH):
    x, y = get_batch()
    x = x.to(device)
    y = y.to(device)

    predicted = baseline(x)

    loss = loss_func(predicted, y)

    baseline_loss += loss.item() / MB_PER_EPOCH

print("Baseline (bicubic) loss: %0.05f" % baseline_loss)

print()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch #{epoch+1}")
    st = time.time()
    running_loss = 0.0
    for mb in range(MB_PER_EPOCH):
        x, y = get_batch()
        x = x.to(device)
        y = y.to(device)
        
        opt.zero_grad()
        predicted = sfs(x)
        
        loss = loss_func(predicted, y)
        loss.backward()

        opt.step()

        running_loss += loss.item() / MB_PER_EPOCH
    print(" Average loss: %0.05f" % running_loss)
    print(" Time taken: %0.02f" % (time.time()-st))
    generate_checkpoint_image(cx, cy, sfs, device).save("run-checkpoints/checkpoint%03d.png" % epoch)

##baseline = Baseline()
##baseline_loss = 0.0
##for mb in range(MB_PER_EPOCH):
##    x, y = get_batch()
##    x = x.to(device)
##    y = y.to(device)
##
##    predicted = baseline(x)
##
##    loss = loss_func(predicted, y)
##
##    baseline_loss += loss.item() / MB_PER_EPOCH
##
##print("Baseline (Bicubic) loss:",baseline_loss)
