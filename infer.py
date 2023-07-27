from codebase.resnet18.model import ResNet18
from zhenglin.dl.networks.unet import UNet

import cv2
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
# from zhenglin.dl.template.v1.train
import matplotlib.pyplot as plt
import time

DEVICE = 0

model_dir = '/home/zhenglin/AnimationSketchCloser/codebase/unet/models/generator_500.pth'

img = cv2.imread('imgs/1002.png', cv2.COLOR_BGR2GRAY)
img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('reshaped.png', img)

print(img.shape)

img = img[None, None, ...]
print(img.shape)
# model = ResNet18().to(DEVICE)
model = UNet(1, 1, 1).to(DEVICE)

model.load_state_dict(torch.load(model_dir, map_location=torch.device(DEVICE)))
model.eval()

print("inference starts ============>")

out_img = np.ones_like(img)

start_time = time.time()
for i in range(2):  # patch_size: 256, 256
    for j in range(4):
        print(f'patch_{i}_{j}===================>')
        patch = img[:, :, i*256: (i+1)*256, j*256: (j+1)*256] / 255
        Patch = Variable(torch.from_numpy(patch).type(torch.cuda.FloatTensor), requires_grad=False).to(DEVICE)
        out = model(Patch)
        out = out.cpu().detach().numpy()
        cv2.imwrite(f'imgs/res_{i}_{j}.png', (out[0, 0, :, :] * 255).astype(np.uint8))
        
        # out_img[:, :, i*256: (i+1)*256, j*256: (j+1)*256] = out.cpu().detach().numpy()
print(f"inference finished, cost time: {time.time() - start_time} ============>")

# cv2.imwrite('res.png', (out_img[0, 0, :, :]).astype(np.uint8))