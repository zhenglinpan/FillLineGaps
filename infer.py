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

img = cv2.imread('imgs/1002.png', cv2.IMREAD_GRAYSCALE)

print(img.shape)

img = img[None, None, ...]
print(img.shape)
model = UNet(1, 1, 1).to(DEVICE)

model.load_state_dict(torch.load(model_dir, map_location=torch.device(DEVICE)))
model.eval()

print("inference starts ============>")

out_img = np.zeros_like(img)

start_time = time.time()
for i in range(2):  # patch_size: 256, 256
    for j in range(4):
        print(f'patch_{i}_{j}===================>')
        patch = img[:, :, i*256: (i+1)*256, j*256: (j+1)*256] / 255
        Patch = Variable(torch.from_numpy(patch).type(torch.cuda.FloatTensor), requires_grad=False).to(DEVICE)
        out = model(Patch)
        save_image(out, f'imgs/res_{i}_{j}.png')
        save_image(Patch, f'imgs/ori_{i}_{j}.png')
print(f"inference finished, cost time: {time.time() - start_time} ============>")
