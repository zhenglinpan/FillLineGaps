from codebase.resnet18.model import ResNet18
from codebase.lite_unet.model import LightUNet
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

# model_dir = '/home/zhenglin/SketchCloser/codebase/lite_unet/models/LightUNet_generator_1000_color_dls.pth'
model_dir = '/home/zhenglin/SketchCloser/codebase/lite_unet/models/generator_700_mono_mono_fuji.pth'

img = cv2.imread('/home/zhenglin/SketchCloser/imgs/frame_1136.png', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('/home/zhenglin/SketchCloser/datasets/0171.png')
# img = cv2.imread('/home/zhenglin/SketchCloser/imgs/frame_1002_paint.png')

print("img.shape: ", img.shape)

if len(img.shape) == 2:
    img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)[1]
    img = img[None, None, ...]
else:
    img = img.transpose(2, 0, 1)[None, ...]

model = LightUNet(1, 1, 1).to(DEVICE)

model.load_state_dict(torch.load(model_dir, map_location=torch.device(DEVICE)))
model.eval()

print("inference starts ============>")

out_img = np.zeros_like(img)

start_time = time.time()
for i in range(3):  # patch_size: 256, 256
    for j in range(5):
        print(f'patch_{i}_{j}===================>')
        patch = img[:, :, i*256: (i+1)*256, j*256: (j+1)*256] / 255
        Patch = Variable(torch.from_numpy(patch).type(torch.cuda.FloatTensor), requires_grad=False).to(DEVICE)
        out = model(Patch)
        save_image(out, f'imgs/dls_res_{i}_{j}.png')
        save_image(Patch, f'imgs/dls_ori_{i}_{j}.png')
print(f"inference finished, cost time: {time.time() - start_time} ============>")

# start_time = time.time()
# for i in range(2):
#     for j in range(2):
#         print(f'patch_{i}_{j}===================>')
#         patch = img[:, :, i*270: (i+1)*270, j*480: (j+1)*480] / 255
#         Patch = Variable(torch.from_numpy(patch).type(torch.cuda.FloatTensor), requires_grad=False).to(DEVICE)
#         print(Patch.shape)
#         out = model(Patch)
#         save_image(out, f'imgs/dls_res_{i}_{j}.png')
#         save_image(Patch, f'imgs/dls_ori_{i}_{j}.png')
# print(f"inference finished, cost time: {time.time() - start_time} ============>")