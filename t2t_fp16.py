"""
Use torch2trt make inference faster
"""
import sys
import onnx
import torch
from torchvision.models.alexnet import alexnet
from torch2trt import TRTModule
from torch2trt import torch2trt

from codebase.lite_unet.model import LightUNet

DEVICE = 0
PTH_PATH = './codebase/lite_unet/models/LightUNet_generator_700_mono_fuji.pth'
TRT_PATH = './codebase/lite_unet/models/LightUNet_generator_700_mono_fuji_fp16.trt'

x = torch.zeros(1, 1, 256, 256).to(DEVICE)    # placeholder

### Load pretrained model from .pth
model = LightUNet(1, 1, 1).to(DEVICE)
model.load_state_dict(torch.load(PTH_PATH, map_location=torch.device(DEVICE)))
model.eval()

### Convert to TensorRT feeding dummy input under fp16 mode
model_trt = torch2trt(model, [x], fp16_mode=True)

### Make inference like using pytorch
y = model(x)
y_trt = model_trt(x)
print(f'precision loss: {torch.max(torch.abs(y - y_trt))}')

### Save converted model for future use
torch.save(model_trt.state_dict(), TRT_PATH)

### Load model from .trt
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(TRT_PATH))