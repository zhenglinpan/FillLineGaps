"""
Use torch2trt make inference faster
"""
import sys
import onnx
import torch
from PIL import Image
import tensorrt as trt
from torchvision.models.alexnet import alexnet
from torchvision import transforms

from torch2trt import TRTModule
from torch2trt import torch2trt

from glob import glob
from codebase.lite_unet.model import LightUNet

DEVICE = 0
PTH_PATH = './codebase/lite_unet/models/LightUNet_generator_700_mono_fuji.pth'
TRT_PATH = './codebase/lite_unet/models/LightUNet_generator_700_mono_fuji_int8.trt'
CALIBRATION_FILE_ROOT = './datasets/fuji'

class CalibrationDataset(): # no need to inherit
    """
    Taking target images from training set as calibration dataset
    """
    def __init__(self, file_root, crop_size):
        self.file_root = file_root
        self.crop_size = crop_size
                
        self.file_list = glob(self.file_root + '/*.png')
        self.trans = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        img = self.trans(img)
        img = [None, ...]
        return [img]


### Load pretrained model from .pth
model = LightUNet(1, 1, 1).to(DEVICE)
model.load_state_dict(torch.load(PTH_PATH, map_location=torch.device(DEVICE)))
model.eval()

### Convert to TensorRT feeding dummy input under fp16 mode
x = torch.zeros(1, 1, 256, 256).to(DEVICE)
dataset = CalibrationDataset(CALIBRATION_FILE_ROOT, x.shape[-1])
model_trt = torch2trt(model, [x], int8_calib_dataset=dataset, int8_calib_algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION)

### Make inference like using pytorch
y = model(x)
y_trt = model_trt(x)
print(f'precision loss: {torch.max(torch.abs(y - y_trt))}')

### Save converted model for future use
torch.save(model_trt.state_dict(), TRT_PATH)

### Load model from .trt
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(TRT_PATH))