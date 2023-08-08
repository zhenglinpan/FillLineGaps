import os, sys, argparse

sys.path.append('../..')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.cuda.amp import GradScaler, autocast

from zhenglin.dl.networks.unet import UNet
from zhenglin.dl.networks.discriminator import Discriminator
from zhenglin.dl.utils import LinearLambdaLR

from tqdm import tqdm

from dataset import AnimeSketch
from model import LightUNet, LighterUNet
from zhenglin.dl.networks.unet import UNet

parser = argparse.ArgumentParser()
### dataset args
parser.add_argument('--dataroot', type=str, default='/home/zhenglin/SketchCloser/datasets/fuji', help='root directory of the dataset')
parser.add_argument('--patch_size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--num_workers', type=int, default=4, help='number of cpu threads to use during batch generation')
### training args
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--end_epoch', type=int, default=600, help='number of epochs of training')
parser.add_argument('--decay_epoch', type=int, default=400, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate')
parser.add_argument('--resume', action="store_true", help='continue training from a checkpoint')
args = parser.parse_args()

### set gpu device
DEVICE = 0

generator = LightUNet(1, 1, 1).to(DEVICE)

criterion_pixel = nn.MSELoss().to(DEVICE)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LinearLambdaLR(args.end_epoch, args.start_epoch, args.decay_epoch).step)

grad_scaler = GradScaler()

Tensor = torch.cuda.FloatTensor
batch_shape = (args.batch_size, args.input_nc, args.patch_size, args.patch_size)
target_real = Variable(torch.ones(batch_shape), requires_grad=False).to(DEVICE)
target_fake = Variable(torch.zeros(batch_shape), requires_grad=False).to(DEVICE)

dataset = AnimeSketch(args.dataroot, args.patch_size, color=False)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

generator.train()

for epoch in tqdm(range(args.start_epoch, args.end_epoch + 1)):
    for i, batch in enumerate(dataloader):
        gt = Variable((batch['gt']).type(Tensor)).to(DEVICE)   # line color: black
        opened = Variable((batch['opened']).type(Tensor)).to(DEVICE)
        mask = Variable(batch['mask'].type(Tensor)).to(DEVICE)
        attacked = batch['attack']
        
        if any(attacked):
            continue    # skip this iteration
        
        ##### Generator #####
        optimizer_G.zero_grad()
        with autocast():
            closed = generator(opened)
            
            loss_pixel = criterion_pixel(closed, gt)
            
            loss_G = loss_pixel * 10.0

        grad_scaler.scale(loss_G).backward()
        grad_scaler.step(optimizer_G)
        grad_scaler.update()
        
    lr_scheduler_G.step()
    
    if epoch % 20 == 0:
        save_image(closed[0], f'imgs/{epoch}_fake.png')
        save_image(gt[0], f'imgs/{epoch}_real.png')
        save_image(opened[0], f'imgs/{epoch}_opened.png')
        save_image(mask[0] * 255, f'imgs/{epoch}_mask.png')
        
    if epoch % 100 == 0:
        torch.save(generator.state_dict(), f'models/LightUNet_mix_generator_{epoch}.pth')