import os, sys, argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# from zhenglin.dl.template.v1.train
from zhenglin.dl.networks.unet import UNet
from zhenglin.dl.networks.discriminator import Discriminator
from zhenglin.dl.utils import LinearLambdaLR
# from zhenglin.dl.networks_.noise2void.noise2void

from tqdm import tqdm
# import wandb
# wandb.init(project="sketch closer")

from dataset_ import AnimeSketchNoise2Void

parser = argparse.ArgumentParser()
### dataset args
parser.add_argument('--dataroot', type=str, default='/home/zhenglin/AnimationSketchCloser/datasets', help='root directory of the dataset')
parser.add_argument('--patch_size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--num_workers', type=int, default=4, help='number of cpu threads to use during batch generation')
### training args
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--end_epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--resume', action="store_true", help='continue training from a checkpoint')
args = parser.parse_args()

### set gpu device
DEVICE = 0

generator = UNet(1, 1, 0).to(DEVICE)
discriminator = Discriminator(1, 1).to(DEVICE)

criterion_gan = nn.L1Loss().to(DEVICE)
criterion_pixel = nn.MSELoss().to(DEVICE)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
optimizer_D = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LinearLambdaLR(args.end_epoch, args.start_epoch, args.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LinearLambdaLR(args.end_epoch, args.start_epoch, args.decay_epoch).step)

Tensor = torch.cuda.FloatTensor
batch_shape = (args.batch_size, args.input_nc, args.patch_size, args.patch_size)
target_real = Variable(torch.ones(batch_shape), requires_grad=False).to(DEVICE)
target_fake = Variable(torch.zeros(batch_shape), requires_grad=False).to(DEVICE)

dataset = AnimeSketchNoise2Void(args.dataroot, args.patch_size, kernel=10)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

generator.train()
discriminator.train()

for epoch in tqdm(range(args.start_epoch, args.end_epoch + 1)):
    for i, batch in enumerate(dataloader):
        clear = Variable(batch['clear'].type(Tensor)).to(DEVICE)   # line color: black
        blind = Variable(batch['blind'].type(Tensor)).to(DEVICE)
        mask = Variable(batch['mask'].type(Tensor)).to(DEVICE)
        attacked = Variable(batch['attack'].type(Tensor)).to(DEVICE)
        
        if attacked:
            continue    # skip this iteration
        
        ##### Generator #####
        optimizer_G.zero_grad()
        
        closed = generator(blind)
        pred_fake = discriminator(closed)
        
        loss_pixel = criterion_pixel(closed * mask, clear * mask)
        loss_gan = criterion_gan(pred_fake, target_real)
        
        loss_G = loss_pixel * 1.0 + loss_gan * 1.0
        
        loss_G.backward()
        optimizer_G.step()

        ##### Discriminator ###
        optimizer_D.zero_grad()
        
        pred_fake = discriminator(closed.detach())
        pred_true = discriminator(clear)
        
        loss_real = criterion_gan(pred_fake, target_fake)
        loss_fake = criterion_gan(pred_true, target_real)
        
        loss_D = loss_real * 0.5 + loss_fake * 0.5
        
        loss_D.backward()
        optimizer_D.step()
        
        # wandb.log({'loss_G': loss_G.item(), 
        #            'loss_D': loss_D.item(), 
        #            'loss_pixel': loss_pixel.item(),
        #            'loss_gan': loss_gan.item(),
        #            'loss_real': loss_real.item(),
        #            'loss_fake': loss_fake.item()})
        
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    
    if epoch % 10 == 0:
        save_image(closed[0], f'imgs/{epoch}_fake.png')
        save_image(clear[0], f'imgs/{epoch}_real.png')
        save_image(blind[0], f'imgs/{epoch}_opened.png')
        save_image(mask[0], f'imgs/{epoch}_mask.png')
        
    if epoch % 20 == 0:
        torch.save(generator.state_dict(), f'models/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'models/discriminator_{epoch}.pth')
        
# TODO 1: Blind Mask √
# TODO 2: JPEG AND PNG √