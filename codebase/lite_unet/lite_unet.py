from zhenglin.dl.utils import summary
from model import LightUNet, LighterUNet
from zhenglin.dl.networks.unet import UNet

model0 = UNet(1, 1, 1)
model1 = LightUNet(1, 1, 1)
model2 = LighterUNet(1, 1, 1)

summary(model0)
summary(model1)
summary(model2)