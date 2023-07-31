import torch
import torch.nn as nn
from math import sqrt

class dw_conv(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, padding, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(chan_in, chan_in, kernel_size, stride, padding, groups=chan_in, dilation=dilation, bias=bias)
        self.pointwise = nn.Conv2d(chan_in, chan_out, 1, 1, 0, bias=bias)
        self.relu6 = nn.ReLU6()  # use relu6 instead of relu in low bitwidth
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu6(x)
        return x

class LightUNet(nn.Module):
    """
        Use depthwise separable convolution to reduce the number of parameters.
        A reference: https://github.com/Yingping-LI/Light-U-net
        
        Remember to increase the training epoch since the model is harder to train.
        A reference: https://arxiv.org/pdf/2003.11066.pdf
        
        params count: 429,067 (73% of UNet)
        
    """
    def __init__(self,chan_in, chan_out, long_skip, nf=32):
        super().__init__()
        self.long_skip = long_skip
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.relu = nn.ReLU()
        self.with_bn = True
        self.conv1_1 = dw_conv(self.chan_in, nf, 3, 1, 1)
        self.bn1_1   = nn.BatchNorm2d(nf)
        self.conv1_2 = nn.Conv2d(nf, nf, 3,1,1)
        self.bn1_2   = nn.BatchNorm2d(nf)
        self.conv2_1 = dw_conv(nf, nf*2, 3,1,1)
        self.bn2_1   = nn.BatchNorm2d(nf*2)
        self.conv2_2 = nn.Conv2d(nf*2, nf*2, 3,1,1)
        self.bn2_2   = nn.BatchNorm2d(nf*2)
        self.conv3_1 = dw_conv(nf*2, nf*4, 3,1,1)
        self.bn3_1   = nn.BatchNorm2d(nf*4)
        self.conv3_2 = nn.Conv2d(nf*4, nf*4, 3,1,1)
        self.bn3_2   = nn.BatchNorm2d(nf*4)
        
        self.dc2     =nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1,bias=False)

        self.conv4_1 = dw_conv(nf*4, nf*2, 3,1,1)
        self.bn4_1   = nn.BatchNorm2d(nf*2)
        self.conv4_2 = nn.Conv2d(nf*2, nf*2, 3,1,1)
        self.bn4_2   = nn.BatchNorm2d(nf*2)
        
        self.dc1     =nn.ConvTranspose2d(nf*2, nf, 4, stride=2, padding=1,bias=False)
        
        self.conv5_1 = dw_conv(nf*2, nf, 3,1,1)
        self.bn5_1   = nn.BatchNorm2d(nf)
        self.conv5_2 = nn.Conv2d(nf, nf, 3,1,1)
        self.bn5_2   = nn.BatchNorm2d(nf)
        self.conv5_3 = nn.Conv2d(nf, self.chan_out, 3,1,1)


        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()
        print('initialization weights is done')

    def forward(self, x1):
        if self.with_bn:
            x1_ = self.relu(self.bn1_2(self.conv1_2(self.relu(self.bn1_1(self.conv1_1(x1))))))
            x2 = self.relu(self.bn2_2(self.conv2_2(self.relu(self.bn2_1(self.conv2_1(self.maxpool(x1_)))))))
            x3 = self.relu(self.bn3_2(self.conv3_2(self.relu(self.bn3_1(self.conv3_1(self.maxpool(x2)))))))
            x4 = self.relu(self.dc2(x3))  
            x4_2 = torch.cat((x4, x2), 1)
            x5 = self.relu(self.bn4_2(self.conv4_2(self.relu(self.bn4_1(self.conv4_1(x4_2))))))
            x6 = self.relu(self.dc1(x5))  
            x6_1 = torch.cat((x6, x1_), 1)
            x7 = self.relu(self.bn5_2(self.conv5_2(self.relu(self.bn5_1(self.conv5_1(x6_1))))))
        else:
            x1_ = self.relu(self.conv1_2(self.relu(self.conv1_1(x1))))
            x2 = self.relu(self.conv2_2(self.relu(self.conv2_1(self.maxpool(x1_)))))
            x3 = self.relu(self.conv3_2(self.relu(self.conv3_1(self.maxpool(x2)))))
            x4 = self.relu(self.dc2(x3))  
            x4_2 = torch.cat((x4, x2), 1)
            x5 = self.relu(self.conv4_2(self.relu(self.conv4_1(x4_2))))
            x6 = self.relu(self.dc1(x5))  
            x6_1 = torch.cat((x6, x1_), 1)
            x7 = self.relu(self.conv5_2(self.relu(self.conv5_1(x6_1))))
        x8 = self.conv5_3(x7)
        if self.long_skip == True:        
            return x8 + x1[:,0:self.chan_out,:,:]
        else:
            return x8


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


class LighterUNet(nn.Module):
    """
        Use depthwise separable convolution to reduce the number of parameters.
        A reference: https://github.com/Yingping-LI/Light-U-net
        
        Remember to increase the training epoch since the model is harder to train.
        A reference: https://arxiv.org/pdf/2003.11066.pdf
        
        params count: 219,339 (37% of UNet)
        
    """
    def __init__(self,chan_in, chan_out, long_skip, nf=32):
        super().__init__()
        self.long_skip = long_skip
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.relu = nn.ReLU()
        self.with_bn = True
        self.conv1_1 = dw_conv(self.chan_in, nf, 3, 1, 1)
        self.bn1_1   = nn.BatchNorm2d(nf)
        self.conv1_2 = dw_conv(nf, nf, 3,1,1)
        self.bn1_2   = nn.BatchNorm2d(nf)
        self.conv2_1 = dw_conv(nf, nf*2, 3,1,1)
        self.bn2_1   = nn.BatchNorm2d(nf*2)
        self.conv2_2 = dw_conv(nf*2, nf*2, 3,1,1)
        self.bn2_2   = nn.BatchNorm2d(nf*2)
        self.conv3_1 = dw_conv(nf*2, nf*4, 3,1,1)
        self.bn3_1   = nn.BatchNorm2d(nf*4)
        self.conv3_2 = dw_conv(nf*4, nf*4, 3,1,1)
        self.bn3_2   = nn.BatchNorm2d(nf*4)
        
        self.dc2     =nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1,bias=False)

        self.conv4_1 = dw_conv(nf*4, nf*2, 3,1,1)
        self.bn4_1   = nn.BatchNorm2d(nf*2)
        self.conv4_2 = dw_conv(nf*2, nf*2, 3,1,1)
        self.bn4_2   = nn.BatchNorm2d(nf*2)
        
        self.dc1     =nn.ConvTranspose2d(nf*2, nf, 4, stride=2, padding=1,bias=False)
        
        self.conv5_1 = dw_conv(nf*2, nf, 3,1,1)
        self.bn5_1   = nn.BatchNorm2d(nf)
        self.conv5_2 = dw_conv(nf, nf, 3,1,1)
        self.bn5_2   = nn.BatchNorm2d(nf)
        self.conv5_3 = dw_conv(nf, self.chan_out, 3,1,1)


        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()
        print('initialization weights is done')

    def forward(self, x1):
        if self.with_bn:
            x1_ = self.relu(self.bn1_2(self.conv1_2(self.relu(self.bn1_1(self.conv1_1(x1))))))
            x2 = self.relu(self.bn2_2(self.conv2_2(self.relu(self.bn2_1(self.conv2_1(self.maxpool(x1_)))))))
            x3 = self.relu(self.bn3_2(self.conv3_2(self.relu(self.bn3_1(self.conv3_1(self.maxpool(x2)))))))
            x4 = self.relu(self.dc2(x3))  
            x4_2 = torch.cat((x4, x2), 1)
            x5 = self.relu(self.bn4_2(self.conv4_2(self.relu(self.bn4_1(self.conv4_1(x4_2))))))
            x6 = self.relu(self.dc1(x5))  
            x6_1 = torch.cat((x6, x1_), 1)
            x7 = self.relu(self.bn5_2(self.conv5_2(self.relu(self.bn5_1(self.conv5_1(x6_1))))))
        else:
            x1_ = self.relu(self.conv1_2(self.relu(self.conv1_1(x1))))
            x2 = self.relu(self.conv2_2(self.relu(self.conv2_1(self.maxpool(x1_)))))
            x3 = self.relu(self.conv3_2(self.relu(self.conv3_1(self.maxpool(x2)))))
            x4 = self.relu(self.dc2(x3))  
            x4_2 = torch.cat((x4, x2), 1)
            x5 = self.relu(self.conv4_2(self.relu(self.conv4_1(x4_2))))
            x6 = self.relu(self.dc1(x5))  
            x6_1 = torch.cat((x6, x1_), 1)
            x7 = self.relu(self.conv5_2(self.relu(self.conv5_1(x6_1))))
        x8 = self.conv5_3(x7)
        if self.long_skip == True:        
            return x8 + x1[:,0:self.chan_out,:,:]
        else:
            return x8


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                    

class EffNet_UNet():
    """
        Use matrix decomposition to reduce the number of parameters
        A method orginally proposed in EffNet: AN EFFICIENT STRUCTURE FOR CONVOLUTIONAL NEURAL NETWORKS
    """
    pass


class EffUNet():
    """
        A method proposed in "Eff-UNet: A Novel Architecture for Semantic Segmentation in Unstructured Environment"
        Use mobile inverted bottleneck convolution (MBConv) as the basic building block
    """
    pass

    
class SqueezeUNet():
    pass


class UNeXt():
    pass


class UnetPlusPlus():
    pass


class UNet3Plus():
    pass
