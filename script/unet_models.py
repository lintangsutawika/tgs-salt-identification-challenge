#  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#  resnet18 :  BasicBlock, [2, 2, 2, 2]
#  resnet34 :  BasicBlock, [3, 4, 6, 3]
#  resnet50 :  Bottleneck  [3, 4, 6, 3]
#

# https://medium.com/neuromation-io-blog/deepglobe-challenge-three-papers-from-neuromation-accepted-fe09a1a7fa53
# https://github.com/ternaus/TernausNetV2
# https://github.com/neptune-ml/open-solution-salt-detection
# https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution

##############################################################3
#  https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py
#  https://pytorch.org/docs/stable/torchvision/models.html
import os 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

from loss import *
from metric import *

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        # x = self.bn(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

        self.cse = cSEGate(out_channels)
        self.sse = sSEGate(out_channels)

    def forward(self, x ,e=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        if e is not None:
            x = torch.cat([x,e],1)

        x = F.elu(self.conv1(x),inplace=True)
        x = F.elu(self.conv2(x),inplace=True)

        g1 = self.cse(x)
        g2 = self.sse(x)
        x = g1*x + g2*x

        return x

class cSEGate(nn.Module):
    def __init__(self, in_channels):
        super(cSEGate, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv1d(in_channels, in_channels//2, kernel_size=(1,1), stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels//2, in_channels, kernel_size=(1,1), stride=1, padding=0, dilation=1, groups=1, bias=True)
    
    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.sigmoid(self.conv2(x))
        return x

class sSEGate(nn.Module):    
    def __init__(self, in_channels):
        super(sSEGate, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=(1,1), stride=1, padding=0, dilation=1, groups=1, bias=True)
    
    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        return x

# resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
# resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
# resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
# resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
# resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'

class UNetResNet34(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def load_pretrain(self, pretrain_file):
        self.encoder.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self ):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )# 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  #128
        self.encoder4 = self.resnet.layer3  #256
        self.encoder5 = self.resnet.layer4  #512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder5 = Decoder(256+512, 512, 64)#Decoder(512+256, 512, 256)
        self.decoder4 = Decoder(64 +256, 256, 64)#Decoder(256+256, 512, 256)
        self.decoder3 = Decoder(64 +128, 128, 64)#Decoder(128+256, 256,  64)
        self.decoder2 = Decoder(64 + 64,  64, 64)#Decoder( 64+ 64, 128, 128)
        self.decoder1 = Decoder(64     ,  32, 64)#Decoder(128    , 128,  32)

        self.logit    = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64,  1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        #batch_size,C,H,W = x.shape

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0],
        ],1)


        x = self.conv1(x)
        # x = F.max_pool2d(x, kernel_size=2, stride=2)

        e2 = self.encoder2( x)  #; print('e2',e2.size())
        e3 = self.encoder3(e2)  #; print('e3',e3.size())
        e4 = self.encoder4(e3)  #; print('e4',e4.size())
        e5 = self.encoder5(e4)  #; print('e5',e5.size())


        #f = F.max_pool2d(e5, kernel_size=2, stride=2 )  #; print(f.size())
        #f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=True)#False
        #f = self.center(f)                       #; print('center',f.size())
        f = self.center(e5)
        d5 = self.decoder5(f, e5)   #self.decoder5(torch.cat([f, e5], 1))  #; print('d5',f.size())
        d4 = self.decoder4(d5, e4)  #self.decoder4(torch.cat([d5, e4], 1))  #; print('d4',f.size())
        d3 = self.decoder3(d4, e3)  #self.decoder3(torch.cat([d4, e3], 1))  #; print('d3',f.size())
        d2 = self.decoder2(d3, e2)  #self.decoder2(torch.cat([d3, e2], 1))  #; print('d2',f.size())
        d1 = self.decoder1(d2)                      #self.decoder1(d2)                      # ; print('d1',f.size())

        f = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ),1)

        # f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)                     #; print('logit',logit.size())
        return logit

    ##-----------------------------------------------------------------
    def criterion(self, logit, truth ):

        #loss = PseudoBCELoss2d()(logit, truth)
        #loss = FocalLoss2d()(logit, truth, type='sigmoid')
        loss = RobustFocalLoss2d()(logit, truth, type='sigmoid')
        return loss

    # def criterion(self,logit, truth):
    #
    #     loss = F.binary_cross_entropy_with_logits(logit, truth)
    #     return loss

    def metric(self, logit, truth, threshold=0.5 ):
        prob = torch.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


SaltNet = UNetResNet34

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters=32):
        """
        :param num_classes:
        :param num_filters:
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutions are from VGG11
        self.encoder = torchvision.models.vgg11().features
        
        # "relu" layer is taken from VGG probably for generality, but it's not clear 
        self.relu = self.encoder[1]
        
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1, )

    def forward(self, x):

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)

        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        # Deconvolutions with copies of VGG11 layers of corresponding size 
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return F.sigmoid(self.final(dec1))

# Sanity Check
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    batch_size = 8
    C,H,W = 1, 128, 128

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (2,   (batch_size,C,H,W)).astype(np.float32)
    #------------
    if torch.cuda.is_available():
        input = torch.from_numpy(input).float().cuda()
        truth = torch.from_numpy(truth).float().cuda()
        net = SaltNet().cuda()
    else:
        input = torch.from_numpy(input).float()
        truth = torch.from_numpy(truth).float()
        net = SaltNet()
    #---
    
    print("Set Model to Train")
    net.set_mode('train')
    # print(net)
    # exit(0)
    logit = net(input)
    loss  = net.criterion(logit, truth)
    dice  = net.metric(logit, truth)

    print('loss : %0.8f'%loss.item())
    print('dice : %0.8f'%dice.item())
    print('')

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)

    i=0
    optimizer.zero_grad()
    while i<=100:

        logit = net(input)
        loss  = net.criterion(logit, truth)
        dice  = net.metric(logit, truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] loss, dice  :  %0.5f,%0.5f'%(i, loss.item(),dice.item()))
        i = i+1

    print( 'sucessful!')
