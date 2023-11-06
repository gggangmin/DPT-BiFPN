import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU(inplace=False)
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=256, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        
        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 3))
        self.w1_relu = nn.ReLU(inplace=False)
        self.w2 = nn.Parameter(torch.Tensor(3, 3))
        self.w2_relu = nn.ReLU(inplace=False)
    
    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x= inputs

        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 = w1/(torch.sum(w1, dim=0) + self.epsilon)
        w2 = self.w2_relu(self.w2)
        w2 = w2/(torch.sum(w2, dim=0) + self.epsilon)

        
        p6_td = p6_x
        p5_td = self.p5_td(w1[0, 0] * p5_x + w1[1, 0] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 1] * p4_x + w1[1, 1] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 2] * p3_x + w1[1, 2] * F.interpolate(p4_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p4_out))
        #p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))
        return [p3_out, p4_out, p5_out, p6_out]
    
class BiFPN(nn.Module):
    def __init__(self, size, feature_size=256, num_layers=3, epsilon=0.0001):
        # num_layers = BIFPN block 수
        super(BiFPN, self).__init__()
        # size = 4,8,16,32
        self.p3 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)
        self.p6 = nn.Conv2d(size[3], feature_size, kernel_size=1, stride=1, padding=0)
        
        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self, inputs):
        c3, c4, c5, c6 = inputs
        # 192, 96, 48, 24
        '''
        torch.Size([1, 256, 12, 12])
        torch.Size([1, 256, 24, 24])
        torch.Size([1, 256, 48, 48])
        torch.Size([1, 256, 96, 96])
        '''
        
        # Calculate the input column of BiFPN
        p3_x = self.p3(c3) 
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(c6)
        # print(p3_x.shape, type(p3_x))
        # print(p4_x.shape, type(p4_x))
        # print(p5_x.shape, type(p5_x))
        # print(p6_x.shape, type(p6_x))
        # print(p7_x.shape, type(p7_x))
        
        features = [p3_x, p4_x, p5_x, p6_x]
        # output has 5 features
        output = self.bifpn(features)
        #print(output)
        return output