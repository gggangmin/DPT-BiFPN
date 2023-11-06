import numpy as np
import torch
import torch.nn as nn
#from FOD.DCNv2 import DeformableConv2d as dcn_v2

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class Fusion(nn.Module):
    def __init__(self, resample_dim):
        super(Fusion, self).__init__()
        self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)
        #self.resample = nn.ConvTranspose2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1)

    def forward(self, x, previous_stage=None):
        if previous_stage == None:
            previous_stage = torch.zeros_like(x)
        output_stage1 = self.res_conv1(x)
        output_stage1 += previous_stage
        output_stage2 = self.res_conv2(output_stage1)
        if output_stage2.shape[-1] != 192:
            output_stage2 = nn.functional.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=True)
        return output_stage2
"""
class FaPNFusion(nn.Module):
    def __init__(self, resample_dim):
        super(FaPNFusion, self).__init__()
        self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)
        self.offset = nn.Conv2d(resample_dim * 2, resample_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.dcpack_L2 =  dcn_v2(resample_dim, resample_dim, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, previous_stage=None):
        if previous_stage == None:
            previous_stage = torch.zeros_like(x)
            x = self.res_conv1(x)
        else:
            x = self.res_conv1(x)
            """"""
            # difference conv
            offset = self.offset(torch.cat([x,previous_stage],dim=1))
            #[feat,offset]
            previous_stage = self.relu(self.dcpack_L2([previous_stage,offset])) #alined
            """"""
        output_stage1 = previous_stage + x
        #x = self.res_conv1(x)
        #output_stage1 = self.res_conv1(x)
        #output_stage1 += previous_stage
        output_stage2 = self.res_conv2(output_stage1)
        if output_stage2.shape[-1] != 192:
            output_stage2 = nn.functional.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=True)
        return output_stage2
"""