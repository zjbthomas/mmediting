import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import normal_init

from mmedit.ops import resize
from mmedit.models.registry import BACKBONES

import torch.nn.functional as F

import math

@BACKBONES.register_module()
class Upsampler(nn.Module):
    def __init__(self,
            channels,
            scale=2,
            conv_cfg=None,
            act_cfg=dict(type='ReLU')):

        super().__init__()

        self.channels = channels
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.output_convs = nn.ModuleList()

        # first layers
        self.first_conv = ConvModule(
                 3, # for RGB
                 self.channels, 
                 3,
                 stride=1,
                 padding=1,
                 conv_cfg=self.conv_cfg,
                 norm_cfg=None,
                 act_cfg=self.act_cfg)

        # upsampling layers
        num_upsample = int(math.log(scale, 2))
        for _ in range(num_upsample):
            upsample_conv = ConvModule(
                 self.channels,
                 self.channels,
                 3,
                 stride=1,
                 padding=1,
                 conv_cfg=self.conv_cfg,
                 norm_cfg=None,
                 act_cfg=self.act_cfg)
            self.output_convs.append(upsample_conv)
        
        # final output layers
        self.final_conv = ConvModule(
                 self.channels,
                 3, # for RGB
                 3,
                 stride=1,
                 padding=1,
                 conv_cfg=self.conv_cfg,
                 norm_cfg=None,
                 act_cfg=None) # no act at the end

    def forward(self, inputs):
        """Forward function."""

        output = self.first_conv(inputs)

        for i in range(len(self.output_convs)):
            output = self.output_convs[i](F.interpolate(output, scale_factor=2, mode='nearest'))

        output = self.final_conv(output)

        return output

    def init_weights(self, pretrained):
        """Initialize weights of classification layer."""
        # TODO - apply pretrained