import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import normal_init

from mmedit.ops import resize
from mmedit.models.registry import BACKBONES

import torch.nn.functional as F

@BACKBONES.register_module()
class Upsampler(nn.Module):
    def __init__(self,
            channels,
            conv_cfg=None,
            act_cfg=dict(type='ReLU')):

        super().__init__()

        self.channels = channels
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.output_convs = nn.ModuleList()
        self.output_convs.append(
            ConvModule(
                 self.channels,
                 self.channels,
                 3,
                 stride=1,
                 padding=1,
                 conv_cfg=self.conv_cfg,
                 norm_cfg=None,
                 act_cfg=self.act_cfg))
        
        self.output_convs.append(
            ConvModule(
                 self.channels,
                 3, # for RGB
                 3,
                 stride=1,
                 padding=1,
                 conv_cfg=self.conv_cfg,
                 norm_cfg=None,
                 act_cfg=dict(type='Tanh') # always use Tanh as output act
                 ))

    def forward(self, inputs):
        """Forward function."""

        output = inputs

        for i in range(len(self.output_convs)):
            output = self.output_convs[i](output)

        return output

    def init_weights(self, pretrained):
        """Initialize weights of classification layer."""
        # TODO - apply pretrained