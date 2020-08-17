# -*- coding:utf-8 -*-
###
# File: shufflenetv2.py
# Created Date: Sunday, August 16th 2020, 4:19:55 pm
# Author: yusnows
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 yusnows
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY


__all__ = [
    'InvertedResidual', 'ShuffleNetV2', 'build_shufflenetv2_backbone',
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(CNNBlockBase):
    def __init__(self, inp, oup, stride, norm="BN"):
        super(InvertedResidual, self).__init__(inp, oup, stride)

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1, norm=norm),
                Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False,
                    norm=get_norm(norm, branch_features)),
                nn.ReLU(inplace=True),)

        self.branch2 = nn.Sequential(
            Conv2d(
                inp if(self.stride > 1) else branch_features, branch_features, kernel_size=1,
                stride=1, padding=0, bias=False, norm=get_norm(norm, branch_features)),
            nn.ReLU(inplace=True),
            self.depthwise_conv(
                branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1,
                norm=norm),
            Conv2d(
                branch_features, branch_features, kernel_size=1, stride=1, padding=0,
                bias=False, norm=get_norm(norm, branch_features)),
            nn.ReLU(inplace=True),)

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False, norm="BN"):
        return Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i, norm=get_norm(norm, o))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetSepBlock(CNNBlockBase):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, bias=False, pooling=True, norm="BN"):
        super(ShuffleNetSepBlock, self).__init__(input_channels, output_channels, 4)
        self.pooling = pooling
        self.conv = nn.Sequential(
            Conv2d(
                input_channels, output_channels, kernel_size, stride, padding, bias=bias,
                norm=get_norm(norm, output_channels)),
            nn.ReLU(inplace=True))
        if pooling:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        if self.pooling:
            x = self.maxpool(x)
        return x


class ShuffleNetV2(Backbone):
    def __init__(self, input_channels, stages_repeats, stages_out_channels, norm="BN", num_classes=None, out_features=None):
        super(ShuffleNetV2, self).__init__()
        self.num_classes = num_classes
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        output_channels = self._stage_out_channels[0]
        self.stem = ShuffleNetSepBlock(input_channels, output_channels, 3, 2, 1, bias=False, pooling=True, norm=norm)
        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": output_channels}

        input_channels = output_channels

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        self.stages_and_names = []
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            stage = nn.Sequential(*seq)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in seq])
            )
            self._out_feature_channels[name] = seq[-1].out_channels
            input_channels = output_channels

        if num_classes is not None:
            output_channels = self._stage_out_channels[-1]
            stage5 = [ShuffleNetSepBlock(input_channels, output_channels, 1, 1, 0, bias=False, pooling=True)]
            self.stage5 = nn.Sequential(*stage5)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(output_channels, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"
        if out_features is None:
            out_feaures = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.stage5(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ShuffleNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self


def _shufflenetv2(arch, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def build_shufflenetv2_backbone(cfg, input_shape):
    width_mul = cfg.MODEL.SHUFFLENETV2.WIDTH_MUL
    norm = cfg.MODEL.SHUFFLENETV2.NORM
    stages_repeats = [4, 8, 4]
    stages_out_channels = {
        "W0_5": [24, 48, 96, 192, 1024],
        "W1_0": [24, 116, 232, 464, 1024],
        "W1_5": [24, 176, 352, 704, 1024],
        "W2_0": [24, 244, 488, 976, 2048]}[width_mul]
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.SHUFFLENETV2.OUT_FEATURES
    model = ShuffleNetV2(input_shape.channels, stages_repeats,
                         stages_out_channels, norm=norm, out_features=out_features)
    return model.freeze(freeze_at)
