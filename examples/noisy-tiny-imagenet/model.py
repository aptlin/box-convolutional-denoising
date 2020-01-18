import torch.nn.functional as F
from torch import nn

from box_convolution import BoxConv2d


class BottleneckBoxConv(nn.Module):
    def __init__(
        self,
        in_channels,
        num_boxes,
        max_input_h,
        max_input_w,
        dropout_prob=0.0,
        reparam_factor=1.5625,
    ):

        super().__init__()
        assert (
            in_channels % num_boxes == 0
        ), "Input channels must must be divisible by the number of boxes"
        bt_channels = in_channels // num_boxes  # bottleneck channels

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1, 1), bias=False),
            nn.BatchNorm2d(bt_channels),
            nn.ReLU(True),
            BoxConv2d(
                bt_channels,
                num_boxes,
                max_input_h,
                max_input_w,
                reparametrization_factor=reparam_factor,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout_prob),
        )

    def forward(self, x):
        return F.relu(x + self.main_branch(x))


class BoxDnCNN(nn.Sequential):
    def __init__(
        self,
        in_channels=3,
        out_features=64,
        num_blocks=17,
        num_boxes=4,
        max_input_h=64,
        max_input_w=64,
    ):
        assert (
            out_features % num_boxes == 0
        ), "Input channels must be divisible by the number of boxes"

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            *[
                BottleneckBoxConv(
                    out_features, num_boxes, max_input_h, max_input_w
                )
                for _ in range(num_blocks)
            ],
            nn.Conv2d(out_features, in_channels, kernel_size=3, padding=1)
        )
