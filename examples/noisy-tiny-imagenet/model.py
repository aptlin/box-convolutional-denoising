import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DnCNN(nn.Module):
    def __init__(
        self, num_blocks=4, input_image_shape=[3, 32, 32], block_num_filters=64
    ):
        super(DnCNN, self).__init__()
        num_input_channels, *_ = input_image_shape

        self.input_convrelu = nn.Sequential(
            nn.Conv2d(
                num_input_channels, block_num_filters, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

        blocks = [
            Block(block_num_filters, block_num_filters)
            for _ in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)

        self.output_conv = nn.Conv2d(
            block_num_filters, num_input_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        input_ = self.input_convrelu(x)
        blocks = self.blocks(input_)
        output = self.output_conv(blocks)
        return output
