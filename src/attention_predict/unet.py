# This script includes an implementation of the UNet model that is based on the version
# available at:
# https://github.com/dlmbl/unet/blob/c9753f92e40b0df969f21241b79ab375ffdb1ccc/solution.py
#
# Acknowledgments:
# The original implementation of the UNet model was retrieved from the Deep Learning at
# MBL (dlmbl) repository.
# Significant portions of this script are either unchanged or modified versions of the
# original code.
# We extend our gratitude to the contributors of the dlmbl repository for making their
# code available for reuse and adaptation.
#
# License Information:
# The original code from which this script is derived is subject to the terms of the MIT
# license, as stated in the repository.
from torch import Tensor
import torch.nn as nn


class ConvBlock(nn.Module):
    conv_pass: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str,
        num_groups: int,
    ) -> None:
        """A convolution block for a U-Net. Contains two convolutions, each followed by
            a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends
                on the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block.
                Depends on the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str): The type of convolution padding to use. Either "same" or
                "valid". Defaults to "same".
        """
        super().__init__()

        conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_groups,
        )
        conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_groups,
        )
        # Parameter initializations
        nn.init.kaiming_normal_(conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(conv2.weight, nonlinearity="relu")

        self.conv_pass = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
        )

    def forward(self, x) -> Tensor:
        return self.conv_pass(x)
