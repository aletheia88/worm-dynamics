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
import torch


class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
        unet_dim: int = 2,
    ):
        """A U-Net for 2D input that expects tensors shaped like::
            ``(batch, channels, height, width)``.
        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
                or "bilinear." Defaults to "nearest."
            unet_dim (optional):
                The dimension of the U-Net. Choices are U-Net 1D, 2D and 3D.
                Default to 2.
        """
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode
        self.unet_dim = unet_dim

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()

        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            #print(f'left conv: in: {fmaps_in} -> out: {fmaps_out}')
            self.left_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding,
                    ndim=self.unet_dim
                )
            )

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()

        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            #print(f'right conv: in: {fmaps_in} -> out: {fmaps_out}')
            self.right_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding,
                    ndim=self.unet_dim
                )
            )

        self.downsample = Downsample(self.downsample_factor, ndim=self.unet_dim)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode,
        )
        self.crop_and_concat = CropAndConcat()
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1],
            self.out_channels,
            self.final_activation,
            ndim=self.unet_dim
        )

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        if level == 0:
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the decoder convolutional pass in the given level.
        """
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)
        concat_fmaps = self.compute_fmaps_encoder(level)[1]
        # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, x):

        # left side
        convolution_outputs = []
        layer_input = x

        for i in range(self.depth - 1):

            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            #print(f'left conv{i}: {conv_out.shape} -> {downsampled.shape}')
            layer_input = downsampled

        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out
        #print(f'bottle neck: {layer_input.shape}')

        # right
        for i in range(0, self.depth - 1)[::-1]:

            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            #print(f'right conv{i}: {layer_input.shape} -> {upsampled.shape} -> {concat.shape} -> {conv_output.shape}')
            layer_input = conv_output

        return self.final_conv(layer_input)


class Downsample(torch.nn.Module):

    def __init__(self, downsample_factor: int, ndim: int = 2):

        """Initialize a MaxPool2d module with the input downsample fator"""

        super().__init__()

        if ndim not in (1, 2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 1, 2 or 3."
            raise ValueError(msg)

        self.downsample_factor = downsample_factor
        # Define what the downop should be based on `ndim`.
        downops = {1: torch.nn.MaxPool1d, 2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        self.down = downops[ndim](downsample_factor)

    def check_valid(self, image_size: tuple[int, ...]) -> bool:
        """Check if the downsample factor evenly divides each image dimension.
        Returns `True` for valid image sizes and `False` for invalid image sizes.
        Note: there are multiple ways to do this!
        """
        for dim in image_size:
            if dim % self.downsample_factor != 0:
                return False
        return True

    def forward(self, x):

        if not self.check_valid(tuple(x.size()[2:])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


class ConvBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
        ndim: int = 2,
    ):
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
            ndim (int): Number of dimensions for the convolution operation. Use 2 for 2D
                convolutions and 3 for 3D convolutions. Defaults to 2.
        """
        super().__init__()
        if ndim not in (1, 2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 1, 2 or 3."
            raise ValueError(msg)
        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        convops = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        self.conv_pass = torch.nn.Sequential(
            convops[ndim](
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            convops[ndim](
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )

        for _name, layer in self.named_modules():
            if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        output = self.conv_pass(x)
        return output


class OutputConv(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: torch.nn.Module | None = None,
        ndim: int = 2,
    ):
        """
        A module that uses a convolution with kernel size 1 to get the appropriate
        number of output channels, and then optionally applies a final activation.

        Args:
            in_channels (int): The number of feature maps that will be input to the
                OutputConv block.
            out_channels (int): The number of channels that you want in the output
            activation (str | None, optional): Accepts the name of any torch activation
                function  (e.g., ``ReLU`` for ``torch.nn.ReLU``) or None for no final
                activation. Defaults to None.
            ndim (int): Number of dimensions for convolution operation. Use 2 for 2D
                convolutions and 3 for 3D convolutions. Defaults to 2.
        """
        super().__init__()
        if ndim not in (1, 2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 1, 2 or 3."
            raise ValueError(msg)
        # Use the convolution module matching `ndim`.
        convops = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        self.final_conv = convops[ndim](in_channels, out_channels, 1, padding=0)

        self.activation = activation

    def forward(self, x):

        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CropAndConcat(torch.nn.Module):

    def forward(self, encoder_output, upsample_output):
        encoder_cropped = center_crop(encoder_output, upsample_output)

        return torch.cat([encoder_cropped, upsample_output], dim=1)


def center_crop(x, y):

    """Center-crop x to match spatial dimensions given by y."""

    x_target_size = x.size()[:2] + y.size()[2:]

    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]


if __name__ == "__main__":

    depth = 5
    in_channels = 5
    out_channels = 5
    unet_dim = 1

    # (batch, channels, height, width)
    x = torch.rand(1, in_channels, 1600)
    model = UNet(depth, in_channels, out_channels, unet_dim=unet_dim)
    y = model(x)
    print(f'output dim: {y.shape}')

