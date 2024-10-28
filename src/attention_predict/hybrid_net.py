import torch


class HybridNet(torch.nn.Module):

    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        kernel_size: int = 3,
        padding: str = 'same',
        downsample_factor: int = 2,
        upsample_mode: str = 'nearest',
        final_activation: torch.nn.Module | None = None,
    ):
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.downsample_factor = downsample_factor
        self.upsample_mode = upsample_mode
        self.final_activation = final_activation

        self.encoder_block = torch.nn.ModuleList()
        self.attention_block = torch.nn.ModuleList()
        self.decoder_block = torch.nn.ModuleList()

        for level in range(self.depth):

            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            print(f'left conv: in: {fmaps_in} -> out: {fmaps_out}')
            self.encoder_block.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding
                )
            )

        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            print(f'right conv: in: {fmaps_in} -> out: {fmaps_out}')
            self.decoder_block.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding
                )
            )

        self.downsample = Downsample(self.downsample_factor)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode,
        )
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1],
            self.out_channels,
            self.final_activation
        )

    def forward(self, x):
        # encoder block
        convolution_outputs = []
        layer_input = x

        for i in range(self.depth - 1):

            conv_out = self.encoder_block[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            print(f'left conv{i}: {conv_out.shape} -> {downsampled.shape}')
            layer_input = downsampled

        conv_out = self.encoder_block[-1](layer_input)
        layer_input = conv_out
        print(f'bottle neck: {layer_input.shape}')

        # TODO: attention block

        # decoder block
        for i in range(0, self.depth - 1)[::-1]:

            upsampled = self.upsample(layer_input)
            conv_output = self.decoder_block[i](upsampled)
            print(f'right conv{i}: {layer_input.shape} -> {upsampled.shape} -> {conv_output.shape}')
            layer_input = conv_output

        return self.final_conv(layer_input)


    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:

        """ Compute the number of input and output feature maps for
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
        fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out


class Downsample(torch.nn.Module):

    def __init__(self, downsample_factor: int):

        """Initialize a MaxPool2d module with the input downsample fator"""

        super().__init__()

        self.downsample_factor = downsample_factor
        self.down = torch.nn.MaxPool1d(downsample_factor)

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
        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        self.conv_pass = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )

        for _name, layer in self.named_modules():
            if isinstance(layer, (torch.nn.Conv1d)):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        output = self.conv_pass(x)
        return output


class AttentionBlock(torch.nn.Module):

    def __init__(self, query_dims, key_dims, value_dims):

        self.query_dims = query_dims
        self.key_dims = key_dims
        self.value_dims = value_dims

        self.attention = torch.nn.MultiheadAttention(
            self.query_dims,
            kdim=self.key_dims,
            vdim=self.value_dims,
            num_heads=1,
            batch_first=True,
        )


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
        self.final_conv = torch.nn.Conv1d(in_channels, out_channels, 1, padding=0)
        self.activation = activation

    def forward(self, x):

        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



if __name__ == '__main__':

    depth = 5
    in_channels = 6
    out_channels = 6
    max_len = 400
    # (batch, channels, height, width)
    x = torch.rand(1, in_channels, max_len)
    model = HybridNet(depth, in_channels, out_channels)
    y = model(x)
    print(f'output dim: {y.shape}')

