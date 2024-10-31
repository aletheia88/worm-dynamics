import torch


class HybridNet(torch.nn.Module):

    def __init__(
        self,
        depth: int,
        num_inputs: int,
        window_size: int,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        kernel_size: int = 3,
        padding: str = 'same',
        downsample_factor: int = 2,
        hidden_dims: int = 512,
        upsample_mode: str = 'nearest',
        final_activation: torch.nn.Module | None = None,
        device: str = 'cpu',
    ):
        super().__init__()

        self.depth = depth
        self.num_inputs = num_inputs
        self.window_size = window_size
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.downsample_factor = downsample_factor
        self.hidden_dims = hidden_dims
        self.upsample_mode = upsample_mode
        self.final_activation = final_activation
        self.device = device

        self.downsample = Downsample(self.downsample_factor)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode)

        self.encoder_block = torch.nn.ModuleList()
        self.decoder_block = torch.nn.ModuleList()

        for level in range(self.depth):

            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            if level == 0:
                fmaps_in = 1
            self.encoder_block.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding
                ).to(self.device)
            )

        self.one_hots = torch.eye(self.num_inputs, device=self.device)
        self.latent_xdims = self.window_size // (2**(self.depth - 1)) # 25
        self.latent_ydims = fmaps_out # 1024

        # `embedding_dims` = 1024 * (400/2^4) * N + N
        # embedding_dims = self.num_inputs + self.latent_xdims * \
        #         self.latent_ydims * self.depth
        embedding_dims = self.num_inputs + self.latent_xdims * self.latent_ydims

        self.attention_block = AttentionBlock(
            embedding_dims,
            self.num_inputs,
            device=self.device
        )
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            self.decoder_block.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding
                ).to(self.device)
            )
        self.feedforward = FeedforwardBlock(
            embedding_dims,
            self.hidden_dims,
            self.latent_xdims,
            self.latent_ydims,
        ).to(self.device)

        self.final_conv = OutputConv(
            in_channels=self.compute_fmaps_decoder(0)[1],
            out_channels=1,
            activation=self.final_activation
        ).to(self.device)

    def forward(self, inputs):

        num_samples = inputs.shape[0]
        attention_inputs = []

        ### encoder block ###
        for i in range(self.num_inputs):

            embedded_outputs = []
            layer_input = inputs[:, i, :].unsqueeze(1)

            for j in range(self.depth - 1):

                conv_out = self.encoder_block[j](layer_input)
                # if j == 3:
                #     embedded_outputs.append(conv_out.view(num_samples, -1))
                downsampled = self.downsample(conv_out)
                layer_input = downsampled

            conv_out = self.encoder_block[-1](layer_input) # bottleneck block
            flattened_features = conv_out.view(num_samples, -1)
            embedded_outputs.append(flattened_features)

            embedded_outputs = torch.cat(embedded_outputs, 1).to(self.device)

            one_hot_encoding = self.one_hots[i].repeat(num_samples, 1)

            embedding = torch.cat((embedded_outputs, one_hot_encoding), 1)
            attention_inputs.append(embedding)

        ### attention block ###
        attention_inputs = torch.stack(attention_inputs, 1).to(self.device)
        # attention_weights: (b, num_inputs, num_inputs)
        # attention_outputs: (b, num_inputs, embedding_dims)
        attention_outputs, attention_weights = self.attention_block(attention_inputs)

        ### decoder block ###
        decoded_outputs = []
        for i in range(self.num_inputs):

            # layer_input: (b, embedding_dims)
            attention_output = attention_outputs[:, i, :].unsqueeze(1)

            ### feedforward block ###
            layer_input = self.feedforward(attention_output)

            for j in range(0, self.depth - 1)[::-1]:

                upsampled = self.upsample(layer_input)
                conv_output = self.decoder_block[j](upsampled)
                layer_input = conv_output

            final_conv = self.final_conv(layer_input)
            decoded_outputs.append(final_conv)

        model_outputs = torch.cat(decoded_outputs, 1).to(self.device)

        return model_outputs, attention_weights

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
            fmaps_in = self.num_inputs
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

    def __init__(self, embedding_dims, num_inputs, device):

        super().__init__()
        self.device = device
        self.attention = torch.nn.MultiheadAttention(
            embedding_dims, # key dims
            kdim=embedding_dims,
            vdim=embedding_dims,
            num_heads=1,
            batch_first=True,
            device=self.device
        )
        self.attention_mask = torch.eye(
                num_inputs,
                dtype=torch.bool,
                device=self.device)

    def forward(self, inputs):

        keys = inputs
        queries = inputs
        values = inputs
        attention_outputs, attention_weights = self.attention(
                queries,
                keys,
                values,
                attn_mask=self.attention_mask)

        return attention_outputs, attention_weights


class FeedforwardBlock(torch.nn.Module):

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        latent_xdims: int,
        latent_ydims: int,
    ):
        super().__init__()

        self.latent_xdims = latent_xdims
        self.latent_ydims = latent_ydims

        latent_dims = latent_xdims * latent_ydims
        layers = [torch.nn.Linear(input_dims, hidden_dims),
                  torch.nn.Linear(hidden_dims, latent_dims)]

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, inputs):

        return self.mlp(inputs).view(-1, self.latent_ydims, self.latent_xdims)


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
    num_inputs = 6
    window_size = 400
    batch_size = 2
    device = 'cuda:2'
    # (batch, channels, height, width)
    x = torch.rand(batch_size, num_inputs, window_size).to(device)
    model = HybridNet(depth, num_inputs, window_size, device=device) 
    y, attn_weights = model(x)
    print(f'outputs dim: {y.shape}')
