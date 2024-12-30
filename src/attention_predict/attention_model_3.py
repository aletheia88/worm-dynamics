import torch
from .attention import AttentionBlock
from .unet import Downsample, ConvBlock, OutputConv


class AttentionModel3(torch.nn.Module):
    """A U-Net, but with independent attention layers between any path connecting the
    encoder and decoder (i.e., skip connections for the top levels and between the last
    convolution and upsampling on the lowest level).
    """

    def __init__(
        self,
        depth: int,
        num_neurons: int,
        num_behaviors: int,
        window_size: int,
        num_fmaps: int = 8,
        fmap_inc_factor: int = 2,
        kernel_size: int = 3,
        padding: str = 'same',
        downsample_factor: int = 2,
        upsample_mode: str = 'nearest',
        final_activation: torch.nn.Module | None = None,
        attention_scheme: str = 'all',
        device: str = 'cpu',
    ):
        super().__init__()

        self.depth = depth
        self.num_neurons = num_neurons
        self.num_inputs = num_neurons + num_behaviors
        self.window_size = window_size
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.downsample_factor = downsample_factor
        self.upsample_mode = upsample_mode
        self.final_activation = final_activation
        self.device = device

        self.downsample = Downsample(self.downsample_factor, ndim=1)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode)

        self.encoder_block = torch.nn.ModuleList()
        self.attention_block = torch.nn.ModuleList()
        self.decoder_block = torch.nn.ModuleList()
        self.final_convs = torch.nn.ModuleList()

        for _ in range(self.num_inputs):

            encoder = torch.nn.ModuleList()

            for level in range(self.depth):

                fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
                encoder.append(
                    ConvBlock(
                        fmaps_in,
                        fmaps_out,
                        self.kernel_size,
                        self.padding,
                        ndim=1
                    ).to(self.device)
                )
            self.encoder_block.append(encoder)

        self.length = self.window_size // (2**(self.depth - 1))
        self.channels = fmaps_out

        embedding_dims = self.length * self.channels

        for _ in range(self.depth):

            self.attention_block.append(
                AttentionBlock(
                    embedding_dims,
                    num_neurons,
                    num_behaviors,
                    attention_scheme,
                    device=self.device,
                )
            )

        for _ in range(self.num_inputs):

            decoder = torch.nn.ModuleList()

            for level in range(self.depth - 1):

                fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
                decoder.append(
                    ConvBlock(
                        fmaps_in,
                        fmaps_out,
                        self.kernel_size,
                        self.padding,
                        ndim=1
                    ).to(self.device)
                )
            self.decoder_block.append(decoder)

            self.final_convs.append(
                OutputConv(
                    in_channels=self.compute_fmaps_decoder(0)[1],
                    out_channels=1,
                    activation=self.final_activation,
                    ndim=1
                ).to(self.device)
            )

    def forward(self, inputs):

        num_samples = inputs.shape[0]
        # outputs per level and variable
        level_outputs = {
            level: [] for level in range(self.depth)
        }

        ### encoder block ###
        for i in range(self.num_inputs):

            layer_input = inputs[:, i, :].unsqueeze(1)

            # get the encoder for variable i
            encoder = self.encoder_block[i]

            # go down the levels of the U-Net
            for level in range(self.depth - 1):
                conv_out = encoder[level](layer_input)
                level_outputs[level].append(conv_out.view(num_samples, 1, -1))
                downsampled = self.downsample(conv_out)
                layer_input = downsampled

            # lowest level of the U-Net
            conv_out = encoder[-1](layer_input)
            level_outputs[self.depth - 1].append(conv_out.view(num_samples, 1, -1))

        # level_outputs[level][i]: (num_samples, 1, embedding_dims)

        # concatenate level outputs of all variables
        for level in range(self.depth):
            level_outputs[level] = torch.concatenate(level_outputs[level], axis=1)

        ### attention block ###
        # attention_weights: (num_samples, num_inputs, num_inputs)
        # attention_outputs: (num_samples, num_inputs, embedding_dims)
        level_attention_outputs = []
        level_attention_weights = []

        for level in range(self.depth):
            attention_outputs, attention_weights = self.attention_block[level](level_outputs[level])
            level_attention_outputs.append(attention_outputs)
            level_attention_weights.append(attention_weights)

        # level_attention_outputs[level]: (num_samples, num_inputs, embedding_dims(level))

        ### decoder block ###
        decoded_outputs = []
        for i in range(self.num_inputs):

            decoder = self.decoder_block[i]

            # go up the levels of the U-Net
            for level in range(0, self.depth)[::-1]:

                _, fmaps_out = self.compute_fmaps_encoder(level)
                level_input = level_attention_outputs[level][:, i, :].unsqueeze(1)
                # level_input: (num_samples, 1, embedding_dims(level))
                # if lowest level
                if level == self.depth - 1:
                    level_input = level_input.view(num_samples, self.compute_fmaps_encoder(level)[1], -1)
                    prev_level_output = level_input
                    continue

                # reshape into (..., channels, length)
                level_input = level_input.view(num_samples, fmaps_out, -1)
                upsampled = self.upsample(prev_level_output)
                # concatenate channels
                decoder_input = torch.concatenate([level_input, upsampled], axis=1)
                prev_level_output = decoder[level](decoder_input)

            final_output = self.final_convs[i](prev_level_output)
            decoded_outputs.append(final_output)

        model_outputs = torch.cat(decoded_outputs, 1).to(self.device)

        return model_outputs, level_attention_weights

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
            fmaps_in = 1
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
