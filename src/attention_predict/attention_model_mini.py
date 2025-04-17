import torch
import torch.nn as nn
from .attention_mini import AttentionBlockMini
from .unet import ConvBlock


class AttentionModelMini(torch.nn.Module):
    """A U-Net, but with a shared attention layer between any path connecting the
    encoder and decoder (i.e., skip connections for the top levels and between the last
    convolution and upsampling on the lowest level).
    """

    def __init__(
        self,
        depth: int,
        num_neurons: int,
        num_behaviors: int,
        input_dims: tuple[int, int, int],
        attention_scheme: tuple[str, str],
        window_size: int,
        num_fmaps: int = 8,
        fmap_inc_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        downsample_factor: int = 2,
        upsample_mode: str = "nearest",
        final_act: torch.nn.Module = nn.Identity(),
    ):
        super().__init__()

        N, C, L = input_dims

        self.kernel_size = kernel_size
        self.padding = padding
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor

        scheme_map = {
            "neurons": num_neurons,
            "behaviors": num_behaviors,
            "all": C,
        }

        attn_from, attn_to = attention_scheme
        self.num_encoders = scheme_map[attn_from]
        self.num_decoders = scheme_map[attn_to]

        self.length = self.window_size // (2 ** (self.depth - 1))
        encoder_features = self.encoder_pass_features(depth)
        decoder_features = self.decoder_pass_features(depth)
        embedding_dims = self.length * encoder_features[-1][1]

        # TODO: initialize with torch.zeros and correct dims
        self.register_buffer(
            "skip_connections", torch.nested.nested_tensor([...], layout=torch.jagged)
        )

        self.model = nn.Sequential()

        # Build encoder pass
        for level, (in_channels, out_channels) in enumerate(encoder_features):
            self.model.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    self.kernel_size,
                    self.padding,
                    self.num_encoders,
                )
            )
            # Forward pass hook to cache each block's output
            self.model[-1].register_forward_pass_hook(self.generate_output_hook(level))
            # Downsample
            self.model.append(nn.MaxPool1d(downsample_factor))

        # Attention block
        self.model.append(
            AttentionBlockMini(
                embedding_dims,
                (attn_from, attn_to),  # Attention scheme
                N,
                control_experiment=True,
            )
        )

        # Build decoder pass
        for level, (in_channels, out_channels) in enumerate(decoder_features):
            self.model.append(
                nn.Upsample(scale_factor=downsample_factor, mode=upsample_mode)
            )
            # Forward pass hook to hcat skip connection tensor with upsampled output
            self.model[-1].register_forward_pass_hook(self.generate_input_hook(level))
            self.model.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    self.kernel_size,
                    self.padding,
                    self.num_decoders,
                )
            )

        # Output convolution
        self.model.append(
            nn.Conv1d(
                decoder_features[-1][1],
                self.num_decoders,
                1,
                padding=0,
                groups=self.num_decoders,
            )
        )
        self.model.append(final_act)

    def forward(self, inputs):
        return self.model(inputs)  # 😂

    def encoder_fmap(self, level: int) -> tuple[int, int]:
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
            fmaps_in = self.num_encoders
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def decoder_fmap(self, level: int) -> tuple[int, int]:
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
        concat_fmaps = self.encoder_fmap(level)[1]
        # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)
        return fmaps_in, fmaps_out

    def encoder_pass_features(self, depth: int):
        return [self.encoder_fmap(level) for level in range(depth)]

    def decoder_pass_features(self, depth: int):
        return [self.decoder_fmap(level) for level in reversed(range(depth - 1))]

    def generate_output_hook(self, level):
        def hook(module, args, output):
            self.skip_connections[level] = output

        return hook

    def generate_input_hook(self, level):
        def hook(module, args, output):
            return torch.hcat(output, self.skip_connections[level])

        return hook
