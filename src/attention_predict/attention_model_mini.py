from re import I
import torch
from torch import Tensor
import torch.nn as nn
from .attention_mini import AttentionBlockMini
from .unet import ConvBlock
from typing import Callable, List, Tuple, Optional, Any


# NOTE: Could also use a named tuple, but this provides type annotations for fields
class Channels:
    input: int
    output: int

    def __init__(self, input: int, output: int) -> None:
        self.input = input
        self.output = output


class AttentionModelMini(torch.nn.Module):
    """A U-Net, but with a shared attention layer between any path connecting the
    encoder and decoder (i.e., skip connections for the top levels and between the last
    convolution and upsampling on the lowest level).
    """

    model: nn.Sequential
    skip_connections: List[Tensor]
    input_dims: Tuple[int, int, int]
    num_encoders: int
    num_decoders: int
    num_fmaps: int
    inc_factor: int

    def __init__(
        self,
        depth: int = 3,
        num_neurons: int = 4,
        num_behaviors: int = 8,
        num_fmaps: int = 8,
        attention_scheme: tuple[str, str] = ("neurons", "behaviors"),
        input_dims: tuple[int, int, int] = (1, 4, 1600),
        fmap_inc_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        downsample_factor: int = 2,
        upsample_mode: str = "nearest",
        final_act: torch.nn.Module = nn.Identity(),
    ) -> None:
        super().__init__()
        # Instead of "NfromB" you input ("behaviors", "neurons")
        self.input_dims = input_dims
        N, C, L = input_dims
        scheme_map = {
            "neurons": num_neurons,
            "behaviors": num_behaviors,
            "all": C,
        }
        attn_from, attn_to = attention_scheme
        self.num_encoders = scheme_map[attn_from]
        self.num_decoders = scheme_map[attn_to]
        self.num_fmaps = num_fmaps
        self.inc_factor = fmap_inc_factor

        encoder_features = self.encoder_pass_features(depth)
        decoder_features = self.decoder_pass_features(depth)

        # length = window_size // (2 ** (depth - 1))
        downsampled_length = L // (2 ** (depth))
        last_encoder_features = encoder_features[-1]
        embedding_length = downsampled_length * last_encoder_features.output

        # TODO: have each of these be a non-persistent registered buffer
        # by using interpolated strings for attribute names.
        # Type checking should be fine since they will be in a container.
        self.skip_connections = []
        for layer in range(depth):
            features = encoder_features[layer]
            level_output = torch.zeros(N, C, features.output * L)
            self.skip_connections.append(level_output)

        self.model = nn.Sequential()

        # Build encoder pass
        for level, features in enumerate(encoder_features):
            conv_block = ConvBlock(
                features.input,
                features.output,
                kernel_size,
                padding=padding,
                num_groups=self.num_encoders,
            )
            # Forward pass hook to cache each block's output
            conv_block.register_forward_hook(self.generate_output_hook(level))
            self.model.append(conv_block)
            # Downsample
            self.model.append(nn.MaxPool1d(downsample_factor))

        # Attention block
        self.model.append(
            AttentionBlockMini(
                embedding_length,
                (self.num_encoders, self.num_decoders),  # Attention scheme
                N,
                control_experiment=True,
            )
        )

        # Build decoder pass
        for level, features in enumerate(decoder_features):
            upsample = nn.Upsample(scale_factor=downsample_factor, mode=upsample_mode)
            # Forward pass hook to hcat skip connection tensor with upsampled output
            upsample.register_forward_hook(self.generate_input_hook(level))
            self.model.append(upsample)
            self.model.append(
                ConvBlock(
                    features.input,
                    features.output,
                    kernel_size,
                    padding=padding,
                    num_groups=self.num_decoders,
                )
            )

        # Output convolution
        self.model.append(
            nn.Conv1d(
                # Conv input matched to last [-1] output
                decoder_features[-1].output,
                self.num_decoders,
                1,  # doesnt padding="same" do this automatically????
                padding=0,
                groups=self.num_decoders,
            )
        )
        self.model.append(final_act)

    def forward(self, inputs) -> Tensor:
        return self.model(inputs)  # 😂

    def encoder_fmap(self, level: int) -> Channels:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        fmaps_in = self.num_encoders * (
            1 if level == 0 else self.num_fmaps * self.inc_factor ** (level - 1)
        )
        fmaps_out = self.num_encoders * (self.num_fmaps * self.inc_factor**level)
        return Channels(int(fmaps_in), int(fmaps_out))

    def decoder_fmap(self, level: int) -> Channels:
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
        fmaps_out = self.num_decoders * (self.num_fmaps * self.inc_factor**level)
        concat_fmaps = self.encoder_fmap(level).output
        fmaps_in = self.num_decoders * (
            concat_fmaps + (self.num_fmaps * self.inc_factor ** (level + 1))
        )
        return Channels(int(fmaps_in), int(fmaps_out))

    def encoder_pass_features(self, depth: int) -> list[Channels]:
        return [self.encoder_fmap(level) for level in range(depth)]

    def decoder_pass_features(self, depth: int) -> list[Channels]:
        return [self.decoder_fmap(level) for level in reversed(range(depth - 1))]

    def generate_output_hook(
        self, level: int
    ) -> Callable[[nn.Module, Tuple[Any, ...], Any], Optional[Any]]:
        def hook(module: nn.Module, args: Tuple[Tensor], output: Tensor) -> None:
            # TODO: Check whether this explicit copy, `torch.clone` is necessary
            N, C, _ = self.input_dims
            self.skip_connections[level] = torch.reshape(
                torch.clone(output), (N, C, -1)
            )

        return hook

    def generate_input_hook(
        self, level: int
    ) -> Callable[[nn.Module, Tuple[Any, ...], Any], Optional[Any]]:
        def hook(module: nn.Module, args: Tuple[Tensor], output: Tensor) -> Tensor:
            # FIXME: CHECK DIMS!!
            return torch.cat([output, self.skip_connections[level]], dim=1)

        return hook


if __name__ == "__main__":
    model = AttentionModelMini()
    print("Model initialized...")
