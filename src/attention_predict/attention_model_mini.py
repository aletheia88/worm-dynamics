import torch
from torch import Tensor
import torch.nn as nn
import einops
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
    encoder_features: List[Channels]
    decoder_features: List[Channels]
    depth: int

    def __init__(
        self,
        depth: int = 3,
        num_neurons: int = 4,
        num_behaviors: int = 8,
        num_fmaps: int = 4,
        attention_scheme: tuple[str, str] = ("neurons", "behaviors"),
        input_dims: tuple[int, int, int] = (2, 4, 512),
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
        self.depth = depth
        self.encoder_features = self.encoder_pass_features(depth)
        self.decoder_features = self.decoder_pass_features(depth - 1)

        downsampled_length = L // (2 ** (depth - 1))
        embedding_length = downsampled_length * (
            self.encoder_features[-1].output // self.num_encoders
        )

        self.skip_connections = []
        for layer in range(depth):
            channels_out = self.encoder_features[layer].output
            level_output = torch.zeros(N, channels_out, L // 2**layer)
            self.skip_connections.append(level_output)

        # Attention block
        self.attention_block = AttentionBlockMini(
            embedding_length,
            (self.num_encoders, self.num_decoders),  # Attention scheme
            self.input_dims,
        )

        self.model = nn.Sequential()

        # Build encoder pass
        for level, features in enumerate(self.encoder_features):
            conv_block = ConvBlock(
                features.input,
                features.output,
                kernel_size,
                padding=padding,
                num_groups=self.num_encoders,
            )
            # Forward pass hook to apply and cache attention block output
            # If lowest level, return attention block output instead of caching
            conv_block.register_forward_hook(self.generate_output_hook(level))
            self.model.append(conv_block)

            # Don't downsample after last convolution...
            if level < (depth - 1):
                self.model.append(nn.MaxPool1d(downsample_factor))

        # Build decoder pass--the first input is attention_block(lowest_encoder_convblock)
        # so we jump immediately to (last_layer - 1) working from lowest to highest level
        for level, features in enumerate(self.decoder_features):
            upsample = nn.Upsample(scale_factor=downsample_factor, mode=upsample_mode)
            # Forward pass hook to hcat upsampled result with attention_block output at current level
            upsample.register_forward_hook(self.generate_input_hook(level))
            self.model.append(upsample)
            # Finally, feed concatenated result to next conv block
            self.model.append(
                ConvBlock(
                    features.input,
                    features.output,
                    kernel_size,
                    padding=padding,
                    num_groups=self.num_encoders,
                )
            )
        # Output convolution
        # TODO: We may have to vmap and use independent convolutional blocks here since
        # input + output channels might not be an integer multiple of groups=num_decoders
        self.model.append(
            nn.Conv1d(
                # Conv input matched to last [-1] output
                self.encoder_features[-1].output,
                self.num_decoders,
                1,
                padding=0,
                groups=self.num_decoders,  # might not work in this particular case
            )
        )
        self.model.append(final_act)

    def forward(self, inputs) -> Tensor:
        return self.model(inputs)  # 😂

    def encoder_fmap(self, level: int) -> Channels:
        fmaps_in = self.num_encoders * (
            1 if level == 0 else self.num_fmaps * self.inc_factor ** (level - 1)
        )
        fmaps_out = self.num_encoders * (self.num_fmaps * self.inc_factor**level)
        return Channels(int(fmaps_in), int(fmaps_out))

    def decoder_fmap(self, level: int) -> Channels:
        # NOTE: This could be simplified by just referencing previously
        # calculated encoder pass features...probably a recursive pattern
        # AFAIK the number of decoders doesn't matter until the very last
        # "output" convolutional block.
        fmaps_out = self.num_encoders * (self.num_fmaps * self.inc_factor**level)
        concat_fmaps = self.encoder_fmap(level).output
        fmaps_in = self.num_encoders * (
            concat_fmaps + (self.num_fmaps * self.inc_factor ** (level + 1))
        )
        return Channels(int(fmaps_in), int(fmaps_out))

    def encoder_pass_features(self, depth: int) -> list[Channels]:
        return [self.encoder_fmap(level) for level in range(depth)]

    def decoder_pass_features(self, depth: int) -> list[Channels]:
        return [self.decoder_fmap(level) for level in reversed(range(1, depth))]

    def generate_output_hook(
        self, level: int
    ) -> Callable[[nn.Module, Tuple[Any, ...], Any], Optional[Any]]:
        # If we are on the lowest level we will return: attention_block(conv_block_output)
        if level == (self.depth - 1):
            # TODO: potentially use einops layers and make + execute a nn.Sequential
            def attention_out(
                _module: nn.Module, _args: Tuple[Tensor], output: Tensor
            ) -> Tensor:
                V = einops.rearrange(
                    torch.clone(output),
                    "N (c_out enc) L -> N enc (c_out L)",
                    enc=self.num_encoders,
                )

                return self.attention_block(V)

            return attention_out
        # Otherwise conv output is unmodified and we _cache_ attention block output for later
        else:

            def cache_attention(
                module: nn.Module, args: Tuple[Tensor], output: Tensor
            ) -> None:
                V = einops.rearrange(
                    torch.clone(output),
                    "N (c_out enc) L -> N enc (c_out L)",
                    enc=self.num_encoders,
                )

                self.skip_connections[level] = self.attention_block(V)

            return cache_attention

    def generate_input_hook(
        self, level: int
    ) -> Callable[[nn.Module, Tuple[Any, ...], Any], Optional[Any]]:
        def hook(module: nn.Module, args: Tuple[Tensor], output: Tensor) -> Tensor:
            # Here the "output" argument is always the upsampled attention block output
            # from the _previous_ level
            N, C, _ = output.shape
            # Reshape attention block output from _current_ level and concatenate with previous
            # This will become input of next convolutional decoder block
            skip_out = self.skip_connections[level]
            skip_out = torch.reshape(skip_out, (N, C, -1))
            return torch.cat([output, skip_out], dim=1)

        return hook


if __name__ == "__main__":
    model = AttentionModelMini()
    print("Model initialized...")
