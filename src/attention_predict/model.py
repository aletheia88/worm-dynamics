from typing import List, Tuple

import torch
import torch.nn as nn

# from einops.layers.torch import Rearrange
from torch import Tensor

from .components import (  # , EncoderBlock, DecoderBlock
    AttentionBlock,
    ConvBlock,
    ConvOutSkip,
    ConvSkip,
    DecoderSkip,
    DownsampleSkip,
    ReshapeSkip,
)


class Channels:
    input: int
    output: int

    def __init__(self, input: int, output: int) -> None:
        self.input = input
        self.output = output


class MiniAttentionModel(torch.nn.Module):
    """A U-Net with an attention layer between any path connecting encoders and decoders
    (i.e. skip connections for each level and the bottleneck on the lowest level).
    """

    def __init__(
        self,
        # TODO: we should parse the first 4 arguments from a single data structure
        num_decoders: int = 8,
        input_dims: Tuple[int, int, int] = (2, 5, 512),
        depth: int = 4,
        features_basis: int = 4,
        final_activation: nn.Module = nn.Identity(),
    ) -> None:
        super().__init__()

        batch_size, num_encoders, window_length = input_dims
        scale_factor: int = 2

        # Number of input/output channels per convolutional block
        encoder_features = self.encoder_pass_features(
            depth, num_encoders, scale_factor, features_basis
        )
        decoder_features = self.decoder_pass_features(
            depth, num_decoders, scale_factor, features_basis
        )

        # NOTE embedding length is invariant across Unet levels
        E_v = (encoder_features[0].output // num_encoders) * window_length

        model = nn.Sequential()
        # Attention block
        self.attention_block = AttentionBlock(
            E_v,
            num_encoders,
            num_decoders,
            batch_size,
            depth,
        )

        downsample = DownsampleSkip(scale_factor)

        # Encoder pass blocks
        for level, features in enumerate(encoder_features):
            attention_out_features = decoder_features[level].output // num_decoders
            model.append(ConvSkip(features.input, features.output, num_encoders))
            model.append(self.attention_block)
            model.append(ReshapeSkip(attention_out_features))
            if level < (depth - 1):
                model.append(downsample)

        # Decoder pass blocks
        for level in reversed(range(depth - 1)):
            features = decoder_features[level]
            model.append(
                DecoderSkip(
                    ConvBlock(features.input, features.output, num_groups=num_decoders),
                    scale_factor,
                )
            )

        # Output convolution
        model.append(
            ConvOutSkip(
                nn.Conv1d(
                    decoder_features[0].output,
                    num_decoders,
                    1,
                    padding=0,
                    groups=num_decoders,  # might not work in this particular case
                )
            )
        )
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        skip_connections: List[Tensor] = []
        x = self.model((x, skip_connections))
        ## Encoder pass
        # x, skip0 = self.encoders[0](x)
        # x, skip1 = self.encoders[1](x)
        # x, skip2 = self.encoders[2](x)
        # _, x = self.encoders[3](x)

        ## Decoder pass
        # x = self.decoders[0](x, skip2)
        # x = self.decoders[1](x, skip1)
        # x = self.decoders[2](x, skip0)

        ## Output convolution
        # x = self.conv_out(x)
        return x

    def feature_map(self, level: int, scale_factor: int, features_basis: int) -> Tensor:
        fmaps_in = 1 if level == 0 else features_basis * scale_factor ** (level - 1)
        fmaps_out = features_basis * scale_factor**level
        return torch.tensor([fmaps_in, fmaps_out])

    def encoder_pass_features(
        self, depth: int, num_encoders: int, scale_factor: int, features_basis: int
    ) -> List[Channels]:
        # Scale features per level by the number of encoders per level
        features: List[Channels] = []
        for level in range(depth):
            feats_in_out: Tensor = (
                self.feature_map(level, scale_factor, features_basis) * num_encoders
            )
            features.append(Channels(*(feats_in_out)))
        return features

    def decoder_pass_features(
        self, depth: int, num_decoders: int, scale_factor: int, features_basis: int
    ) -> List[Channels]:
        # Scale features per level by the number of decoders per level and recursively
        # concatenate outputs to inputs
        features: List[Channels] = []
        for level in range(depth):
            feats_out: int = num_decoders * int(
                self.feature_map(level, scale_factor, features_basis)[1]
            )
            prev_level_out: int = num_decoders * int(
                self.feature_map(level + 1, scale_factor, features_basis)[1]
            )
            feats_in: int = 0 if level == (depth - 1) else feats_out + prev_level_out
            features.append(Channels(feats_in, feats_out))

        return features
