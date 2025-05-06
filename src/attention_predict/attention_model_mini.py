from typing import List, Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from .attention_mini import AttentionBlock, ConvBlock


class Channels:
    input: int
    output: int

    def __init__(self, input: int, output: int) -> None:
        self.input = input
        self.output = output


class AttentionModelMini(torch.nn.Module):
    """A U-Net with an attention layer between any path connecting encoders and decoders
    (i.e. skip connections for each level and the bottleneck on the lowest level).
    """

    def __init__(
        self,
        # TODO: we should parse the first 4 arguments from a single data structure
        num_decoders: int = 8,
        input_dims: Tuple[int, int, int] = (2, 5, 512),
        depth: int = 4,
        features_basis: int = 8,
        final_activation: nn.Module = nn.Identity(),
    ) -> None:
        super().__init__()
        batch_size, num_encoders, window_length = input_dims
        scale_factor: int = 2
        encoder_features = self.encoder_pass_features(
            depth, num_encoders, scale_factor, features_basis
        )
        decoder_features = self.decoder_pass_features(
            depth, num_decoders, scale_factor, features_basis
        )

        # NOTE embedding length is invariant across Unet levels
        E_v = (encoder_features[0].output // num_encoders) * window_length

        # Attention block
        self.attention_block = AttentionBlock(
            E_v,
            num_encoders,
            num_decoders,
            batch_size,
            depth,  # Attention scheme
        )

        self.downsample = nn.MaxPool1d(scale_factor)
        self.upsample = nn.Upsample(scale_factor=scale_factor)

        # Encoder pass blocks
        encoders = nn.ModuleList()
        for level, features in enumerate(encoder_features):
            encoders.append(
                ConvBlock(features.input, features.output, num_groups=num_encoders)
            )

        # Attention block output reshaping
        reshapes = nn.ModuleList()
        for features in decoder_features:
            reshapes.append(
                Rearrange(
                    "N n_dec (feats L) -> N (n_dec feats) L",
                    feats=(features.output // num_decoders),
                )
            )

        # Decoder pass blocks
        decoders = nn.ModuleList()
        for level in reversed(range(depth - 1)):
            features = decoder_features[level]
            decoders.append(
                ConvBlock(features.input, features.output, num_groups=num_decoders)
            )

        self.encoders = encoders
        self.decoders = decoders
        self.reshapes = reshapes
        # Output convolution
        self.conv_out = nn.Conv1d(
            # Conv input matched to last [-1] output
            decoder_features[0].output,
            num_decoders,
            1,
            padding=0,
            groups=num_decoders,  # might not work in this particular case
        )

    def forward(self, input: Tensor) -> Tensor:
        # Encoder pass
        # Level 0
        x = self.encoders[0](input)
        skip0 = self.attention_block(x)
        skip0 = self.reshapes[0](skip0)
        x = self.downsample(x)

        # Level 1
        x = self.encoders[1](x)
        skip1 = self.attention_block(x)
        skip1 = self.reshapes[1](skip1)
        x = self.downsample(x)

        # Level 2
        x = self.encoders[2](x)
        skip2 = self.attention_block(x)
        skip2 = self.reshapes[2](skip2)
        x = self.downsample(x)

        # Level 3 (bottom)
        x = self.encoders[3](x)
        skip3 = self.attention_block(x)
        skip3 = self.reshapes[3](skip3)

        # Decoder pass
        # Level 2
        x = self.upsample(skip3)
        x = torch.cat([x, skip2], dim=1)

        x = self.decoders[0](x)

        x = self.upsample(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoders[1](x)

        x = self.upsample(x)
        x = torch.cat([x, skip0], dim=1)
        x = self.decoders[2](x)

        x = self.conv_out(x)
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
