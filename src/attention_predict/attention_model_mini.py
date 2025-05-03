from re import I
import torch
from torch import Tensor
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
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
    embedding_L: int

    def __init__(
        self,
        depth: int = 4,
        num_neurons: int = 5,
        num_behaviors: int = 8,
        num_fmaps: int = 4,
        attention_scheme: tuple[str, str] = ("neurons", "behaviors"),
        input_dims: tuple[int, int, int] = (2, 5, 512),
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
        self.decoder_features = self.decoder_pass_features(depth)

        # NOTE embedding length is invariant across Unet levels
        # as long as: fmap_inc_factor == downsample_factor
        embedding_length = (self.encoder_features[0].output // self.num_encoders) * L
        self.embedding_L = embedding_length
        self.skip_connections = []
        for level in range(depth - 1):
            level_output = torch.zeros(
                N, self.encoder_features[level].output, L // (2**level)
            )
            self.skip_connections.append(level_output)

        # Attention block
        self.attention_block = AttentionBlockMini(
            embedding_length,
            (self.num_encoders, self.num_decoders),  # Attention scheme
            self.input_dims,
        )

        downsample = nn.MaxPool1d(downsample_factor)
        upsample = nn.Upsample(scale_factor=downsample_factor, mode=upsample_mode)

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
            # After each convolution, independently apply and cache the result of
            # attention blocks for later use during the decoder pass before downsampling.
            if level < depth - 1:
                conv_forked_attention = self.add_attention_hook(
                    conv_block, features.output, level
                )
                self.model.append(conv_forked_attention)
                self.model.append(downsample)

            else:  # on the lowest layer...
                # Send attention block output to the subsequent
                # decoder pass (without forking) and do not downsample the result.
                self.model.append(conv_block)
                self.model.append(self.attention_block)
                self.model.append(
                    Rearrange(
                        "N n_enc (feats L) -> N (n_enc feats) L",
                        feats=features.output // self.num_encoders,
                        n_enc=self.num_encoders,
                    )
                )

        # Continue sequential/chain with decoder pass
        for level in reversed(range(1, depth)):
            # Hook to concatenate skip connection with upsampled input
            upsample_concat = self.add_concat_hook(upsample, level)
            self.model.append(upsample_concat)
            # Decoder block
            features = self.decoder_features[level - 1]
            self.model.append(
                ConvBlock(
                    features.input,
                    features.output,
                    kernel_size,
                    padding=padding,
                    num_groups=self.num_encoders,
                )
            )
        ## Output convolution
        ## TODO: We may have to vmap and use independent convolutional blocks here since
        ## input + output channels might not be an integer multiple of groups=num_decoders
        # self.model.append(
        #    nn.Conv1d(
        #        # Conv input matched to last [-1] output
        #        self.decoder_features[-1].output,
        #        self.num_decoders,
        #        1,
        #        padding=0,
        #        groups=self.num_decoders,  # might not work in this particular case
        #    )
        # )
        # self.model.append(final_act)

    def forward(self, inputs) -> Tensor:
        return self.model(inputs)  # 😂

    def encoder_fmap(self, level: int) -> Channels:
        fmaps_in = (
            self.num_encoders
            if level == 0
            else self.num_encoders * (self.num_fmaps * self.inc_factor ** (level - 1))
        )
        fmaps_out = self.num_encoders * (self.num_fmaps * self.inc_factor**level)
        return Channels(int(fmaps_in), int(fmaps_out))

    def decoder_fmap(self, level: int) -> Channels:
        feats_out = self.encoder_fmap(level).output
        prev_level_out = self.encoder_fmap(level + 1).output
        feats_in = feats_out + prev_level_out
        return Channels(int(feats_in), int(feats_out))

    def encoder_pass_features(self, depth: int) -> List[Channels]:
        return [self.encoder_fmap(level) for level in range(depth)]

    def decoder_pass_features(self, depth: int) -> List[Channels]:
        return [self.decoder_fmap(level) for level in range(depth - 1)]

    def add_attention_hook(
        self, conv_block: ConvBlock, features_out: int, level: int
    ) -> ConvBlock:
        # TODO: Add more value assignments to the Rearrange layer
        # It is not strictly necessary, but provides an implicit runtime assertion

        def cache_attention(
            module: nn.Module, args: Tuple[Tensor], output: Tensor
        ) -> None:
            attn_input = einops.rearrange(
                output,
                "N (n_enc feats) L -> N n_enc (feats L)",
                n_enc=self.num_encoders,
            )
            attn_out = self.attention_block(attn_input)
            assert attn_out.shape == attn_input.shape, (
                "Output was shape {aout}, but expected shape was {ain}".format(
                    aout=attn_out.shape, ain=attn_input.shape
                )
            )
            reshaped = einops.rearrange(
                attn_out,
                "N n_enc (feats L) -> N (n_enc feats) L",
                feats=features_out // self.num_encoders,
                n_enc=self.num_encoders,
            )
            self.skip_connections[level] = reshaped

        conv_block.register_forward_hook(cache_attention)
        return conv_block

    def add_concat_hook(self, layer: nn.Upsample, level: int) -> nn.Upsample:
        def cat_inputs(
            module: nn.Module, args: Tuple[Tensor], upsampled: Tensor
        ) -> Tensor:
            next_level = level - 1
            next_level_input = self.skip_connections[next_level]
            concatenated_input = torch.cat([next_level_input, upsampled], 1)
            return concatenated_input

        layer.register_forward_hook(cat_inputs)
        return layer


if __name__ == "__main__":
    model = AttentionModelMini()
    print("Model initialized...")
