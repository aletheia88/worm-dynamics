from typing import List, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor


class AttentionBase(nn.Module):
    QK: Tensor
    K_proj: nn.Sequential
    Q_proj: nn.Sequential
    V_proj: nn.Sequential
    depth: int

    def __init__(
        self, E_v: int, L_source: int, L_target: int, batch_size: int, depth: int
    ) -> None:
        super().__init__()
        self.depth = depth
        E_qk: int = L_source

        # Register buffers so they inherit device of parent module.
        # Q and K are invariant for each forward pass and Q == K
        self.register_buffer(
            "QK",
            einops.repeat(torch.eye(E_qk), "H W -> N H W", N=batch_size),
            persistent=False,
        )

        add_head_dim = Rearrange("N H W -> N 1 H W")
        self.remove_head_dim = Rearrange("N 1 H W -> N H W")

        self.Q_proj = nn.Sequential(
            nn.Linear(L_source, L_source, bias=False),
            nn.Linear(L_source, L_target, bias=False),
            # FIXME: Check with Alicia whether this transpose makes sense
            Rearrange("N H W -> N 1 W H"),
        )
        self.K_proj = nn.Sequential(
            nn.Linear(L_source, L_source, bias=False),
            nn.Linear(L_source, L_source, bias=False),
            add_head_dim,
        )
        self.V_proj = nn.Sequential(
            Rearrange(
                "N (n_enc feats) L -> N n_enc (feats L)", n_enc=L_source, N=batch_size
            ),
            nn.Linear(E_v, E_v, bias=False),
            add_head_dim,
        )

    def forward(self, input: Tensor) -> Tensor:
        # input dims: (N, n_encoders*channels, window_size // 2 ** level-1)
        query = key = self.QK
        Q: Tensor = self.Q_proj(query)
        K: Tensor = self.K_proj(key)
        V: Tensor = self.V_proj(input)

        out: Tensor = F.scaled_dot_product_attention(Q, K, V)
        out = self.remove_head_dim(out)  # output dims: (N, L_target, E_v)
        assert out.shape[-1] == 2048

        return out


class AttentionBlock(nn.Module):
    def __init__(self, *args: int) -> None:
        super().__init__()
        self.attention = AttentionBase(*args)

    def forward(self, inputs: Tuple[Tensor, List[Tensor]]) -> Tuple[Tensor, List[Tensor]]:
        x, skip_connections = inputs
        skip_connections.append(self.attention(x))
        return x, skip_connections


class ConvBlock(nn.Module):
    double_conv: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        kernel_size: int = 3,
        padding: str | int = "same",
    ) -> None:
        super().__init__()

        conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            groups=num_groups,
        )
        conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            groups=num_groups,
        )
        # Parameter initializations
        nn.init.kaiming_normal_(conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(conv2.weight, nonlinearity="relu")

        self.double_conv = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        out: Tensor = self.double_conv(input)
        return out


class ReshapeSkip(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.reshape = Rearrange(
            "N n_dec (feats L) -> N (n_dec feats) L",
            feats=features,
        )

    def forward(self, inputs: Tuple[Tensor, List[Tensor]]) -> Tuple[Tensor, List[Tensor]]:
        x, skip_connections = inputs
        skip_connections[-1] = self.reshape(skip_connections[-1])
        return x, skip_connections


class ConvSkip(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        kernel_size: int = 3,
        padding: str | int = "same",
    ) -> None:
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels, out_channels, num_groups, kernel_size, padding
        )
        return

    def forward(self, inputs: Tuple[Tensor, List[Tensor]]) -> Tuple[Tensor, List[Tensor]]:
        x, skip_connections = inputs
        x = self.conv_block(x)
        return x, skip_connections


class DownsampleSkip(nn.Module):
    def __init__(self, scale_factor: int) -> None:
        super().__init__()
        self.downsample = nn.MaxPool1d(scale_factor)

    def forward(self, inputs: Tuple[Tensor, List[Tensor]]) -> Tuple[Tensor, List[Tensor]]:
        x, skip_connections = inputs
        x = self.downsample(x)
        return x, skip_connections


class EncoderBlock(nn.Module):
    def __init__(
        self,
        attention_block: AttentionBlock,
        encoder: ConvBlock,
        reshape: Rearrange,
        downsample: nn.Module = nn.Identity(),
    ) -> None:
        super().__init__()
        self.attention_block = nn.Sequential(attention_block, reshape)
        self.encoder = encoder
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        skip: Tensor = self.attention_block(x)
        x = self.downsample(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, decoder: ConvBlock, scale_factor: int):
        super().__init__()
        self.decoder = decoder
        self.upsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.decoder(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, decoder: ConvBlock, scale_factor: int) -> None:
        super().__init__()
        self.decoder = DecoderBlock(decoder, scale_factor)

    def forward(self, inputs: Tuple[Tensor, List[Tensor]]) -> Tuple[Tensor, List[Tensor]]:
        _, skip_connections = inputs
        x = skip_connections.pop()
        skip = skip_connections.pop()
        x = self.decoder(x, skip)
        return x, skip_connections


class DecoderSkip(nn.Module):
    def __init__(self, decoder: ConvBlock, scale_factor: int) -> None:
        super().__init__()
        self.decoder = DecoderBlock(decoder, scale_factor)

    def forward(self, inputs: Tuple[Tensor, List[Tensor]]) -> Tuple[Tensor, List[Tensor]]:
        x, skip_connections = inputs
        skip = skip_connections.pop()
        x = self.decoder(x, skip)
        return x, skip_connections


class ConvOutSkip(nn.Module):
    def __init__(self, conv_out: nn.Conv1d) -> None:
        super().__init__()
        self.conv_out = conv_out

    def forward(self, inputs: Tuple[Tensor, List[Tensor]]) -> Tensor:
        x, _ = inputs
        x = self.conv_out(x)
        return x
