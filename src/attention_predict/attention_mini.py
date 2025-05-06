import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor


class AttentionBlock(nn.Module):
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


# class Downsample(nn.Module):
#     downsample: nn.Module
#
#     def __init__(self, scale_factor: int) -> None:
#         super().__init__()
#         self.downsample = nn.MaxPool1d(scale_factor)
#
#     def forward(self, inputs: Tuple[Tensor, List[Any]]) -> Tuple[Tensor, List[Any]]:
#         input, skip_connections = inputs
#         out: Tensor = self.downsample(input)
#         return (out, skip_connections)
#
#
# class UpsampleConcat(nn.Module):
#     upsample: nn.Upsample
#     reshape: Rearrange
#
#     def __init__(self, scale_factor: int, num_features: int) -> None:
#         super().__init__()
#         self.reshape = Rearrange(
#             "N n_dec (feats L) -> N (n_dec feats) L", feats=num_features
#         )  # decoder_features[level].output // num_decoders,
#         self.upsample = nn.Upsample(scale_factor)  # default mode = "nearest"
#
#     def forward(self, inputs: Tuple[Tensor, List[Any]]) -> Tuple[Tensor, List[Any]]:
#         input, skip_connections = inputs
#         upsampled: Tensor = self.upsample(input)
#         skip_out: Tensor = self.reshape(skip_connections.pop())
#         assert False, "Up: {ups}, Skip: {skip}".format(
#             ups=upsampled.shape, skip=skip_out.shape
#         )
#         out: Tensor = torch.cat([skip_out, upsampled], 1)
#         return out, skip_connections
