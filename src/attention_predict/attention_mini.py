import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops
from einops.layers.torch import Rearrange
from typing import Callable, Tuple


# TODO: Check whether we need modify dims, accounting for number of channels since we are using
# groups/depth-first convolutions now.
class AttentionBlockMini(nn.Module):
    K: Tensor
    Q: Tensor
    K_weights: nn.Sequential
    Q_weights: nn.Sequential
    V_weights: nn.Sequential
    batch_size: int
    embedding_L: int
    attention_dims: Tuple[int, int]
    sqrt_dk: int

    def __init__(
        self,
        embedding_L: int,
        attention_dims: Tuple[int, int],
        value_dims: Tuple[int, int, int],
    ) -> None:
        super().__init__()

        N, _, _ = value_dims
        self.batch_size = N
        self.attention_dims = attention_dims
        source_L, target_L = attention_dims
        self.embedding_L = embedding_L
        self.sqrt_dk = target_L**0.5

        # Register buffers so they inherit device of parent module.
        # K and Q are invariant for each forward pass and K == Q
        self.register_buffer(
            "K",
            einops.repeat(torch.eye(source_L), "H W -> N H W", N=self.batch_size),
            persistent=False,
        )
        self.register_buffer(
            "Q",
            einops.repeat(torch.eye(source_L), "H W -> N H W", N=self.batch_size),
            persistent=False,
        )
        # TODO: make registered persistent buffers for K & Q to cache intermediate
        # projected values to the state_dict. This will allow recomputing the attention matrix
        # add_head_dim = Rearrange("N H W -> N 1 H W")
        self.Q_weights = nn.Sequential(
            nn.Linear(source_L, source_L, bias=False),
            nn.Linear(source_L, source_L, bias=False),
            # add_head_dim,
        )
        self.K_weights = nn.Sequential(
            nn.Linear(source_L, source_L, bias=False),
            nn.Linear(source_L, target_L, bias=False),
            # add_head_dim,
        )
        self.V_weights = nn.Sequential(
            nn.Linear(embedding_L, embedding_L, bias=False),
            # add_head_dim,
        )
        # self.remove_head_dim = Rearrange("N 1 H W -> N H W")

    def forward(self, inputs: Tensor) -> Tensor | Tuple[Tensor, Tensor]:
        Q_weighted = self.Q_weights(self.Q)
        K_weighted = self.K_weights(self.K)
        V_weighted = self.V_weights(inputs)
        # out = F.scaled_dot_product_attention(Q_weighted, K_weighted, V_weighted)
        attention_matrix = Q_weighted @ K_weighted / self.sqrt_dk
        # out = self.remove_head_dim(out)
        attention_matrix = F.softmax(attention_matrix, dim=-1)
        assert attention_matrix.shape[1] == 5, (
            "Attention matrix shape is: {dims}".format(dims=attention_matrix.shape)
        )
        out = attention_matrix @ V_weighted

        return out
