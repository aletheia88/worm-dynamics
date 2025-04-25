from re import I
import torch
import torch.nn as nn
import functools
from torch import Tensor
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    _score_mod_signature,
)
import einops
from typing import Callable, Tuple


# TODO: Check whether we need modify dims, accounting for number of channels since we are using
# groups/depth-first convolutions now.
class AttentionBlockMini(nn.Module):
    K: Tensor
    Q: Tensor
    attention_matrix: Tensor
    K_weights: nn.Sequential
    Q_weights: nn.Sequential
    V_weights: nn.Linear
    spda: Callable[[Tensor, Tensor, Tensor], Tensor | Tuple[Tensor, Tensor]]
    batch_size: int
    embedding_L: int
    attention_dims: Tuple[int, int]

    def __init__(
        self,
        embedding_L: int,
        attention_dims: Tuple[int, int],
        value_dims: Tuple[int, int, int],
        self_attention: bool = False,
        control_experiment: bool = True,
    ) -> None:
        super().__init__()

        N, C, L = value_dims
        self.batch_size = N
        self.attention_dims = attention_dims
        source_L, target_L = attention_dims
        self.embedding_L = embedding_L
        # mask_mod: returns False for indices of the attention mask that should be skipped
        # score_mod: makes a copy of the attention_matrix and returns the unmodified matrix
        if self_attention & (not control_experiment):
            mask_mod = self.generate_mask_mod()
            block_mask = create_block_mask(
                mask_mod, None, None, source_L, source_L, _compile=True
            )
        else:
            block_mask = None

        score_mod = self.generate_score_mod()

        self.sdpa = functools.partial(
            flex_attention, score_mod=score_mod, block_mask=block_mask
        )
        # Register buffers so they inherit device of parent module.
        # K and Q are invariant for each forward pass and K == Q
        self.register_buffer(
            "K",
            einops.repeat(torch.eye(source_L), "H W -> N H W", N=self.batch_size),
            persistent=False,
        )
        self.register_buffer(
            "Q",
            einops.repeat(torch.eye(target_L), "H W -> N H W", N=self.batch_size),
            persistent=False,
        )

        # Register a _persistent_ buffer for attn matrix so its accessible
        # from the state dict during training --> don't need to output from module
        self.register_buffer(
            "attention_matrix",
            einops.repeat(
                torch.zeros((target_L, source_L)), "H W -> N H W", N=self.batch_size
            ),
        )
        self.Q_weights = nn.Sequential(
            nn.Linear(target_L, embedding_L, bias=False),
        )
        self.K_weights = nn.Sequential(
            nn.Linear(source_L, embedding_L, bias=False),
        )
        self.V_weights = nn.Linear(embedding_L, embedding_L, bias=False)

    def forward(self, inputs: Tensor) -> Tensor | Tuple[Tensor, Tensor]:
        in_size = inputs.shape
        out_size = torch.Size(
            [self.batch_size, self.attention_dims[0], self.embedding_L]
        )
        assert in_size == out_size, (
            "Input size is {in_size}, but expected size: {out_size}".format(
                in_size=in_size, out_size=out_size
            )
        )
        K_weighted = self.K_weights(self.K).unsqueeze(1)
        Q_weighted = self.Q_weights(self.Q).unsqueeze(1)
        V_weighted = self.V_weights(inputs).unsqueeze(1)
        out = self.sdpa(Q_weighted, K_weighted, V_weighted).squeeze()
        # out = nn.functional.scaled_dot_product_attention(
        #    Q_weighted, K_weighted, V_weighted
        # ).squeeze()

        return out

    def generate_mask_mod(self) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        def mask_mod(
            _batch: Tensor, _head: Tensor, q_idx: Tensor, kv_idx: Tensor
        ) -> Tensor:
            return torch.as_tensor(not q_idx == kv_idx)

        return mask_mod

    def generate_score_mod(self) -> _score_mod_signature:
        def score_mod(
            score: Tensor, _B: Tensor, _head: Tensor, _q_idx: Tensor, _k_idx: Tensor
        ) -> Tensor:
            self.attention_matrix = torch.clone(score).detach()
            return score

        return score_mod
