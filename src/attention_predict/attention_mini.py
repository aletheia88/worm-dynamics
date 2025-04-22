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
    KQ: Tensor
    attention_matrix: Tensor
    K_weights: nn.Sequential
    Q_weights: nn.Sequential
    V_weights: nn.Linear
    spda: Callable[[Tensor, Tensor, Tensor], Tensor | Tuple[Tensor, Tensor]]

    def __init__(
        self,
        embedding_L: int,
        attention_scheme: Tuple[int, int],
        N: int,
        self_attention: bool = False,
        control_experiment: bool = True,
    ) -> None:
        super().__init__()
        # All possible attention schemes can be expressed with a tuple of ints.
        # ex. "NfromB" = (fromB: int, toN: int)
        from_dim, to_dim = attention_scheme

        # mask_mod: returns False for indices of the attention mask that should be skipped
        # score_mod: makes a copy of the attention_matrix and returns the unmodified matrix
        if self_attention & (not control_experiment):
            mask_mod = self.generate_mask_mod()
            block_mask = create_block_mask(
                mask_mod, None, None, from_dim, from_dim, _compile=True
            )
        else:
            block_mask = None

        score_mod = self.generate_score_mod()

        self.sdpa = functools.partial(
            flex_attention, score_mod=score_mod, block_mask=block_mask
        )

        # Register buffers so they inherit device of parent module.
        # K and Q are invariant for each forward pass and K == Q
        # TODO: Verify that this is correct for all attention schemes.
        self.register_buffer(
            "KQ",
            einops.repeat(torch.eye(from_dim), "H W -> N H W", N=N),
            persistent=False,
        )
        # Register a _persistent_ buffer for attn matrix so its accessible
        # from the state dict during training --> don't need to output from module
        self.register_buffer(
            "attention_matrix",
            einops.repeat(torch.zeros((to_dim, from_dim)), "H W -> N H W", N=N),
        )

        self.K_weights = nn.Sequential(
            nn.Linear(from_dim, from_dim, bias=False),
            nn.Linear(from_dim, to_dim, bias=False),
        )
        self.Q_weights = nn.Sequential(
            nn.Linear(from_dim, from_dim, bias=False),
            nn.Linear(from_dim, from_dim, bias=False),
        )
        self.V_weights = nn.Linear(embedding_L, embedding_L, bias=False)

    def forward(self, inputs: Tensor) -> Tensor | Tuple[Tensor, Tensor]:
        K_weighted = self.K_weights(self.KQ)
        Q_weighted = self.Q_weights(self.KQ)
        V_weighted = self.V_weights(inputs)
        return self.sdpa(Q_weighted, K_weighted, V_weighted)

    def generate_mask_mod(self) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        def mask_mod(
            _batch: Tensor, _head: Tensor, q_idx: Tensor, kv_idx: Tensor
        ) -> Tensor:
            return torch.as_tensor(not q_idx == kv_idx)

        return mask_mod

    def generate_score_mod(self) -> _score_mod_signature:
        def score_mod(
            score: Tensor, B: Tensor, head: Tensor, q_idx: Tensor, k_idx: Tensor
        ) -> Tensor:
            self.attention_matrix.copy_(score)
            return score

        return score_mod


if __name__ == "__main__":
    L_embed = 100
    attention_scheme = (14, 7)
    N = 4
    mod = AttentionBlockMini(L_embed, attention_scheme, N)

    print("Model initialized.")
