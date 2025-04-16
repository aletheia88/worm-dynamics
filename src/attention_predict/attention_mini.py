import torch
import torch.nn as nn
import torch.nn.function as F
import einops


class AttentionBlockMini(nn.Module):
    def __init__(self, embedding_L, attention_scheme, N, control_experiment=False):
        super().__init__()
        # All attention schemes can be expressed with a tuple, ex. "NfromB" = (fromB, toN)
        from_dim, to_dim = attention_scheme
        self.sqrt_dk = to_dim**0.5
        # Register buffers so that all tensors auto initialize on the same device
        if not control_experiment & from_dim == to_dim:
            self.register_buffer(
                "mask",
                torch.eye(from_dim, dtype=torch.bool),
                persistent=False,
            )
        else:
            self.register_buffer(
                "mask",
                torch.zeros((to_dim, from_dim), dtype=torch.bool),
                persistent=False,
            )
        # K and Q are invariant for each forward pass and K == Q
        self.register_buffer(
            "KQ",
            einops.repeat(torch.eye(from_dim), "h w -> n h w", n=N),
            persistent=False,
        )
        # Register a _persistent_ buffer for attn matrix so its accessible
        # from the state dict during training --> don't need to output from module
        self.register_buffer(
            "attention_matrix",
            torch.zeros((1, 2, 3)),  # FIXME: math is hard
            persistent=True,
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

    def forward(self, inputs, encoder_outputs):
        sqrt_dk = self.sqrt_dk
        weighted_key = self.K_weights(self.KQ)
        weighted_query = self.Q_weights(self.KQ)
        weighted_value = self.V_weights(inputs)
        attention_matrix = weighted_query @ weighted_key
        attention_matrix = (
            einops.rearrange(attention_matrix, "n h w -> n w h") / sqrt_dk
        )
        attention_matrix = attention_matrix.masked_fill(self.mask, float("-inf"))
        attention_matrix = F.softmax(attention_matrix, dim=-1)
        out = attention_matrix @ weighted_value
        return out, encoder_outputs
