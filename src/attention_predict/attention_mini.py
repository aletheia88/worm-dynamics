import torch
import torch.nn as nn
import torch.nn.function as F
import einops


class AttentionBlockMini(nn.Module):
    def __init__(self, embedding_L, attention_scheme, N, control_experiment=False):
        super().__init__()
        # Because we always use a single quadrant in the mini model,
        # all possible attention schemes can be expressed with a tuple.
        # ex. "NfromB" = (fromB, toN)
        from_dim, to_dim = attention_scheme
        self.sqrt_dk = to_dim**0.5
        # Register buffers so that all child tensors can be moved together
        # to the GPU via model.cuda()
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
            einops.repeat(torch.eye(from_dim), "H W -> N H W", N=N),
            persistent=False,
        )
        # Register a _persistent_ buffer for attn matrix so its accessible
        # from the state dict during training --> don't need to output from module
        self.register_buffer(
            "attention_matrix",
            torch.zeros((1, 2, 3)),  # FIXME: placeholder vals, use correct dims
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
        K_weighted = self.K_weights(self.KQ)
        Q_weighted = self.Q_weights(self.KQ)
        V_weighted = self.V_weights(inputs)
        # TODO: We can use flex attention to achieve the same as below
        # but with an optimized backend
        attention = Q_weighted @ K_weighted
        attention = einops.rearrange(attention, "n h w -> n w h") / sqrt_dk
        attention = attention.masked_fill(self.mask, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        out = attention @ V_weighted
        return out

    def flex_callback(self):
        # TODO: This should copy the attention matrix to the buffer, so
        # we can save it from the state dict if we want to
        pass
