
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
# Use arbitrary dims for projected Q/K in Alicia's code
num_encoders = 4
num_decoders = 5
# if depth = 5 and window_size
depth = 5
window_size = 400
num_fmaps = 8
inc_factor = 2
fmaps_out = num_fmaps * (inc_factor**(depth - 1) )

L = window_size // 2**(depth-1)
embedding_L = fmaps_out * L # assume window_size of 512

# Setup according to original model
K = torch.rand(num_encoders, num_encoders) # dummy output of first Linear projection
Q = torch.rand(num_encoders, num_encoders)

kproj = nn.Linear(num_encoders, num_decoders)
qproj = nn.Linear(num_encoders, num_encoders)

K_weighted = kproj(K)
Q_weighted = qproj(Q)
V_weighted = torch.rand(num_encoders, embedding_L)

### SDPA using Alicia's original ###

# att_matrix dims = (num_decoders, num_encoders)
att_matrix = (Q_weighted @ K_weighted).T

# att_matrix dims = (num_decoders, num_encoders)
att_matrix = F.softmax(att_matrix, dim=-1)

# output dims = (num_decoders, embedding_L)
output1 = att_matrix @ V_weighted

### SDPA using torch implementation ###

# Torch scaled_dot_product_attention is equivalent to:
#  attn_weight = query @ key.T
#  attn_weight = torch.softmax(attn_weight, dim=-1)
#  output = attn_weight @ value

# Main difference is Q @ K.T instead of (Q @ K).T
# Since (QK.T).T == QK.T, 

# Torch expects that dims of K = (S, Eqk) and Q = (L, Eqk) where:
# S = source = num_encoders
# L = target = num_decoders
# and we assume that Eqk = num_encoders since from the model K = Q = torch.eye(num_encoders) -> Linear(num_encoders,num_encoders)

# Currently I am swapping the weight dimensions of the final Linear projections, i.e.:
# Q_weights = Linear(num_encoders, num_DEcoders)
# K_weights = Linear(num_encoders, num_ENcoders)
K_weighted_torch = Q_weighted # (S, Eqk)
Q_weighted_torch = K_weighted # (Eqk, L)

# Add batch and single head dim + transpose Q
K_h = einops.rearrange(K_weighted_torch, "S Eqk -> 1 1 S Eqk") 
Q_h = einops.rearrange(Q_weighted_torch, "Eqk L -> 1 1 L Eqk")
V_h = einops.rearrange(V, "S Ev -> 1 1 S Ev")

output2 = F.scaled_dot_product_attention(Q_h, K_h, V_h)
output2 = output2.squeeze()





