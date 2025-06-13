import torch
import torch.utils.benchmark as benchmark

# from torch.profiler import profile, record_function, ProfilerActivity
from attention_predict.model import MiniAttentionModel
from attention_predict.old.attention_model_mini import AttentionModelMini

# Enable F32 tensor cores:
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

num_encoders = 36
num_decoders = 34
L = 512
depth = 5

inp_dims = (32, num_encoders, L)
fake_input = torch.rand(inp_dims).to(device)

# Revised model
model = MiniAttentionModel(
    num_decoders=num_decoders, input_dims=inp_dims, depth=depth, features_basis=32
)
model.to(device)
model.compile(fullgraph=True, mode="max-autotune")
# model(fake_input)

# Alicia's old implementation
alicia = AttentionModelMini(
    depth,  # depth
    num_encoders,  # num neurons
    num_decoders,  # num behaviors
    "NfromB",
    window_size=512,
    num_fmaps=32,
    device="cuda",
)
alicia.to(device)


# Benchmark the compiled model
t0 = benchmark.Timer(
    stmt="model(fake_input)", globals={"fake_input": fake_input, "model": model}
).timeit(100)

t1 = benchmark.Timer(
    stmt="alicia(fake_input)", globals={"fake_input": fake_input, "alicia": alicia}
).timeit(100)

print(t0)
print(t1)

# activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
# sort_by_keyword = "cuda_time_total"
#
# with profile(activities=activities, record_shapes=True) as prof:
#     with record_function("model_inference"):
#         alicia(fake_input)
#
# with open("outtest2.txt", "w") as f:
#     print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10), file=f)
#
# print(prof.key_averages().table(row_limit=10))
#
# prof.export_chrome_trace("foo.json")
