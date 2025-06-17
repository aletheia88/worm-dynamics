import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_predict.dataset import ShuffledWormData

# from torch.profiler import profile, record_function, ProfilerActivity
from attention_predict.model import MiniAttentionModel
from attention_predict.old.attention_model_mini import AttentionModelMini
from attention_predict.old.dataset import CElegansDatasetPlus

# Enable F32 tensor cores:
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

batch_size = 32
num_encoders = 36
num_decoders = 34
L = 512
depth = 5

inp_dims = (batch_size, num_encoders, L)
fake_input = torch.rand(inp_dims).to(device)

# Revised model
model = MiniAttentionModel(
    num_decoders=num_decoders, input_dims=inp_dims, depth=depth, features_basis=32
)
model.to(device)
model.compile(mode="max-autotune")
model(fake_input)  # warmup model


# Benchmark the compiled model
# t0 = benchmark.Timer(
#     stmt="model(fake_input)", globals={"fake_input": fake_input, "model": model}
# ).timeit(100)

# Alicia's old implementation
alicia = AttentionModelMini(
    depth,  # depth
    num_encoders,  # num neurons
    num_decoders,  # num behaviors
    "BfromN",
    window_size=512,
    num_fmaps=32,
    device="cuda",
)
alicia.to(device)
alicia(fake_input)


# t1 = benchmark.Timer(
#     stmt="alicia(fake_input)", globals={"fake_input": fake_input, "alicia": alicia}
# ).timeit(100)
#
# print(t0)
# print(t1)

################################################################################

def training_loop_bench(model, loss_fn, optimizer, dataloader, max_iterations):
    for n_iter, (inputs, targets) in tqdm(enumerate(dataloader)):
        if n_iter == max_iterations:
            break
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda'):
            pred, _ = model(inputs)
            loss = loss_fn(targets, pred)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


num_epochs = 5  # 2500
max_iterations = 50  # 1_000_000
learning_rate = 5e-5
mse_loss = torch.nn.MSELoss().cuda()
scaler = torch.amp.GradScaler()

# New model setup
fake_dataset = torch.rand((96, 70, 1600)).to(device)
d = ShuffledWormData(fake_dataset, num_encoders, 1, L)
new_dataloader = DataLoader(d, batch_size=batch_size, shuffle=True)
new_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_loop_bench(model, mse_loss, new_optimizer, new_dataloader, max_iterations)

# Old model setup
fake_np_dataset = np.random.rand(96, 70, 1600)
d2 = CElegansDatasetPlus(fake_np_dataset, fake_np_dataset, 1, L, num_encoders, device="cuda")
old_dataloader = DataLoader(d2, batch_size=batch_size, shuffle=True)
old_optimizer = torch.optim.Adam(alicia.parameters(), lr=learning_rate)

training_loop_bench(alicia, mse_loss, old_optimizer, old_dataloader, max_iterations)
