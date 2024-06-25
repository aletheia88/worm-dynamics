from parameters import Parameters
from log import Log
from model import WormTransformer


def initialize(inputs, attention_log_freq):

    B, T, C = inputs.shape
    idx = random.choice(list(range(B)))

    positional_embeddings = torch.from_numpy(
        get_positional_encoding(T, C)).to(device)

    log = Log(attention_log_freq)
    model = WormTransformer(positional_embeddings, log).to(device)

    for param in model.parameters():
        param.data = param.data.double()

    return model, log

@torch.no_grad()
def estimate_loss():

    model.eval()
    xb, yb = get_batch(valid_data)
    y, valid_loss = model(xb, yb, mask_index=1)
    log.iteration_logs[-1].valid_loss = valid_loss.item()

    model.train()

attention_log_freq, loss_log_freq = 100, 1

model, log = initialize(torch.tensor(train_data), attention_log_freq)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for num_iter in tqdm(range(max_iters)[:]):

    log.log_iteration(num_iter)
    xb, yb = get_batch(train_data)
    y, loss = model(xb, yb, mask_index=1)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    log.iteration_logs[-1].train_loss = loss.item()
    if num_iter % eval_iters == 0:
        estimate_loss()

