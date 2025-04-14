from tqdm import tqdm
import attention_predict
import torch


def train():

    device = "cuda:3"
    window_size = 400
    window_stride = 1
    ds_name = 'data0108_norm'
    batch_size = 32
    base = '/home/alicia/store1/alicia/attention_predict'

    training_dataset = attention_predict.dataset.CElegansDatasetPlus(
        f'{base}/data/{ds_name}_train.npy',
        f'{base}/data/{ds_name}_train_ds.npy',
        window_stride=window_stride,
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True)

    depth = 4
    num_neurons = 4
    num_behaviors = 3
    window_size = 400
    attention_scheme ='BfromN'
    num_fmaps = 16
    # (batch, channels, height, width)
    # x = torch.rand(batch_size, num_neurons, window_size).to(device)

    model = attention_predict.AttentionModelMini(
            depth,
            num_neurons,
            num_behaviors,
            attention_scheme,
            window_size,
            num_fmaps=num_fmaps,
            device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    neuron_indices = list(range(num_neurons))
    # indices of behavior variables in model inputs
    behavior_indices = list(range(num_neurons, num_neurons + num_behaviors))
    loss_indices = list(range(num_behaviors))

    mse_loss = torch.nn.MSELoss()

    for n_epoch in tqdm(range(2)):

        for n_iter, (inputs, _, _, _, _) in tqdm(enumerate(training_dataloader)):

            outputs, attention_weights = model(inputs[:, neuron_indices, :])
            print(f'outputs: {outputs.shape}')
            targets = inputs[:, behavior_indices, :]
            loss = aggregate_loss(targets, outputs, mse_loss)
            print(f'loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def aggregate_loss(targets, outputs, mse_loss):

    return mse_loss(targets, outputs)


def seed_everything(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # # configure cudNN for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    seed_everything(1976)
    train()
    # y, level_attn_weights = model(x)

#     print(f'outputs dim: {y.shape}')
#     for attn_weights in level_attn_weights:
#         print(f'attention weights: {attn_weights.shape}')
#         print(attn_weights)
