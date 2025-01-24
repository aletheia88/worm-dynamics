from attention_predict.dataset import CElegansDataset, CElegansDatasetPlus
from attention_predict.unet import UNet
from copy import deepcopy
from tqdm import tqdm
import json
import numpy as np
import torch


def train(
        training_dataset,
        model,
        batch_size,
        num_iterations,
        num_epochs,
        learning_rate,
        random_seed,
        log_directory,
        log_ckpt_freq=None,
):

    ckpt_directory = f'{log_directory}/checkpoints'
    ensure_dir_exists([ckpt_directory])

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True)

    reconstruction_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_dict = {'training': [], 'validation': []}
    num_inputs = next(iter(training_dataloader))[0].shape[1]
    num_batches = len(training_dataloader)

    # set random seed for reproducible masking
    # torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    for n_epoch in tqdm(range(num_epochs)):

        loss_average = 0

        for n_iter, (inputs, _, _, _, _) in enumerate(training_dataloader):

            if n_iter == num_iterations:
                break

            targets = deepcopy(inputs)
            # Randomly masking out either column 1 or column 2
            mask_index = np.random.choice([0, 1]) 
            inputs[:, mask_index, :] = 0
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = reconstruction_loss(targets, outputs)
            loss.backward()
            optimizer.step()
            loss_average += loss.item()

        # Estimate loss on validation datasets
        loss_dict['training'].append(loss_average / num_batches)

        with open(f'{log_directory}/losses.json', 'w') as f:
            json.dump(loss_dict, f, indent=4)

        if log_ckpt_freq is not None and n_epoch % log_ckpt_freq == 0:
            torch.save(
                {
                    'epoch': n_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'training_loss': loss_average / num_batches
                },
                    f'{log_directory}/checkpoints/model_ckpt{n_epoch}.pt'
            )


def ensure_dir_exists(directories):
    import os
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


if __name__ == "__main__":

    device = "cuda:2"
    window_size = 400
    window_stride = 1
    ds_name = 'velocity_headangle'
    batch_size = 32
    base = '/home/alicia/notebook/alicia/worm-dynamics'

    training_dataset = CElegansDatasetPlus(
        f'{base}/data/{ds_name}_train.npy',
        f'{base}/data/{ds_name}_train_ds.npy',
        window_stride=window_stride,
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )
    # validation_dataset = CElegansDatasetPlus(
    #     f'{base}/data/{ds_name}_valid.npy',
    #     f'{base}/data/{ds_name}_valid_ds.npy',
    #     window_stride=window_stride,
    #     window_size=window_size,
    #     device=device,
    #     slices=slice(0, 1600)
    # )

    depth = 5
    in_channels = np.load(f'{base}/data/{ds_name}.npy').shape[1]
    out_channels = in_channels
    unet_dim = 1
    num_iterations = 1_000_000
    num_epochs = 100
    learning_rate = 1e-4
    exp_name = 'exp_2024100800'
    log_directory = f'/home/alicia/store1/alicia/attention_predict/{exp_name}'
    log_ckpt_freq = 10
    random_seed = 1912 # Alan Turing's birthday :)

    model = UNet(
            depth, in_channels, out_channels, unet_dim=unet_dim
    ).to(device)

    train(
            training_dataset,
            model,
            batch_size,
            num_iterations,
            num_epochs,
            learning_rate,
            random_seed,
            log_directory,
            log_ckpt_freq=log_ckpt_freq
    )
