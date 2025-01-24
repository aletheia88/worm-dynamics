from attention_predict.dataset import CElegansDataset, CElegansDatasetPlus
from attention_predict.unet import UNet
from copy import deepcopy
from tqdm import tqdm
import json
import numpy as np
import torch


def aggregate_loss(targets, outputs, recorded_neuron_indices, behavior_indices):
    """ Compute MSE loss for each sample in a batch separately on the reconstruction of
    recorded neural and behavioral activities. """

    # both `targets` and `outputs` have shape (batch_size, num_inputs, window_size)
    batch_size = targets.shape[0]
    mse = torch.nn.MSELoss()
    loss = 0
    for n in range(batch_size):
        # for now assume neuron indices are 0 and 1 (AVA and MC)
        indices = recorded_neuron_indices[n] + behavior_indices
        loss += mse(targets[n, indices, :], outputs[n, indices, :])

    return loss


def get_mask_indices(recorded_neuron_indices,
                     num_inputs,
                     neuron_indices,
                     behavior_indices):
    """
    Function observes the following masking scheme -
    FOR NEURONS
        * Randomly masking out AVA or MC if both are recorded;
        * Automatically masking out MC if only AVA is recorded.
    FOR BEHAVIORS
        * Randomly masking out one behavior;
        * Masking out pumping if AVA is masked; masking out velocity if MC is masked.
    """
    mask_indices = {} # keys correspond to sample index in a batch
    all_indices = list(range(num_inputs))

    for n in recorded_neuron_indices.keys():

        # randomly mask out one neuron if all neurons are recored
        if len(recorded_neuron_indices[n]) == 2:
            neuron_index = np.random.choice(recorded_neuron_indices[n])
        else:
            options = list(set(neuron_indices) - set(recorded_neuron_indices[n]))
            neuron_index = np.random.choice(options)

        # For now we disallow simultaneous masking of AVA-velocity or MC-pumping pairs
        # if neuron_index == 0: # masking AVA-pumping
        #     behavior_index = 3
        # elif neuron_index == 1: # masking MC-velocity
        #     behavior_index = 2
        # Randomly select behavior without neuron-behavior pairing priors
        behavior_index = np.random.choice(behavior_indices)
        mask_indices[n] = [neuron_index, behavior_index]

    return mask_indices


def get_mask_indices_v1(
        num_inputs,
        recorded_neuron_indices,
        neuron_indices,
        num_mask_indices=3):
    """
    Selects three columns to mask out acccording to the following scheme -
    FOR NEURONS and BEHAVIORS
        * Automatically masking out the missing neurons (maximum possible is 2);
        * Randomly select (3 - # missing neurons) columns to mask out
    """
    mask_indices = {}
    all_indices = list(range(num_inputs))

    for n in recorded_neuron_indices.keys():
        neuron_mask_indices = list(set(neuron_indices) - set(recorded_neuron_indices[n]))
        remaining_indices = list(set(all_indices) - set(neuron_mask_indices))
        num_to_be_masked = num_mask_indices - len(neuron_mask_indices)
        rest_mask_indices = np.random.choice(remaining_indices, num_to_be_masked)
        mask_indices[n] = neuron_mask_indices + rest_mask_indices.tolist()

    return mask_indices


def get_recorded_neuron_indices(targets, num_neurons):

    batch_size = targets.shape[0]
    recorded_neuron_indices = {}

    for n in range(batch_size):
        recorded_neuron_indices[n] = [
            i for i in range(num_neurons)
            if (torch.behavior_indices = list(range(num_neurons, num_inputs))max(targets[n, i, :]).item() != 0
            and torch.min(targets[n, i, :]).item() != 0)
        ]
    return recorded_neuron_indices


def train(
        num_neurons,
        training_dataset,
        validation_dataset,
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

    # validation_dataloader = torch.utils.data.DataLoader(
    #     validation_dataset,
    #     batch_size=batch_size,
    #     shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_dict = {'training': [], 'validation': []}
    num_inputs = next(iter(training_dataloader))[0].shape[1]
    num_batches = len(training_dataloader)

    # Assume all behaviors are always recorded
    neuron_indices = list(range(num_neurons))
    behavior_indices = list(range(num_neurons, num_inputs))

    # set random seed for reproducible masking
    # torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    for n_epoch in tqdm(range(num_epochs)):

        loss_average = 0

        for n_iter, (inputs, _, _, _, _) in enumerate(training_dataloader):

            if n_iter == num_iterations:
                break

            targets = deepcopy(inputs)
            # Find which neurons are recorded per sample in a batch
            recorded_neuron_indices = get_recorded_neuron_indices(targets, num_neurons)
            # Apply masking to each sample according to which neurons are recorded
            mask_indices = get_mask_indices_v1(num_inputs, recorded_neuron_indices,
                                               neuron_indices)
            for n in mask_indices.keys():
                inputs[n, mask_indices[n], :] = 0

            optimizer.zero_grad()
            outputs = model(inputs)
            # Compute MSE loss for each sample and sum up
            loss = aggregate_loss(targets, outputs, recorded_neuron_indices,
                                  behavior_indices)
            loss.backward()
            optimizer.step()
            loss_average += loss.item()

        # Estimate loss on validation datasets
        loss_dict['training'].append(loss_average / num_batches)
        # loss_dict['validation'].append(validate(
        #         model,
        #         validation_dataloader,
        #         optimizer,
        #         num_iterations,
        #         num_inputs,
        #         num_neurons,
        #         neuron_indices,
        #         behavior_indices))

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


@torch.no_grad()
def validate(
        model,
        validation_dataloader,
        optimizer,
        num_iterations,
        num_inputs,
        num_neurons,
        neuron_indices,
        behavior_indices
):
    loss_average = 0
    model.eval()
    num_batches = len(validation_dataloader)

    for i, (inputs, _, _, _) in enumerate(validation_dataloader):
        if i == num_iterations:
            break
        targets = deepcopy(inputs)

        # Find which neurons are recorded per sample in a batch
        recorded_neuron_indices = get_recorded_neuron_indices(targets, num_neurons)
        # Apply masking to each sample according to which neurons are recorded
        mask_indices = get_mask_indices(recorded_neuron_indices, num_inputs,
                                        neuron_indices, behavior_indices)

        for n in mask_indices.keys():
            inputs[n, mask_indices[n], :] = 0

        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute MSE loss for each sample and sum up
        loss = aggregate_loss(targets, outputs, recorded_neuron_indices,
                              behavior_indices)
        loss_average += loss.item()

    model.train()

    return loss_average / num_batches


def ensure_dir_exists(directories):
    import os
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


if __name__ == "__main__":

    ### Test MSE loss computation ###
    # num_neurons = 2
    # targets = torch.randn((32, 4, 1600))
    # # manually masking out a few neurons
    # samples = list(range(10, 30))
    # targets[samples, 1, :] = 0

    # outputs = torch.randn((32, 4, 1600))
    # loss = aggregate_loss(targets, outputs, num_neurons)
    # print(f'aggregated loss: {loss}')

    device = "cuda:2"
    window_size = 400
    window_stride = 1
    ds_name = 'AVA_MC_SMDV'
    batch_size = 32
    base = '/home/alicia/notebook/alicia/worm-dynamics'

    training_dataset = CElegansDatasetPlus(
        f'{base}/data/{ds_name}_train_shuffle.npy',
        f'{base}/data/{ds_name}_train_ds_shuffle.npy',
        window_stride=window_stride,
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )
    validation_dataset = CElegansDatasetPlus(
        f'{base}/data/{ds_name}_valid.npy',
        f'{base}/data/{ds_name}_valid_ds.npy',
        window_stride=window_stride,
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )

    depth = 5
    in_channels = np.load(f'{base}/data/{ds_name}.npy').shape[1]
    out_channels = in_channels
    unet_dim = 1
    num_iterations = 1_000_000
    num_epochs = 100
    learning_rate = 1e-4
    exp_name = 'exp_2024101100_control'
    log_directory = f'/home/alicia/store1/alicia/attention_predict/{exp_name}'
    log_ckpt_freq = 10
    random_seed = 1912 # Alan Turing's birth year :)

    num_neurons = 3

    model = UNet(
            depth, in_channels, out_channels, unet_dim=unet_dim
    ).to(device)

    train(
            num_neurons,
            training_dataset,
            validation_dataset,
            model,
            batch_size,
            num_iterations,
            num_epochs,
            learning_rate,
            random_seed,
            log_directory,
            log_ckpt_freq=log_ckpt_freq
    )
