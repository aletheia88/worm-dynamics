from attention_predict.dataset import CElegansDataset, CElegansDatasetPlus
from attention_predict.unet import UNet
from copy import deepcopy
from tqdm import tqdm
import json
import numpy as np
import torch


def train(
        num_neurons,
        training_dataset,
        model,
        batch_size,
        num_iterations,
        num_epochs,
        learning_rate,
        log_directory,
        log_ckpt_freq=None,
        random_seed=None
):

    ckpt_directory = f'{log_directory}/checkpoints'
    ensure_dir_exists([ckpt_directory])

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_dict = {'training': [], 'validation': []}
    num_inputs = next(iter(training_dataloader))[0].shape[1]
    num_batches = len(training_dataloader)
    neuron_indices = list(range(num_neurons))
    behavior_indices = list(range(num_neurons, num_inputs))

    # set random seed for reproducible masking
    if random_seed is not None:
        np.random.seed(random_seed)

    for n_epoch in tqdm(range(num_epochs)):

        loss_average = 0

        for n_iter, (inputs, _, _, _, _) in enumerate(training_dataloader):

            if n_iter == num_iterations:
                break

            targets = deepcopy(inputs)

            recorded_neuron_index_dict = get_recorded_neuron_indices(targets, num_neurons)
            ### Masking scheme v0 ###
            # masking 4 random neurons → these neurons are excluded from contributing to
            # the loss function
            # → neuron | velocity | pumping | head angle
            # (after step 1) 50% chance of masking the neuron
            # 50% chance of masking behavior (50% chance of masking either 1 or 2
            # behaviors)

            # mask_index_dict = get_mask_indices(
            #         recorded_neuron_index_dict,
            #         neuron_indices,
            #         behavior_indices)

            ### Masking scheme v1: random masking w/o constraint ###
            mask_index_dict = get_mask_indices_v1(
                    recorded_neuron_index_dict,
                    neuron_indices,
                    behavior_indices)

            for n in mask_index_dict.keys():
                inputs[n, mask_index_dict[n], :] = 0

            optimizer.zero_grad()
            outputs = model(inputs)

            ### Loss v0 for Masking scheme v0 ###
            # loss = aggregate_loss(
            #         targets,
            #         outputs,
            #         mask_index_dict,
            #         behavior_indices,
            #         recorded_neuron_index_dict)

            ### Loss v1 for Masking scheme v1 ###
            loss = aggregate_loss_v1(
                    targets,
                    outputs,
                    behavior_indices,
                    recorded_neuron_index_dict)

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
                    'epoch': n_epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'training_loss': loss_average / num_batches
                },
                    f'{log_directory}/checkpoints/model_ckpt{n_epoch+1}.pt'
            )


def get_mask_indices(
        recorded_neuron_index_dict,
        neuron_indices,
        behavior_indices,
        total_masks=4):

    """ Mask out 4 columns from a total of five. Missing neurons are automatically
    masked. """

    # each key indicates the sample index in a batch
    mask_index_dict = {}

    for n in recorded_neuron_index_dict.keys():
        # automatically masking out the missing neurons
        recorded_neuron_indices = recorded_neuron_index_dict[n]
        missing_neuron_indices = list(set(neuron_indices) -
                                      set(recorded_neuron_indices))
        # masking out additional neurons if needed
        num_to_be_masked = total_masks - len(missing_neuron_indices)
        if num_to_be_masked > 0:
            mask_neuron_indices = missing_neuron_indices + np.random.choice(
                    recorded_neuron_indices,
                    num_to_be_masked,
                    replace=False).tolist()
        else:
            mask_neuron_indices = missing_neuron_indices

        # 50% chance of masking out the only neuron that remains
        # 50% chance of masking out behaviors
        last_neuron_index = list(set(neuron_indices) - set(mask_neuron_indices))
        mask_choice = np.random.choice([*last_neuron_index, -1])
        if mask_choice == -1:
            # if masking out behaviors, 50% chance of masking one behavior and 50%
            # chance of masking two behaviors
            num_behavior_mask_indices = np.random.choice([1, 2])
            mask_index_dict[n] = np.random.choice(behavior_indices,
                                                  num_behavior_mask_indices,
                                                  replace=False).tolist()
        else:
            mask_index_dict[n] = [mask_choice]

    return mask_index_dict


def get_mask_indices_v1(
        recorded_neuron_index_dict,
        neuron_indices,
        behavior_indices,
        num_mask_choices=[4, 5]):

    """ Mask out 4 or 5 columns at random. No restriction on the respective number of
    neural or behavioral columns. Unrecorded neurons are automatically masked out. """

    mask_index_dict = {}

    for n in recorded_neuron_index_dict.keys():

        total_masks = np.random.choice(num_mask_choices)
        recorded_neuron_indices = recorded_neuron_index_dict[n]
        missing_neuron_indices = list(set(neuron_indices) -
                                      set(recorded_neuron_indices))
        num_to_be_masked = total_masks - len(missing_neuron_indices)
        if num_to_be_masked >= 1:
            mask_index_dict[n] = missing_neuron_indices + np.random.choice(
                    recorded_neuron_indices + behavior_indices,
                    num_to_be_masked,
                    replace=False).tolist()
        else:
            mask_index_dict[n] = missing_neuron_indices

    return mask_index_dict


def get_recorded_neuron_indices(targets, num_neurons):

    batch_size = targets.shape[0]
    # each key indicate a sample index in the batch
    # the values corresponding to the column indices where data is recorded
    recorded_neuron_index_dict = {}

    for n in range(batch_size):
        recorded_neuron_index_dict[n] = [
            i for i in range(num_neurons)
            if (torch.max(targets[n, i, :]).item() != 0
            and torch.min(targets[n, i, :]).item() != 0)
        ]
    return recorded_neuron_index_dict


def aggregate_loss(
        targets,
        outputs,
        mask_index_dict,
        behavior_indices,
        recorded_neuron_index_dict):

    """ Compute MSE loss for each sample on the reconstruction of one neuron and the
    standard behaviors. """
    batch_size = targets.shape[0]
    mse = torch.nn.MSELoss()
    loss = 0

    for n in range(batch_size):
        mask_indices = mask_index_dict[n]
        recorded_neuron_indices = recorded_neuron_index_dict[n]
        # when the masked column is only the single remaining neuron
        if len(mask_indices) == 1 and mask_indices[0] < num_neurons:
            loss_columns = mask_indices + behavior_indices
        # when the masked columns are behavior(s)
        else:
            loss_columns = recorded_neuron_indices + behavior_indices
        loss += mse(targets[n, loss_columns, :],
                    outputs[n, loss_columns, :])

    return loss


def aggregate_loss_v1(
        targets,
        outputs,
        behavior_indices,
        recorded_neuron_index_dict):

    """ Compute MSE loss for each sample on the reconstruction of all recorded
    neurons and all behaviors. """
    batch_size = targets.shape[0]
    mse = torch.nn.MSELoss()
    loss = 0

    for n in range(batch_size):
        loss_columns = recorded_neuron_index_dict[n] + behavior_indices
        loss += mse(targets[n, loss_columns, :], outputs[n, loss_columns, :])

    return loss


def ensure_dir_exists(directories):
    import os
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


if __name__ == "__main__":

    device = "cuda:2"
    window_size = 400
    window_stride = 1
    ds_name = 'RID_AVE_RIV_AVD_AIN'
    batch_size = 32
    base = '/store1/alicia/attention_predict/data'

    training_dataset = CElegansDatasetPlus(
        f'{base}/{ds_name}_train_shuffle.npy',
        f'{base}/{ds_name}_train_ds_shuffle.npy',
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
    in_channels = np.load(f'{base}/{ds_name}.npy').shape[1]
    out_channels = in_channels
    unet_dim = 1
    num_iterations = 1_000_000
    num_epochs = 201
    learning_rate = 1e-4
    exp_name = f'exp_2024102201_control'
    log_directory = f'/home/alicia/store1/alicia/attention_predict/{exp_name}'
    log_ckpt_freq = 10
    random_seed = None
    num_neurons = 5

    model = UNet(
            depth, in_channels, out_channels, unet_dim=unet_dim
    ).to(device)

    train(
            num_neurons,
            training_dataset,
            model,
            batch_size,
            num_iterations,
            num_epochs,
            learning_rate,
            log_directory,
            log_ckpt_freq=log_ckpt_freq,
            random_seed=random_seed
    )
