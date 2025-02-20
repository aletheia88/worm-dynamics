# Script for reconstructing whole animal trace given model checkpoint and
# testing/validation dataset
from attention_predict.dataset import CElegansDatasetPlus
from attention_predict.attention_model_mini import AttentionModelMini
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch


def build_mini_attention_model(
    attention_scheme,
    depth,
    num_neurons,
    num_behaviors,
    window_size,
    num_fmaps,
    device
):
    return AttentionModelMini(
            depth,
            num_neurons,
            num_behaviors,
            attention_scheme,
            window_size,
            num_fmaps,
            device=device
        )


def build_dataloader(ds_name, device):

    prj_directory = '/store1/alicia/attention_predict'
    data_path = f'{prj_directory}/data/{ds_name}.npy'
    dataset_path = f'{prj_directory}/data/{ds_name}_ds.npy'

    dataset = CElegansDatasetPlus(
        data_path,
        dataset_path,
        window_stride=400,
        window_size=400,
        device=device,
        slices=slice(0, 1600)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    # shuffle must set to False to reconstruct trace

    return dataloader


@torch.no_grad()
def reconstruct_traces(
    model,
    attention_scheme,
    dataloader,
    ckpt,
    experiment,
    num_worms,
    num_neurons,
    num_behaviors,
    max_length=1600,
    window_size=400
):
    model_ckpt_path = f'/store1/alicia/attention_predict/{experiment}/checkpoints/model_ckpt{ckpt}.pt'

    # load trained model
    checkpoint = torch.load(model_ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    left_slider = 0
    right_slider = 0

    reconstructed_traces = {
        worm_index: { 
            'dataset': None,
            **{
                'inputs': [],
                'ground_truth': [],
                'prediction': [],
                'attn_weights': [],
                'frames': [],
            }
        } for worm_index in range(num_worms)
    }

    neuron_indices = list(range(num_neurons))
    behavior_indices = list(range(num_neurons, num_neurons + num_behaviors))

    for i, (inputs, worm, start_frame, end_frame, ds) in tqdm(enumerate(dataloader)):

        append = False
        worm_index = worm.item()
        reconstructed_traces[worm_index]['dataset'] = ds[0]

        if start_frame.item() == right_slider:
            append = True
            left_slider = right_slider
            right_slider = left_slider + window_size

        if start_frame.item() == max_length - window_size:
            append = True
            # reset the slider positions to append data from next worm
            left_slider = 0
            right_slider = 0

        if append:

            if attention_scheme == 'BfromN':
                outputs, attention_weights = model(inputs[:, neuron_indices, :])
                targets = inputs[:, behavior_indices, :]
            elif attention_scheme == 'NfromB':
                outputs, attention_weights = model(inputs[:, behavior_indices, :])
                targets = inputs[:, neuron_indices, :]
            elif attention_scheme == 'NfromN':
                outputs, attention_weights = model(inputs[:, neuron_indices, :])
                targets = inputs[:, neuron_indices, :]

            mask_outcomes = reconstructed_traces[worm_index]
            mask_outcomes['inputs'].append(inputs.cpu().detach().numpy())
            mask_outcomes['ground_truth'].append(targets.cpu().detach().numpy())
            mask_outcomes['prediction'].append(outputs.cpu().detach().numpy())
            mask_outcomes['frames'].append((start_frame.item(),
                                            end_frame.item()))
            mask_outcomes['attn_weights'].append(
                [attn.cpu().detach().numpy() for attn in attention_weights]
            )

    for i in range(num_worms):
        print(f"\n All frames added to worm{i}: {reconstructed_traces[i]['frames']}")

    return reconstructed_traces
