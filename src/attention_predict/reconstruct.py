# Script for reconstructing whole animal trace given model checkpoint and
# testing/validation dataset
from attention_predict.dataset import CElegansDatasetPlus
from attention_predict.attention_model import AttentionModel
from attention_predict.attention_model_2 import AttentionModel2
from attention_predict.attention_model_3 import AttentionModel3
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch


def build_model(
    architecture,
    attention_scheme,
    depth=3,
    num_neurons=3,
    num_behaviors=3,
    window_size=400,
    device='cuda:3'
):
    if architecture == 'attention_model_1':
        model = AttentionModel(
            depth,
            num_neurons,
            num_behaviors,
            window_size,
            attention_scheme=attention_scheme,
            device=device)
    elif architecture == 'attention_model_2':
        model = AttentionModel2(
            depth,
            num_neurons,
            num_behaviors,
            window_size,
            attention_scheme=attention_scheme,
            device=device)
    elif architecture == 'attention_model_3':
        model = AttentionModel3(
            depth,
            num_neurons,
            num_behaviors,
            window_size,
            attention_scheme=attention_scheme,
            device=device)

    return model


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
        shuffle=False)
    # shuffle must set to False to reconstruct trace

    return dataloader


@torch.no_grad()
def reconstruct_traces(
    model,
    dataloader,
    ckpt,
    experiment,
    num_worms,
    architecture,
    num_neurons=3,
    max_length=1600,
    window_size=400
):
    model_ckpt_path = f'/store1/alicia/attention_predict/{experiment}/checkpoints/model_ckpt{ckpt}.pt'

    # load trained model
    checkpoint = torch.load(model_ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    reconstruction_loss = torch.nn.MSELoss()

    left_slider = 0
    right_slider = 0

    reconstructed_traces = {
        worm_index: { 
            'dataset': None,
            **{
                'ground_truth': [],
                'prediction': [],
                'attn_weights': [],
                'frames': [],
                'mse': [],
            }
        } for worm_index in range(num_worms)
    }

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

            targets = deepcopy(inputs)
            outputs, attn_weights = model(inputs)
            recorded_neuron_indices = get_recorded_neuron_indices(targets,
                                                                  num_neurons)
            loss = reconstruction_loss(
                    targets[:, recorded_neuron_indices, :],
                    outputs[:, recorded_neuron_indices, :])

            mask_outcomes = reconstructed_traces[worm_index]
            mask_outcomes['ground_truth'].append(targets.cpu().detach().numpy())
            mask_outcomes['prediction'].append(outputs.cpu().detach().numpy())
            mask_outcomes['frames'].append((start_frame.item(),
                                            end_frame.item()))
            mask_outcomes['mse'].append(loss.item())
            if architecture in ['attention_model_1', 'attention_model_2']:
                mask_outcomes['attn_weights'].append(attn_weights.cpu().detach().numpy())
            elif architecture == 'attention_model_3':
                mask_outcomes['attn_weights'].append(
                    [attn.cpu().detach().numpy() for attn in attn_weights]
                )

    for i in range(num_worms):
        print(f"\n All frames added to worm{i}: {reconstructed_traces[i]['frames']}")

    return reconstructed_traces


def get_recorded_neuron_indices(targets, num_neurons):

    return [
        i for i in range(num_neurons)
        if (torch.max(targets[0, i, :]).item() != -10
        and torch.min(targets[0, i, :]).item() != -10)
    ]
