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
    perturb_index,
    device
):
    return AttentionModelMini(
            depth,
            num_neurons,
            num_behaviors,
            attention_scheme,
            window_size,
            num_fmaps,
            perturb_index=perturb_index,
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
    device,
    target_index=None,
    perturb_index=None,
    max_length=1600,
    window_size=400
):
    model_ckpt_path = \
        f'/store1/alicia/attention_predict/{experiment}/model_ckpt{ckpt}.pt'

    # load trained model
    checkpoint = torch.load(model_ckpt_path, map_location=device)
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

    input_indices, output_indices = get_indices(
        attention_scheme,
        num_neurons,
        num_behaviors,
        target_index
    )

    for i, (inputs, worm, start_frame, end_frame, ds) in tqdm(enumerate(dataloader)):

        ### if appending selected window ###
        append = False
        ### if appending every window ###
        # append = True
        worm_index = worm.item()
        reconstructed_traces[worm_index]['dataset'] = ds[0]

        ### append every 400-tp window ###
        if start_frame.item() == right_slider:
            append = True
            left_slider = right_slider
            right_slider = left_slider + window_size

        if start_frame.item() == max_length - window_size:
            append = True
            # reset the slider positions to append data from next worm
            left_slider = 0
            right_slider = 0

        #### append 400-tp window within (400, 1159) ###
        # if start_frame >= 401 and start_frame <= 801:
        #     append = True
        # else:
        #     append = False

        if append:
            # TODO: perturb 'AVA' (index 8)
            targets = inputs[:, output_indices, :]
            if perturb_index != None:
                inputs[:, perturb_index * 2, :200] = -0.5
                inputs[:, perturb_index * 2, 200:] = 0.5
            outputs = model(inputs[:, input_indices, :])

            mask_outcomes = reconstructed_traces[worm_index]
            mask_outcomes['inputs'].append(inputs.cpu().detach().numpy())
            mask_outcomes['ground_truth'].append(targets.cpu().detach().numpy())
            mask_outcomes['prediction'].append(outputs.cpu().detach().numpy())
            mask_outcomes['frames'].append((start_frame.item(),
                                            end_frame.item()))
            # mask_outcomes['attn_weights'].append(
            #     [attn.cpu().detach().numpy() for attn in attention_weights]
            # )

    for i in range(num_worms):
        print(f"\n All frames added to worm{i}: {reconstructed_traces[i]['frames']}")

    return reconstructed_traces


def get_indices(
    attention_scheme,
    num_neurons,
    num_behaviors,
    target_index
):
    ### indices for dataset `data0410`
    # neuron_indices = list(range(num_neurons))
    # behavior_indices = list(range(num_neurons, num_neurons + num_behaviors))
    ### indices for dataset `data0612`
    num_inputs = num_neurons + num_behaviors
    neuron_indices = list(range(num_neurons * 2))
    behavior_indices = list(range(num_neurons * 2, num_inputs * 2))

    if attention_scheme == '1fromN':
        input_indices = neuron_indices[:target_index] + neuron_indices[target_index+1:]
        output_indices = [target_index]

    if attention_scheme == 'BfromN':
        input_indices = neuron_indices
        output_indices = behavior_indices

    if attention_scheme == 'NfromB':
        input_indices = behavior_indices
        output_indices = neuron_indices

    if attention_scheme in [
        'NfromN',
        'connectome',
        'anticonnectome',
        'perturb'
    ]:
        input_indices = neuron_indices
        output_indices = neuron_indices

    return input_indices, output_indices
