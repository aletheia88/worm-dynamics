# Script for reconstructing whole animal trace given model checkpoint and
# testing/validation dataset
from attention_predict.dataset import CElegansDataset, CElegansDatasetPlus
from attention_predict.model import PredictModel
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch


@torch.no_grad()
def stimulate(
        data_path,
        dataset_path,
        model,
        model_ckpt_path,
        mask_indices,
        stim_indices,
        max_length,
        window_size,
        window_stride,
        device='cpu'
):
    """ Apply stimulation patterns to observe how masked activities react. """

    num_worms = np.load(data_path).shape[0]

    dataset = CElegansDatasetPlus(
        data_path,
        dataset_path,
        window_stride=window_stride,
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False) # shuffle must set to False for trace reconstruction

    # load trained model
    checkpoint = torch.load(model_ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    reconstruction_loss = torch.nn.MSELoss()
    num_inputs = next(iter(dataloader))[0].shape[1]

    left_slider = 0
    right_slider = 0

    responce_traces = {
        worm: { 
            'dataset': None,
            **{
                column_index: {
                    'ground_truth': [],
                    'prediction': [],
                    'inputs': [],
                    'frames': [],
                    'mse': [],
                } for column_index in range(num_inputs)
            }
        } for worm in range(num_worms)
    }

    for i, (inputs, worm, start_frame, end_frame, ds) in tqdm(enumerate(dataloader)):

        append = False
        worm_index = worm.item()
        responce_traces[worm_index]['dataset'] = ds[0]

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

            # apply stimulation patterns
            stim_pattern_dict = generate_stim_patterns(stim_indices, window_size)

            for stim_index in stim_indices:
                inputs[:, stim_index, :] = torch.tensor(
                        stim_pattern_dict[stim_index]).to(device)

            # apply masking
            inputs[:, mask_indices, :] = 0

            # feed stim patterns and context traces to the model
            outputs = model(inputs)
            stim_outcomes = responce_traces[worm_index][0]
            stim_outcomes['ground_truth'].append(targets.cpu().detach().numpy())
            stim_outcomes['prediction'].append(outputs.cpu().detach().numpy())
            stim_outcomes['inputs'].append(inputs.cpu().detach().numpy())
            stim_outcomes['frames'].append((start_frame.item(),
                                            end_frame.item()))
            # compute MSE loss for each column reconstruction
            for i in range(num_inputs):
                loss = reconstruction_loss(targets[:, i, :], outputs[:, i, :])
                responce_traces[worm_index][i]['mse'].append(loss.item())

    for i in range(num_worms):
        print(f"\n All frames added to worm{i}: {responce_traces[i][0]['frames']}")

    return responce_traces


def generate_stim_patterns(stim_indices, window_size):
    stim_patterns = {col: np.array([]) for col in stim_indices}
    pattern_options = ['wave', 'block']

    for stim_index in stim_indices:
        pattern = np.random.choice(pattern_options)
        pattern = 'block'
        if pattern == 'wave':
            stim_patterns[stim_index] = generate_sinusoidal_stimulus(window_size)
        elif pattern == 'block':
            stim_patterns[stim_index] = generate_block_stimulus(window_size)

    return stim_patterns


def generate_block_stimulus(window_size):
    """ Generates a block stimulus array with variability in both block position and
    width. """

    array = np.zeros(window_size)
    block_duration = np.random.randint(int(window_size * 0.01), int(window_size * 0.5))
    start = np.random.randint(0, window_size - block_duration)
    end = start + block_duration
    array[start:end] = 1.0  # block is set to a constant value

    return array


def generate_sinusoidal_stimulus(window_size):
    """ Generates a sinusoidal stimulus with variability in the frequency. """
    # frequency range between 0.5 and 2 cycles per window
    frequency = np.random.uniform(0.5, 2.0)
    time = np.linspace(0, 2 * np.pi, window_size)
    array = np.sin(frequency * time)  # sine wave with variable frequency

    return array

