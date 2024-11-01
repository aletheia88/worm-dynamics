# Script for reconstructing whole animal trace given model checkpoint and
# testing/validation dataset
from attention_predict.dataset import CElegansDataset, CElegansDatasetPlus
from attention_predict.model import PredictModel
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch


@torch.no_grad()
def reconstruct_traces(
        data_path,
        dataset_path,
        model,
        model_ckpt_path,
        max_length,
        model_type,
        window_size,
        window_stride=1,
        mask_indices=None,
        device='cpu'
):
    """ Reconstruct whole time series using the trained model.

    Args:
        model_type: 'unet' or 'attention'.

    Example:
        >>> prj_directory = '/home/alicia/notebook/alicia/attention_predict'
        >>> data_path = f'{prj_directory}/data/AVA_MC_test.npy'
        >>> dataset_path = f'{prj_directory}/data/AVA_MC_test_ds.npy'
        >>> device = 'cuda:2'
        >>> window_size = 50
        >>> window_stride = 1 # Default is 1
        >>> batch_size = 1
        >>> embedding_dims = 1024
        >>> num_layers = 4
        >>> max_length = 1600
        >>> model_ckpt_path = f'{prj_directory}/experiments/exp_20240915/checkpoints/model_ckpt76.pt'
        >>> model_type = 'attention'
        >>> # initialize the attention model
        >>> model = PredictModel(
        >>>    num_inputs=dataset.num_variables,
        >>>    input_dims=window_size,
        >>>    embedding_dims=embedding_dims,
        >>>    num_layers=num_layers,
        >>>    residual=True,
        >>>    normalize=True,
        >>>    device=device
        >>> ).to(device)
        >>> reconstructed_traces = reconstruct_traces(
        >>>        data_path,
        >>>        dataset_path,
        >>>        model,
        >>>        model_ckpt_path,
        >>>        max_length,
        >>>        model_type,
        >>>        window_size,
        >>>        window_stride,
        >>>        device)
    """
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

    if model_type == 'attention':
        reconstructed_traces = {
                worm: {
                    'ground_truth': [],
                    'prediction': [],
                    'attn_weights': [],
                    'frames': [],
                    'mse': []
                } for worm in range(num_worms)
            }
    elif model_type == 'unet':
        reconstructed_traces = {
            worm: { 
                'dataset': None,
                **{
                    mask_index: {
                        'ground_truth': [],
                        'prediction': [],
                        'frames': [],
                        'mse': [],
                    } for mask_index in range(num_inputs)
                }
            } for worm in range(num_worms)
        }
    elif model_type == 'hybrid':
        reconstructed_traces = {
            worm: { 
                'dataset': None,
                **{
                    mask_index: {
                        'ground_truth': [],
                        'prediction': [],
                        'attn_weights': [],
                        'frames': [],
                        'mse': [],
                    } for mask_index in range(num_inputs)
                }
            } for worm in range(num_worms)
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
            if model_type == 'attention':
                outputs, weights = model(inputs)
                loss = reconstruction_loss(inputs, outputs)

                reconstructed_traces[worm_index]['attn_weights'].append(weights.cpu().detach().numpy())
                reconstructed_traces[worm_index]['ground_truth'].append(inputs.cpu().detach().numpy())
                reconstructed_traces[worm_index]['prediction'].append(outputs.cpu().detach().numpy())
                reconstructed_traces[worm_index]['frames'].append((start_frame.item(),
                                                                   end_frame.item()))
                reconstructed_traces[worm_index]['mse'].append(loss.item())

            elif model_type == 'unet' or model_type == 'hybrid':

                targets = deepcopy(inputs)

                if mask_indices is None:

                    # reconstruct after masking out each kind of activity
                    for mask_index in range(num_inputs):

                        inputs[:, mask_index, :] = 0
                        outputs = model(inputs)
                        loss = reconstruction_loss(
                                targets[:, mask_index, :],
                                outputs[:, mask_index, :])

                        mask_outcomes = reconstructed_traces[worm_index][mask_index]
                        mask_outcomes['ground_truth'].append(targets.cpu().detach().numpy())
                        mask_outcomes['prediction'].append(outputs.cpu().detach().numpy())
                        mask_outcomes['frames'].append((start_frame.item(),
                                                        end_frame.item()))
                        mask_outcomes['mse'].append(loss.item())
                        # set to original inputs to mask new column
                        inputs = deepcopy(targets)

                elif len(mask_indices) > 0:

                    # mask out more than one column at once
                    inputs[:, mask_indices, :] = 0
                    # Replace unmasked data with random noise
                    # unmask_indices = list(set(list(range(num_inputs))) -
                    #                       set(mask_indices))
                    # noise = torch.randn(inputs[:, unmask_indices, :].shape).to(device)
                    # inputs[:, unmask_indices, :] = noise
                    if model_type == 'unet':
                        outputs = model(inputs)
                    elif model_type == 'hybrid':
                        outputs, attn_weights = model(inputs)
                    # keep the outcome of applying masking in the first item
                    mask_outcomes = reconstructed_traces[worm_index][0]
                    mask_outcomes['ground_truth'].append(targets.cpu().detach().numpy())
                    mask_outcomes['prediction'].append(outputs.cpu().detach().numpy())
                    mask_outcomes['frames'].append((start_frame.item(),
                                                    end_frame.item()))
                    # compute MSE loss for each column reconstruction
                    for i in range(num_inputs):
                        loss = reconstruction_loss(targets[:, i, :], outputs[:, i, :])
                        reconstructed_traces[worm_index][i]['mse'].append(loss.item())

        elif (not append and model_type == 'hybrid' and len(mask_indices) > 0):

            inputs[:, mask_indices, :] = 0
            _, attn_weights = model(inputs)
            mask_outcomes = reconstructed_traces[worm_index][0]
            mask_outcomes['attn_weights'].append(attn_weights.cpu().detach().numpy())

    for i in range(num_worms):
        if model_type == 'attention':
            print(f"\n All frames added to worm{i}: {reconstructed_traces[i]['frames']}")
        elif model_type == 'unet' or model_type == 'hybrid':
            print(f"\n All frames added to worm{i}: {reconstructed_traces[i][0]['frames']}")

    return reconstructed_traces
