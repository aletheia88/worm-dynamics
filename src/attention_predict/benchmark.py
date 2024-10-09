from attention_predict.dataset import CElegansDatasetPlus
from attention_predict.unet import UNet
from copy import deepcopy
import numpy as np
import torch


def evaluate(
        model,
        model_ckpt_path,
        mask_indices,
        data_path, # testing datasets
        dataset_path,
        num_inputs,
        num_neurons,
        num_samples=100,
        max_length=1600,
        window_size=400,
        window_stride=1,
        device='cpu'):

    reconstruction_loss = torch.nn.MSELoss()

    # Load the trained model
    checkpoint = torch.load(model_ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = CElegansDatasetPlus(
            data_path,
            dataset_path,
            window_stride=window_stride,
            window_size=window_size,
            device=device,
            slices=slice(0, max_length))
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True) # shuffle to get random windows per worm

    # mse_per_dataset = {
    # '2022-06-14-13': {
    #   0: [mse1, mse2, ..., mse100] # column 0 reconstruction errors
    #   1: [mse1, mse2, ..., mse100] # column 1 reconstruction errors
    #   2: [mse1, mse2, ..., mse100]} # column 2 reconstruction errors
    # }
    mse_per_dataset = {ds: {col: [] for col in range(num_inputs)}
                       for ds in dataset.datasets}

    for _, (inputs, worm, start_frame, end_frame, ds) in enumerate(dataloader):

        if len(mse_per_dataset[ds[0]][0]) >= num_samples:
            break

        worm_idx = worm.item()
        targets = deepcopy(inputs)

        # ensure that masked data all have ground truth to compute MSE
        ground_truth_available = True
        for i in range(num_neurons):
            if (torch.max(targets[0, i, :]).item() == 0 and
                torch.min(targets[0, i, :]).item() == 0):
                ground_truth_available = False
                break
        if not ground_truth_available:
            break

        inputs[:, mask_indices, :] = 0
        outputs = model(inputs)
        for i in range(num_inputs):
            mse_per_dataset[ds[0]][i].append(
                    reconstruction_loss(targets[0, i, :], outputs[0, i, :]))

    return mse_per_dataset


if __name__ == "__main__":

    device = 'cuda:2'

    ds_name = 'AVA_MC_SMDV'
    num_neurons = 3
    depth = 5
    base = '/home/alicia/notebook/alicia/worm-dynamics'
    in_channels = np.load(f'{base}/data/{ds_name}.npy').shape[1]
    out_channels = in_channels
    num_inputs = in_channels
    model = UNet(depth, in_channels, out_channels, unet_dim=1).to(device)
    exp = 'exp_2024100501'
    ckpt_dir = f'/home/alicia/store1/alicia/attention_predict/{exp}/checkpoints'
    model_ckpt_path = f'{ckpt_dir}/model_ckpt100.pt'

    # AVA | MC | SMDV | velocity | pumping | head angle
    mask_indices = [0, 2]
    data_path = f'{base}/data/{ds_name}_test.npy'
    dataset_path = f'{base}/data/{ds_name}_test_ds.npy'

    evaluate(
        model,
        model_ckpt_path,
        mask_indices,
        data_path,
        dataset_path,
        num_inputs,
        num_neurons)
