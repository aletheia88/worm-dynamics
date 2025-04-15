from attention_predict.reconstruct_mini import (
    reconstruct_traces,
    build_mini_attention_model,
    build_dataloader,
)
from tqdm import tqdm
import numpy as np
import torch


@torch.no_grad()
def test_model(
    attention_scheme,
    ds_name,
    model_ckpt,
    experiment,
    depth,
    num_neurons,
    num_behaviors,
    num_fmaps,
    variables_to_reconstruct,  # dictionary: {column_index: variable_name}
    device,
):
    window_size = 400
    prj_directory = "/store1/alicia/attention_predict"
    data_path = f"{prj_directory}/data/{ds_name}.npy"
    num_datasets, num_inputs, _ = np.load(data_path).shape

    behavior_indices = list(variables_to_reconstruct.keys())

    model = build_mini_attention_model(
        attention_scheme,
        depth,
        num_neurons,
        num_behaviors,
        window_size,
        num_fmaps,
        device,
    )
    dataloader = build_dataloader(ds_name, device)
    reconstructed_traces = reconstruct_traces(
        model,
        attention_scheme,
        dataloader,
        model_ckpt,
        experiment,
        num_datasets,
        num_neurons,
        num_behaviors,
    )
    reconstruction_error = {}
    mse_loss = torch.nn.MSELoss()

    for ds_index in range(num_datasets):
        reconstruction_error[ds_index] = {}

        prediction = torch.tensor(
            np.concatenate(
                reconstructed_traces[ds_index]["prediction"], axis=2
            ).squeeze(0)
        )
        ground_truth = torch.tensor(
            np.concatenate(
                reconstructed_traces[ds_index]["ground_truth"], axis=2
            ).squeeze(0)
        )
        attn_weights = reconstructed_traces[ds_index]["attn_weights"]
        for i, variable in variables_to_reconstruct.items():
            target = ground_truth[i, :]
            output = prediction[i, :]
            reconstruction_error[ds_index][variable] = mse_loss(target, output).item()

    return reconstruction_error


if __name__ == "__main__":
    attention_scheme = "BfromN"
    ds_name = "sanity_norm_eval"
    model_ckpt = 70
    experiment = "exp_2025030500"
    variables_to_reconstruct = {2: "velocity"}
    device = "cuda:0"
    num_neurons = 2
    num_behaviors = 1
    num_fmaps = 32
    depth = 3

    reconstruction_error = test_model(
        attention_scheme,
        ds_name,
        model_ckpt,
        experiment,
        depth,
        num_neurons,
        num_behaviors,
        num_fmaps,
        variables_to_reconstruct,
        device,
    )
    print(reconstruction_error)
