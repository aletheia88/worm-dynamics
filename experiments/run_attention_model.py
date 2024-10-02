from attention_predict.dataset import CElegansDataset
from attention_predict.model import PredictModel
from tqdm import tqdm
import json
import numpy as np
import torch


def train(
        training_dataset,
        validation_dataset,
        model,
        batch_size,
        num_iterations,
        num_epochs,
        learning_rate,
        log_directory,
        log_ckpt_freq=None,
        log_loss=False,
        log_trace=False,
        last_epoch=None
):
    ckpt_directory = f'{log_directory}/checkpoints'
    train_directory = f'{log_directory}/train'
    valid_directory = f'{log_directory}/validation'

    ensure_dir_exists([ckpt_directory, train_directory, valid_directory])

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=False)

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False)

    reconstruction_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if last_epoch is None:
        loss_dict = {'training': [], 'validation': []}
    else:
        with open(f'{log_directory}/losses.json', 'r') as f:
            loss_dict = json.load(f)
        optimizer.load_state_dict(checkpoint['optimizer'])

    left_slider = 0
    right_slider = 0

    reconstructed_training_traces = {
            worm: {
                'ground_truth': [],
                'prediction': [],
                'attention': [],
                'frames': [],
                'mse': []
            } for worm in range(num_worms)
        }

    for n_epoch in tqdm(range(num_epochs)):

        train_loss = 0

        for n_iter, (inputs, worm, start_frame, end_frame) in \
            enumerate(training_dataloader):

            if n_iter == num_iterations:
                break

            optimizer.zero_grad()
            outputs, weights = model(inputs)
            loss = reconstruction_loss(inputs, outputs)
            loss.backward()
            optimizer.step()

            outputs, weights = model(inputs)
            loss = reconstruction_loss(inputs, outputs)
            train_loss += loss.item()

            append = False
            idx = worm.item()

            if log_trace and n_epoch % 100 == 0:
                if start_frame.item() == right_slider:
                    append = True
                    left_slider = right_slider
                    right_slider = left_slider + window_size

                if start_frame.item() == max_length - window_size - 1:
                    append = True
                    # reset the slider positions to append data from next worm
                    left_slider = 0
                    right_slider = 0

                if append:
                    reconstructed_training_traces[idx]['frames'].append((start_frame.item(),
                                                                end_frame.item()))
                    reconstructed_training_traces[idx]['ground_truth'].append(inputs.cpu().detach().numpy())
                    reconstructed_training_traces[idx]['prediction'].append(outputs.cpu().detach().numpy())
                    reconstructed_training_traces[idx]['attention'].append(weights.cpu().detach().numpy())
                    reconstructed_training_traces[idx]['mse'].append(loss.item())

                reconstructed_validation_traces = validate(
                        model,
                        validation_dataloader,
                        optimizer,
                        reconstruction_loss,
                        num_iterations)

                write_reconstruction_to_npy(reconstructed_validation_traces)
                write_reconstruction_to_npy(reconstructed_training_traces)

        loss_dict['training'].append(np.mean(train_loss))

        if log_loss:
            with open(f'{log_directory}/losses.json', 'w') as f:
                json.dump(loss_dict, f, indent=4)

        if log_ckpt_freq is not None and n_epoch % log_ckpt_freq == 0:

            if last_epoch is not None:
                current_epoch = last_epoch + n_epoch
            else:
                current_epoch = n_epoch

            torch.save(
                {
                    'epoch': current_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss.item()
                },
                    f'{log_directory}/model_ckpt{current_epoch}.pt'
            )


def write_reconstruction_to_npy(reconstructed_traces):
    # concatenate the traces
    worm_indices = list(reconstructed_traces.keys())
    for worm_idx in worm_indices:

        concatenated_prediction = np.concatenate(
                reconstructed_traces[worm_idx]['prediction'],
                axis=2)
        concatenated_ground_truth = np.concatenate(
                reconstructed_traces[worm_idx]['ground_truth'],
                axis=2)

    pass


@torch.no_grad()
def validate(
        model,
        validation_dataloader,
        optimizer,
        reconstruction_loss,
        num_iterations
):
    model.eval()

    left_slider = 0
    right_slider = 0

    reconstructed_validation_traces = {
            worm: {
                'ground_truth': [],
                'prediction': [],
                'attention': [],
                'frames': [],
                'mse': []
            } for worm in range(num_worms)
        }

    for i, (inputs, worm, start_frame, end_frame) in enumerate(validation_dataloader):

        if i == num_iterations:
            break

        append = False
        idx = worm.item()

        optimizer.zero_grad()
        outputs, weights = model(inputs)
        loss = reconstruction_loss(inputs, outputs)

        if start_frame.item() == right_slider:
            append = True
            left_slider = right_slider
            right_slider = left_slider + window_size

        if start_frame.item() == max_length - window_size - 1:
            append = True
            # reset the slider positions to append data from next worm
            left_slider = 0
            right_slider = 0

        if append:
            reconstructed_validation_traces[idx]['frames'].append((start_frame.item(),
                                                        end_frame.item()))
            reconstructed_validation_traces[idx]['ground_truth'].append(inputs.cpu().detach().numpy())
            reconstructed_validation_traces[idx]['prediction'].append(outputs.cpu().detach().numpy())
            reconstructed_validation_traces[idx]['attention'].append(weights.cpu().detach().numpy())
            reconstructed_validation_traces[idx]['mse'].append(loss.item())

    model.train()

    return reconstructed_validation_traces


def ensure_dir_exists(directories):
    import os
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


if __name__ == "__main__":

    device = "cuda:2"
    window_size = 50
    ds_name = 'AVA_MC'
    batch_size = 32
    embedding_dims = 1024
    num_layers = 4
    num_iterations = 1_000_000
    num_epochs = 50001
    learning_rate = 1e-4

    log_directory = 'exp_2024091702'
    log_ckpt_freq = 100
    log_trace_freq = None
    log_loss = True
    estimate_loss = False
    last_epoch = None

    training_dataset = CElegansDataset(
        f'../data/{ds_name}_train.npy',
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )
    validation_dataset = CElegansDataset(
        f'../data/{ds_name}_valid.npy',
        window_size=window_size,
        device=device,
        slices=slice(0, 1600)
    )

    model = PredictModel(
        num_inputs=training_dataset.num_variables,
        input_dims=window_size,
        embedding_dims=embedding_dims,
        num_layers=num_layers,
        residual=True,
        normalize=True,
        device=device
    ).to(device)

    if last_epoch is not None:
        model_ckpt_path = f'{log_directory}/model_ckpt{last_epoch}.pt'
        checkpoint = torch.load(model_ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.train()

    train(
        training_dataset,
        validation_dataset,
        model,
        batch_size,
        num_iterations=num_iterations,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        log_directory=log_directory,
        log_ckpt_freq=log_ckpt_freq,
        log_loss=log_loss,
        estimate_loss=estimate_loss,
        last_epoch=last_epoch
    )
