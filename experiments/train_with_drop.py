from tqdm import tqdm
import attention_predict
import json
import numpy as np
import torch
import yaml


def train(
    attention_scheme,
    num_epochs,
    num_iterations,
    ds_name,
    batch_size,
    depth,
    num_neurons,
    num_behaviors,
    num_fmaps,
    num_masked_neurons,
    learning_rate,
    log_directory,
    log_ckpt_freq,
    ckpt_path,
    num_drop,
    device
):
    num_inputs = num_neurons + num_behaviors
    inputs = torch.zeros((batch_size, num_inputs, 400))
    neuron_indices = list(range(num_neurons))
    # indices of behavior variables in model inputs
    behavior_indices = list(range(num_neurons, num_inputs))

    def mask_out_neurons(num_masked_neurons, recorded_neuron_indices):

        # masking neurons during training only applies to BfromN and NfromN
        # inputs: (num_samples, num_inputs, 400)
        masked_neuron_indices = torch.randperm(
                num_neurons)[:num_masked_neurons]
        nonlocal inputs
        inputs = inputs[:, masked_neuron_indices, :] = -10
        return inputs

    base = '/home/alicia/store1/alicia/attention_predict'
    training_dataset = attention_predict.dataset.CElegansDatasetPlus(
        f'{base}/data/{ds_name}_train.npy',
        f'{base}/data/{ds_name}_train_ds.npy',
        window_stride=1,
        window_size=400,
        device=device,
        slices=slice(0, 1600)
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    model = attention_predict.AttentionModelMini(
        depth,
        num_neurons,
        num_behaviors,
        attention_scheme,
        window_size=400,
        num_fmaps=num_fmaps,
        device=device
    )
    # load model weights from a previous checkpoint
    last_ckpt = 0
    if ckpt_path != "":
        last_ckpt = int(ckpt_path.split('ckpt')[1].split('.pt')[0])
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print('checkpoint loaded!')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss()

    if ckpt_path == "":
        loss_dict = {'training': [], 'validation': []}
    else:
        with open(f'{log_directory}/losses.json', 'r') as f:
            loss_dict = json.load(f)

    # define input and target indices based on task
    if attention_scheme == 'BfromN':
        input_indices = neuron_indices
        target_indices = behavior_indices
    elif attention_scheme == 'NfromN':
        input_indices = neuron_indices
        target_indices = neuron_indices
    elif attention_scheme == 'NfromB':
        input_indices = behavior_indices
        target_indices = neuron_indices

    for n_epoch in tqdm(range(num_epochs)):

        training_loss = []
        for n_iter, (inputs, _, _, _, _) in tqdm(
                enumerate(training_dataloader)):

            if n_iter == num_iterations:
                break

            targets = inputs[:, target_indices, :]
            # TODO implement dropping neurons
            drop_indices = np.random.choice(neuron_indices, num_drop)
            augmented_inputs = inputs[:, input_indices, :]
            augmented_inputs[:, drop_indices, :] = -10

            outputs, attention_weights = model(augmented_inputs)

            if attention_scheme in ['BfromN', 'BfromN']:
                recorded_neuron_indices = None
            elif attention_scheme in ['NfromB', 'NfromN']:
                recorded_neuron_indices = get_recorded_neurons(targets, num_neurons)

            # average loss across samples and variables of reconstruction
            loss = aggregate_loss(
                targets,
                outputs,
                mse_loss,
                num_neurons,
                attention_scheme,
                recorded_neuron_indices
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_epoch % log_ckpt_freq == 0:
                training_loss.append(loss.item())

        if n_epoch % log_ckpt_freq == 0:

            loss_dict['training'].append(np.mean(training_loss))
            with open(f'{log_directory}/losses.json', 'w') as f:
                json.dump(loss_dict, f, indent=4)

            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'attention': attention_weights,
                    'optimizer': optimizer.state_dict(),
                },
                f'{log_directory}/model_ckpt{last_ckpt+n_epoch+1}.pt'
            )


def seed_everything(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # # configure cudNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir_exists(directories):
    import os
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def aggregate_loss(
    targets,
    outputs,
    mse_loss,
    num_neurons,
    attention_scheme,
    recorded_neuron_indices,
):
    if attention_scheme in ['BfromN', 'BfromB']:
        return mse_loss(targets, outputs)

    if attention_scheme in ['NfromB', 'NfromN']:

        if recorded_neuron_indices is None:
            raise ValueError(
                    'Function input recorded_neurons_indices cannot be None.'
            )
        # loss_mask shape: (num_samples, num_neurons, window_size)
        num_samples, _, window_size = targets.shape
        loss_mask = torch.zeros(
                (num_samples, num_neurons, window_size),
                dtype=bool)

        for n, loss_indices in recorded_neuron_indices.items():
            loss_mask[n, loss_indices, :] = True

        return mse_loss(targets[loss_mask], outputs[loss_mask])


def get_recorded_neurons(targets, num_neurons):

    num_samples = targets.shape[0]
    recorded_neuron_indices = {}

    for n in range(num_samples):
        recorded_neuron_indices[n] = [
            i for i in range(num_neurons)
            if (torch.max(targets[n, i, :]).item() != -10
            and torch.min(targets[n, i, :]).item() != -10)
        ]

    return recorded_neuron_indices


if __name__ == '__main__':

    attention_scheme = 'NfromB'
    device = "cuda:0"
    num_epochs = 201
    num_iterations = 1000
    ds_name = 'data0410_norm0505'
    batch_size = 32
    depth = 5
    num_neurons = 70
    num_behaviors = 4
    num_fmaps = 64
    num_masked_neurons = 0
    learning_rate = 5e-5
    random_seed = 2025
    log_ckpt_freq = 2
    num_drop = 1

    config_dict = {
        'fitting': {
            'ds_name': ds_name,
            'seed': random_seed,
            'batch_size': batch_size,
            'attn_scheme': attention_scheme,
            'num_epochs': num_epochs,
            'num_iters': num_iterations,
            'num_neurons': num_neurons,
            'num_behaviors': num_behaviors,
            'seed': random_seed,
            'learning_rate': learning_rate,
            'num_drop': num_drop
        },
        'model': {
            'depth': depth,
            'num_fmaps': num_fmaps,
        }
    }
    lr = f"{float(learning_rate):.0e}"
    max_num_drop = 0
    experiment = f"depth{depth}_nfmaps{num_fmaps}_lr{lr}_{attention_scheme}_norm0505_drop{num_drop}"
    base = '/store1/alicia/attention_predict/whole_brain'
    experiment_path = f'{base}/{experiment}'
    ensure_dir_exists([experiment_path])
    n_ckpt = 201
    ckpt_path = f'{experiment_path}/model_ckpt{n_ckpt}.pt'
    # ckpt_path = ""

    with open(f'{experiment_path}/config.yaml', 'w') as f:
        yaml.dump(config_dict, f)

    seed_everything(random_seed)
    train(
        attention_scheme,
        num_epochs,
        num_iterations,
        ds_name,
        batch_size,
        depth,
        num_neurons,
        num_behaviors,
        num_fmaps,
        num_masked_neurons,
        learning_rate,
        experiment_path,
        log_ckpt_freq,
        ckpt_path,
        num_drop,
        device
    )
