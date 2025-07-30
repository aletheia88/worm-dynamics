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
    device
):
    num_inputs = num_neurons + num_behaviors
    # inputs = torch.zeros((batch_size, num_inputs, 400))

    # indices of variables in model inputs
    # neuron_indices = list(range(num_neurons))
    # behavior_indices = list(range(num_neurons, num_inputs))
    # indices of variables in `data0612`
    neuron_indices = list(range(num_neurons * 2))
    behavior_indices = list(range(num_neurons * 2, num_inputs * 2))

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
        f'{base}/data/{ds_name}_train_shuffle2.npy',
        f'{base}/data/{ds_name}_train_ds_shuffle2.npy',
        window_stride=20,
        window_size=400,
        device=device,
        slices=slice(0, 1600)
    )
    # sample_weights = attention_predict.dataset._assign_sampling_weights(
    #         training_dataset)
    # sampler = torch.utils.data.WeightedRandomSampler(
    #     weights=torch.as_tensor(sample_weights),
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        # sampler=sampler,
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
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()

    if ckpt_path == "":
        loss_dict = {'training': [], 'validation': []}
    else:
        with open(f'{log_directory}/losses.json', 'r') as f:
            loss_dict = json.load(f)

    # define input and target indices based on task
    if attention_scheme == 'BfromN':
        input_indices = neuron_indices
        target_indices = behavior_indices
    if attention_scheme in [
        'NfromN',
        'connectome',
        'anticonnectome',
        'randconnectome'
    ]:
        input_indices = neuron_indices
        target_indices = neuron_indices
    if attention_scheme == 'NfromB':
        input_indices = behavior_indices
        target_indices = neuron_indices

    for n_epoch in tqdm(range(num_epochs)):

        training_loss = []
        for n_iter, (inputs, _, _, _, _) in enumerate(training_dataloader):

            if n_iter == num_iterations:
                break

            outputs = model(inputs[:, input_indices, :])
            # targets: (num_samples, num_variables * 2, window_size)
            targets = inputs[:, target_indices, :]

            if attention_scheme in ['BfromN', 'BfromN']:
                recorded_neuron_indices = None
            elif attention_scheme in [
                    'NfromB', 'NfromN',
                    'connectome', 'anticonnectome', 'randconnectome']:
                # recorded_neuron_indices = _get_recorded_neurons(
                #     targets, num_neurons)
                recorded_neuron_indices = _get_recorded_neurons(targets)

            # average loss across samples and variables of reconstruction
            loss = _aggregate_loss(
                targets,
                outputs,
                criterion,
                num_neurons,
                attention_scheme,
                recorded_neuron_indices,
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
    criterion,
    num_neurons,
    attention_scheme,
    recorded_neuron_indices,
):
    if attention_scheme in ['BfromN', 'BfromB']:
        return criterion(targets, outputs)

    if attention_scheme in [
            'NfromB', 'NfromN',
            'connectome', 'anticonnectome', 'randconnectome']:

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

        return criterion(outputs[loss_mask], targets[loss_mask])


def _aggregate_loss(
    targets,
    outputs,
    criterion,
    num_neurons,
    attention_scheme,
    recorded_neuron_indices,
):
    # ouputs: (num_samples, num_variables, window_size)
    # targets: (num_samples, num_variables * 2, window_size)
    num_samples, num_vecs, window_size = targets.shape
    num_variables = num_vecs // 2

    if attention_scheme == 'BfromN':
        # E.g.,
        # target indices: 0, 1 | 2, 3 | 4, 5
        # where 0, 2, 4 index into acitvity recordings
        # output indices: 0, 1, 2
        loss_indices = list(range(0, num_vecs, 2))
        return criterion(outputs, targets[:, loss_indices, :])

    if attention_scheme in ['NfromN', 'NfromB', 'connectome', 'anticonnectome']:
        if recorded_neuron_indices is None:
            raise ValueError(
                    'Function input recorded_neurons_indices cannot be None.'
            )
        target_loss_mask = torch.zeros(
                (num_samples, num_vecs, window_size),
                dtype=bool)
        output_loss_mask = torch.zeros(
                (num_samples, num_variables, window_size),
                dtype=bool)

        for n, loss_indices in recorded_neuron_indices.items():
            target_loss_mask[n, loss_indices, 20:380] = True
            output_loss_mask[n, [i//2 for i in loss_indices], 20:380] = True

        return criterion(outputs[output_loss_mask], targets[target_loss_mask])


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


def _get_recorded_neurons(targets):

    recorded_neuron_indices = {}
    num_samples, num_vecs, _ = targets.shape
    # num_vecs = num variables to be predicted * 2

    for n in range(num_samples):
        recorded_neuron_indices[n] = [
            i - 1
            for i in range(1, num_vecs, 2)
            if targets[n, i, 0] == 1
        ]

    return recorded_neuron_indices


if __name__ == '__main__':

    attention_scheme = 'NfromN'
    device = "cuda:0"
    num_epochs = 91
    num_iterations = 1000
    ds_name = f'data0626/data0626-00e_norm0505'
    # ds_name = 'data0715e_norm0505'
    batch_size = 1
    depth = 5
    num_neurons = 70
    num_behaviors = 4
    num_fmaps = 64
    num_masked_neurons = 0
    learning_rate = 5e-5
    random_seed = 2025
    log_ckpt_freq = 2

    config_dict = {
        'fitting': {
            'ds_name': ds_name + '_shuffle2',
            'seed': random_seed,
            'batch_size': batch_size,
            'attn_scheme': attention_scheme,
            'num_epochs': num_epochs,
            'num_iters': num_iterations,
            'num_neurons': num_neurons,
            'num_behaviors': num_behaviors,
            'learning_rate': learning_rate,
        },
        'model': {
            'depth': depth,
            'num_fmaps': num_fmaps,
            'groups_level0': 1
        }
    }
    lr = f"{float(learning_rate):.0e}"
    max_num_drop = 0
    # currbest := '2ch_s20_fixedattn-hard'
    experiment = \
        f"depth{depth}_nfmaps{num_fmaps}_lr{lr}_{attention_scheme}_norm0505_currbest_fix_shuffle2"
    base = '/store1/alicia/attention_predict/whole_brain'
    experiment_path = f'{base}/{experiment}'
    ensure_dir_exists([experiment_path])
    # n_ckpt = 201
    # ckpt_path = f'{experiment_path}/model_ckpt{n_ckpt}.pt'
    ckpt_path = ""

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
        device
    )
