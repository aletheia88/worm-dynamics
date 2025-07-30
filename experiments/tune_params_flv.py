from sklearn.model_selection import KFold
from tqdm import tqdm
import attention_predict
import json
import numpy as np
import os
import torch
import yaml


def seed_everything(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # # configure cudNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_model(
    experiment_path,
    config_dict,
    max_num_drop,
    seed,
    ckpt_path,
    device
):
    """
    This function fits model on all training data available.
    There is NO validation.
    """

    fit_params = config_dict['fitting']
    ds_name = fit_params['ds_name']
    batch_size = fit_params['batch_size']
    num_neurons = fit_params['num_neurons']
    num_behaviors = fit_params['num_behaviors']
    attn_scheme = fit_params['attn_scheme']
    learning_rate = fit_params['learning_rate']
    num_epochs = fit_params['num_epochs']
    num_iterations = fit_params['num_iters']

    model_params = config_dict['model']
    depth = model_params['depth']
    num_fmaps = model_params['num_fmaps']

    base = '/home/alicia/store1/alicia/attention_predict'
    training_dataset = attention_predict.dataset.CElegansDatasetPlus(
        f'{base}/data/{ds_name}_train.npy',
        f'{base}/data/{ds_name}_train_ds.npy',
        window_stride=1,
        window_size=400,
        device=device,
        slices=slice(0, 1600)
    )
    validation_dataset = attention_predict.dataset.CElegansDatasetPlus(
        f'{base}/data/{ds_name}_valid.npy',
        f'{base}/data/{ds_name}_valid_ds.npy',
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
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    neuron_indices = list(range(num_neurons))
    behavior_indices = list(range(num_neurons, num_neurons + num_behaviors))
    reconstruction_loss = torch.nn.MSELoss()

    # define input and target indices based on task
    if attn_scheme == 'BfromN':
        input_indices = neuron_indices
        target_indices = behavior_indices
    elif attn_scheme == 'NfromN':
        input_indices = neuron_indices
        target_indices = neuron_indices
    elif attn_scheme == 'NfromB':
        input_indices = behavior_indices
        target_indices = neuron_indices

    # either load an existing model or initialize a new model
    model = attention_predict.AttentionModelMini(
        depth,
        num_neurons,
        num_behaviors,
        attn_scheme,
        400,
        num_fmaps,
        device=device
    )
    last_ckpt = 0
    if ckpt_path != "":
        last_ckpt = int(ckpt_path.split('ckpt')[1].split('.pt')[0])
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print('checkpoint loaded!')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)
    model, optimizer = train(
        model,
        optimizer,
        reconstruction_loss,
        attn_scheme,
        num_epochs,
        num_iterations,
        num_neurons,
        training_dataloader,
        validation_dataloader,
        input_indices,
        target_indices,
        experiment_path,
        last_ckpt,
        max_num_drop,
        seed,
    )
    # save the checkpoint after the last epoch
    save_model_ckpt(
        model,
        optimizer,
        experiment_path,
        last_ckpt+num_epochs
    )


def cross_validate_model(
    experiment_path,
    config_dict,
    num_drop,
    seed,
    device
):
    """
    This function splits all samples into 5 folds of different training and validation
    datasets. Train and validate model on each fold.
    """

    fit_params = config_dict['fitting']
    ds_name = fit_params['ds_name']
    k_folds = fit_params['k_folds']
    batch_size = fit_params['batch_size']
    num_neurons = fit_params['num_neurons']
    num_behaviors = fit_params['num_behaviors']
    random_seed = fit_params['seed']
    attn_scheme = fit_params['attn_scheme']
    learning_rate = fit_params['learning_rate']
    num_epochs = fit_params['num_epochs']
    num_iterations = fit_params['num_iters']

    model_params = config_dict['model']
    depth = model_params['depth']
    num_fmaps = model_params['num_fmaps']

    def build_model():
        return attention_predict.AttentionModelMini(
            depth,
            num_neurons,
            num_behaviors,
            attn_scheme,
            400,
            num_fmaps,
            device=device
        )

    base = '/home/alicia/store1/alicia/attention_predict'
    training_dataset = attention_predict.dataset.CElegansDatasetPlus(
        f'{base}/data/{ds_name}_train.npy',
        f'{base}/data/{ds_name}_train_ds.npy',
        window_stride=1,
        window_size=400,
        device=device,
        slices=slice(0, 1600)
    )
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    neuron_indices = list(range(num_neurons))
    behavior_indices = list(range(num_neurons, num_neurons + num_behaviors))
    reconstruction_loss = torch.nn.MSELoss()

    # define input and target indices based on task specified by the attention scheme
    if attn_scheme == 'BfromN':
        input_indices = neuron_indices
        target_indices = behavior_indices
    elif attn_scheme == 'NfromN':
        input_indices = neuron_indices
        target_indices = neuron_indices
    elif attn_scheme == 'NfromB':
        input_indices = behavior_indices
        target_indices = neuron_indices

    for fold, (training_ids, validation_ids) in enumerate(kfold.split(training_dataset)):

        print(f'-----Fold: {fold}-----')
        training_subsampler = torch.utils.data.Subset(training_dataset, training_ids)
        validation_subsampler = torch.utils.data.Subset(training_dataset, validation_ids)

        training_dataloader = torch.utils.data.DataLoader(training_subsampler,
                                                          batch_size=batch_size,
                                                          shuffle=True)
        validation_dataloader = torch.utils.data.DataLoader(validation_subsampler,
                                                        batch_size=batch_size,
                                                        shuffle=False)
        # Initiate model and optimize for each training-validation split
        model = build_model()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate)
        log_directory = f'{experiment_path}/fold{fold}'
        ensure_dir_exists([log_directory])
        model, optimizer = train(
            model,
            optimizer,
            reconstruction_loss,
            attn_scheme,
            num_epochs,
            num_iterations,
            num_neurons,
            training_dataloader,
            validation_dataloader,
            input_indices,
            target_indices,
            log_directory,
            num_drop,
            seed
        )
        # Save the checkpoint after the last epoch
        save_model_ckpt(model, optimizer, log_directory, num_epochs)


def ensure_dir_exists(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def train(
    model,
    optimizer,
    reconstruction_loss,
    attn_scheme,
    num_epochs,
    num_iterations,
    num_neurons,
    training_dataloader,
    validation_dataloader,
    input_indices,
    target_indices,
    log_directory,
    last_ckpt,
    max_num_drop,
    seed,
):

    def estimate_loss():

        model.eval()
        validation_loss = []

        with torch.no_grad():
            for n_iter, (inputs, _, _, _, _) in tqdm(enumerate(validation_dataloader)):
                outputs, attention_weights = model(inputs[:, input_indices, :])
                targets = inputs[:, target_indices, :]
                loss = aggregate_loss(
                    outputs,
                    targets,
                    reconstruction_loss,
                    num_neurons,
                    attn_scheme,
                    None
                )
                validation_loss.append(loss.item())

        return np.mean(validation_loss)

    model.train()
    loss_dict = {'training': [], 'validation': []}

    # set numpy random seed for dropping random neuron(s)
    # if max_num_drop > 0:
    #     np.random.seed(seed)

    for n_epoch in tqdm(range(num_epochs)):

        training_loss = []
        for n_iter, (inputs, _, _, _, _) in enumerate(training_dataloader):

            if n_iter == num_iterations:
                break

            optimizer.zero_grad()

            # data augmentation by randomly dropping neurons
            if max_num_drop > 0:
                num_drop = np.random.randint(0, max_num_drop + 1)
                drop_indices = np.random.choice(
                        input_indices,
                        size=num_drop,
                        replace=False
                )
                # missing variables set to -10
                inputs[:, drop_indices, :] = -10

            outputs, attention_weights = model(inputs[:, input_indices, :])
            targets = inputs[:, target_indices, :]

            if attn_scheme in ['BfromN', 'BfromN']:
                recorded_neuron_indices = None
            elif attn_scheme in ['NfromB', 'NfromN']:
                recorded_neuron_indices = get_recorded_neurons(
                        targets,
                        num_neurons)

            loss = aggregate_loss(
                targets,
                outputs,
                reconstruction_loss,
                num_neurons,
                attn_scheme,
                recorded_neuron_indices
            )
            loss.backward()
            optimizer.step()
            if n_epoch % 2 == 0:
                training_loss.append(loss.item())

        # log weights and losses every 10 epochs
        if n_epoch % 2 == 0:
            loss_dict['training'].append(np.mean(training_loss))
            # validation_loss = estimate_loss()
            # model.train()
            # loss_dict['validation'].append(validation_loss)

            with open(f'{log_directory}/losses.json', 'w') as f:
                json.dump(loss_dict, f, indent=4)

            save_model_ckpt(
                model,
                optimizer,
                log_directory,
                last_ckpt+n_epoch+1
            )

    return model, optimizer


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


def aggregate_loss(
    targets,
    outputs,
    reconstruction_loss,
    num_neurons,
    attn_scheme,
    recorded_neuron_indices,
):
    if attn_scheme in ['BfromN', 'BfromB']:
        return reconstruction_loss(targets, outputs)

    elif attn_scheme in ['NfromB', 'NfromN']:

        if recorded_neuron_indices == None:
            raise ValueError(
                    'Function input recorded_neurons_indices cannot be None.')
        # loss_mask shape: (num_samples, num_neurons, window_size)
        num_samples, _, window_size = targets.shape
        loss_mask = torch.zeros(
                (num_samples, num_neurons, window_size),
                dtype=bool)
        for n, loss_indices in recorded_neuron_indices.items():
            loss_mask[n, loss_indices, :] = True

        return reconstruction_loss(targets[loss_mask], outputs[loss_mask])


def save_model_ckpt(model, optimizer, log_directory, n_epoch):

    torch.save(
        {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        f'{log_directory}/model_ckpt{n_epoch}.pt'
    )


def create_params_grid():

    depths = [3, 4, 5]
    num_fmaps_options = [8, 16, 32, 64, 128]
    learning_rates = np.linspace(1e-5, 1e-4, 10).tolist()
    params_grid = {}

    i = 0
    for depth in depths:
        for num_fmaps in num_fmaps_options:
            for learning_rate in learning_rates:
                params_grid[i] = {
                    'depth': depth,
                    'num_fmaps': num_fmaps,
                    'learning_rate': f'{learning_rate:.0e}'
                }
                i += 1

    return params_grid


def main():

    random_seed = 2025
    seed_everything(random_seed)
    ds_name = 'data0428_norm'

    k_folds = 1
    batch_size = 32
    # 74 variables in total: 70 neurons + 1 heat-stim + 3 beh
    num_neurons = 18
    num_behaviors = 3
    attn_scheme = 'BfromN'
    learning_rate = 5e-5

    device = 'cuda:2'
    num_epochs = 200
    num_iterations = 1000
    depth = 5
    num_fmaps = 64

    config_dict = {
        'fitting': {
            'ds_name': ds_name,
            'seed': random_seed,
            'batch_size': batch_size,
            'attn_scheme': attn_scheme,
            'num_epochs': num_epochs,
            'num_iters': num_iterations,
            'k_folds': k_folds,
            'num_neurons': num_neurons,
            'num_behaviors': num_behaviors,
            'seed': random_seed,
            'learning_rate': learning_rate
        },
        'model': {
            'depth': depth,
            'num_fmaps': num_fmaps,
        }
    }
    lr = f"{float(learning_rate):.0e}"
    max_num_drop = 0
    experiment = f"depth{depth}_nfmaps{num_fmaps}_lr{lr}"
    base = '/store1/alicia/attention_predict/neurons17'
    experiment_path = f'{base}/{experiment}'
    ensure_dir_exists([experiment_path])

    with open(f'{experiment_path}/config.yaml', 'w') as f:
        yaml.dump(config_dict, f)

    # cross_validate_model(experiment_path, config_dict, device)
    ckpt_path = ""
    import time
    start_time = time.time()
    fit_model(
        experiment_path,
        config_dict,
        max_num_drop,
        random_seed,
        ckpt_path,
        device
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == '__main__':
    # import pprint
    # params_grid = create_params_grid()
    # pprint.pp(params_grid)
    # import os
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
