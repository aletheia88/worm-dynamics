from assemble import assemble_all
from copy import deepcopy
import numpy as np


def write_raw_data(ds_name, neuron_classes, behavior_index_dict):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    assembled_data, datasets = assemble_all(neuron_classes, behavior_index_dict)
    np.save(f'{data_dir}/{ds_name}_raw.npy', assembled_data)
    np.save(f'{data_dir}/{ds_name}_raw_ds.npy', datasets)
    print(f'dataset {ds_name}_raw saved!')


def split_train_valid_test(
    ds_name,
    split_ratio=[0.7, 0.2, 0.1],
    random_seed=42,
    save=True
):
    data_dir = '/home/alicia/store1/alicia/attention_predict/data'

    data = np.load(f'{data_dir}/{ds_name}_raw.npy')
    datasets = np.load(f'{data_dir}/{ds_name}_raw_ds.npy')
    num_datasets = data.shape[0]

    train_size = int(num_datasets * split_ratio[0])
    valid_size = int(num_datasets * split_ratio[1])
    test_size = num_datasets - train_size - valid_size
    print(f'Splits: {train_size, valid_size, test_size}')

    np.random.seed(random_seed)
    indices = np.random.choice(num_datasets, num_datasets, replace=False)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size+valid_size]
    test_indices = indices[-test_size:]

    train_set = data[train_indices, :, :]
    valid_set = data[valid_indices, :, :]
    test_set = data[test_indices, :, :]

    train_datasets = datasets[train_indices]
    valid_datasets = datasets[valid_indices]
    test_datasets = datasets[test_indices]

    if save:
        np.save(f'{data_dir}/{ds_name}_raw_train.npy', train_set)
        np.save(f'{data_dir}/{ds_name}_raw_valid.npy', valid_set)
        np.save(f'{data_dir}/{ds_name}_raw_test.npy', test_set)
        # Also save the corresponding dataset names
        np.save(f'{data_dir}/{ds_name}_raw_train_ds.npy', train_datasets)
        np.save(f'{data_dir}/{ds_name}_raw_valid_ds.npy', valid_datasets)
        np.save(f'{data_dir}/{ds_name}_raw_test_ds.npy', test_datasets)
        print(f'Train/Valid/Test splits for {ds_name}_raw saved!')
    else:
        return train_set, valid_set, test_set


def write_eval_data(ds_name):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    # find unique datasets from validation and testing that not in training
    # get for their indices in `valid_ds` and `test_ds`
    # use these indices to get traces for each datasets
    # lastly assemble all the datasets
    train_ds = np.load(f'{data_dir}/{ds_name}_raw_train_ds.npy')
    valid_ds = np.load(f'{data_dir}/{ds_name}_raw_valid_ds.npy')
    test_ds = np.load(f'{data_dir}/{ds_name}_raw_test_ds.npy')

    valid_data = np.load(f'{data_dir}/{ds_name}_raw_valid.npy')
    test_data = np.load(f'{data_dir}/{ds_name}_raw_test.npy')

    unique_valid = list(set(valid_ds).difference(train_ds))
    unique_test = list(set(test_ds).difference(train_ds))

    eval_datasets = np.unique(unique_valid + unique_valid).tolist()
    np.save(f'{data_dir}/{ds_name}_raw_eval_ds.npy', eval_datasets)

    # initialize eval data
    _, num_inputs, num_frames = valid_data.shape
    eval_data = np.zeros((len(eval_datasets), num_inputs, num_frames))

    for i, eval_ds in enumerate(eval_datasets):
        if eval_ds in unique_valid:
            ds_index = valid_ds.tolist().index(eval_ds)
            eval_data[i, :, :] = valid_data[ds_index, :, :]
        elif eval_ds in unique_test:
            ds_index = test_ds.tolist().index(eval_ds)
            eval_data[i, :, :] = test_data[ds_index, :, :]
        else:
            print(f'WARNING: {eval_ds} is not found.')

    np.save(f'{data_dir}/{ds_name}_raw_eval.npy', eval_data)
    print(f'Eval data/datasets for {ds_name}_raw_eval are saved!')


def write_control_data(
    ds_name,
    num_animals,
    num_columns,
    length=1600,
    control_type='white_noise',
    deleted_columns=[],
):
    output_path = "/home/alicia/store1/alicia/attention_predict/data"
    if control_type == 'white_noise':
        white_noise = np.random.uniform(-1, 1, (num_animals, num_columns, length))
        np.save(f'{output_path}/{ds_name}.npy', white_noise)
        np.save(f'{output_path}/{ds_name}_ds.npy', ['uncorr_noise'] * num_animals)

    elif control_type == 'zeros':
        zeros = np.zeros((num_animals, num_columns, length))
        np.save(f'{output_path}/{ds_name}.npy', zeros)
        np.save(f'{output_path}/{ds_name}_ds.npy', ['zeros'] * num_animals)

    elif control_type == 'deletion':
        data = np.load(f'{output_path}/{ds_name}.npy')
        data[:, deleted_columns, :] = 0
        np.save(f'{output_path}/{ds_name}_deletion.npy', data)

    print(f'{ds_name} {control_type} control data saved!')


def write_shuffled_data(ds_name, num_neurons, random_seed, shuffle_type):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    shuffled_data, shuffled_datasets = shuffle(ds_name, num_neurons, random_seed,
                                               shuffle_type=shuffle_type)
    np.save(f'{data_dir}/{ds_name}_shuffle.npy', shuffled_data)
    np.save(f'{data_dir}/{ds_name}_ds_shuffle.npy', shuffled_datasets)
    print(f'Shuffled {ds_name} saved!')


if __name__ == "__main__":

    ds_name = 'steve1230'
    neuron_classes = ['SMDD', 'SAADL', 'SAADR', 'SAAV', 'M3', 'M4', 'MI', 'AVB', 'RIB',
                      'RME', 'RMEV', 'RMED', 'URYD', 'URYV']
    num_neurons = len(neuron_classes)
    num_body_angles = 30
    behavior_index_dict = {
            'velocity': num_neurons,
            'pumping': num_neurons + 1,
            'head_angle': num_neurons + 2,
            'body_angles': list(range(num_neurons+3, num_neurons+3+num_body_angles))
    }
    write_raw_data(ds_name, neuron_classes, behavior_index_dict)
    split_train_valid_test(ds_name, random_seed=2025)
    write_eval_data(ds_name)
