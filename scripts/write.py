from assemble import assemble_all, assemble_std_beh
from copy import deepcopy
import numpy as np


def write_raw_data(ds_name, neuron_classes, behavior_index_dict):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    ### TODO: someday incorporate back the body angles
    assembled_data, datasets = assemble_std_beh(
            neuron_classes,
            behavior_index_dict
        )
    np.save(f'{data_dir}/{ds_name}_raw.npy', assembled_data)
    np.save(f'{data_dir}/{ds_name}_raw_ds.npy', datasets)
    print(f'dataset {ds_name}_raw saved!')


def split_train_valid_test(
    ds_name,
    random_seed,
    split_ratio=[0.7, 0.2, 0.1],
    save=True
):
    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    data = np.load(f'{data_dir}/{ds_name}_raw.npy')
    datasets = np.load(f'{data_dir}/{ds_name}_raw_ds.npy')
    num_datasets = data.shape[0]

    # train_size = int(num_datasets * split_ratio[0])
    # valid_size = int(num_datasets * split_ratio[1])
    # test_size = num_datasets - train_size - valid_size
    train_size = 64
    valid_size = 16
    test_size = 20
    print(f'Splits: {train_size, valid_size, test_size}')

    np.random.seed(random_seed)
    indices = np.random.choice(num_datasets, num_datasets, replace=False)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
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
        print(f'train, valid, test splits for {ds_name}_raw saved!')
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

    eval_datasets = np.unique(unique_valid + unique_test).tolist()
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


def create_ds_data0108(random_seed):

    ds_name = 'data0108'
    neuron_classes = [
        'SMDV', 'SMDD', 'SAADL', 'SAADR', 'SAAV',
        'MC', 'M3', 'M4', 'MI',
        'AVA', 'AVB', 'RIB',
        'RME', 'RMEV', 'RMED',
        'URYD', 'URYV'
    ]
    num_neurons = len(neuron_classes)
    num_body_angles = 30
    behavior_index_dict = {
        'velocity': num_neurons,
        'pumping': num_neurons + 1,
        'head_angle': num_neurons + 2,
        'body_angles': list(range(num_neurons+3, num_neurons+3+num_body_angles))
    }
    write_raw_data(ds_name, neuron_classes, behavior_index_dict)
    split_train_valid_test(ds_name, random_seed)
    write_eval_data(ds_name)


def create_ds_data0306(random_seed):

    ds_name = 'data0306'
    neuron_classes = [
        'SMDV', 'SMDD', 'SAADL', 'SAADR', 'SAAV',
        'MC', 'M3', 'M4', 'MI',
        'AVA', 'AVB', 'RIB',
        'RME', 'RMEV', 'RMED',
        'URYD', 'URYV'
    ]
    num_neurons = len(neuron_classes)
    behavior_index_dict = {
        'velocity': num_neurons,
        'pumping': num_neurons + 1,
        'head_angle': num_neurons + 2,
    }
    write_raw_data(ds_name, neuron_classes, behavior_index_dict)
    split_train_valid_test(ds_name, random_seed, split_ratio=[0.7, 0, 0.3])


def create_ds_data0310(random_seed):
    ds_name = 'data0310'
    neuron_classes = [
        'SMDV', 'SMDD', 'SAADL', 'SAADR', 'SAAV',
        'MC', 'M3', 'M4', 'MI',
        'AVA', 'AVB', 'RIB',
        'RME', 'RMEV', 'RMED',
        'URYD', 'URYV'
    ]
    num_neurons = len(neuron_classes)
    behavior_index_dict = {
        'velocity': num_neurons + 1,
        'pumping': num_neurons + 2,
        'head_angle': num_neurons + 3,
    }
    write_raw_data(ds_name, neuron_classes, behavior_index_dict)
    split_train_valid_test(ds_name, random_seed, split_ratio=[0.7, 0, 0.3])


def create_ds_data0327(random_seed):
    ds_name = 'data0327'
    neuron_classes = [
        'SMDV', 'SMDD', 'SAADL', 'SAADR', 'SAAV',
        'MC', 'M3', 'M4', 'MI',
        'AVA', 'AVB', 'RIB',
        'RME', 'RMEV', 'RMED',
        'URYD', 'URYV'
    ]
    num_neurons = len(neuron_classes)
    behavior_index_dict = {
        'velocity': num_neurons + 1,
        'pumping': num_neurons + 2,
        'head_angle': num_neurons + 3,
    }
    write_raw_data(ds_name, neuron_classes, behavior_index_dict)
    split_train_valid_test(ds_name, random_seed)


def create_ds_data0410(random_seed):
    ds_name = 'data0410'
    fig4_neuron_classes = [
        "AVB", "RIB", "RIC", "RID", "AUA", "AVJ", "AVK", "AIM", "AIY", "AIA",
        "AVA", "AVE", "AIB", "RIM", "AVL", "RIF", "RIV", "ADA", "AVD", "RMF",
        "RIA", "AVH", "RIR", "RIS", "RIH", "AIN", "RIP", "AIZ", "URB", "ALA",
        "RMG", "RMD", "RMDD", "RMDV", "RME", "RMEV", "RMED", "SAADL", "SAADR",
        "SAAV", "SMBV", "SMBD", "SMDV", "SMDD", "SIAV", "SIAD", "SIBV", "SIBD",
        "VB02", "ASJ", "IL1L", "IL1R", "IL1D", "IL1V", "URYD", "URYV", "BAG",
        "ASG", "CEPD", "CEPV", "OLL", "OLQD", "OLQV", "IL2L", "IL2R", "IL2D",
        "IL2V", "URAD", "URAV", "ADE", "FLP", "AQR", "URX", "ADL", "ASH",
        "ASEL", "ASER", "ASI", "AFD", "ASK", "AWA", "AWB", "AWC", "I1",
        "I2", "I3", "I4", "I5", "I6", "NSM", "M1", "M3", "M4", "M5", "MC", "MI"
    ]
    neuron_count = {}
    data = np.load('/store1/alicia/attention_predict/data/data0409_raw_train.npy')
    num_datasets, num_inputs, _ = data.shape
    for neuron_index in range(len(fig4_neuron_classes)):
        neuron_type = fig4_neuron_classes[neuron_index]
        neuron_count[neuron_type] = 0
        for ds_index in range(num_datasets):
            sequence = data[ds_index, neuron_index, :]
            if np.max(sequence) == np.min(sequence) == 0:
                continue
            # otherwise this neuron is recorded
            else:
                neuron_count[neuron_type] += 1
    print(neuron_count)
    # filter neuron classes based on number of datasets in which they are recorded
    neuron_classes = []
    min_count = 30
    for neuron_type, count in neuron_count.items():
        if count >= min_count:
            neuron_classes.append(neuron_type)
    print(f'neuron classes: {neuron_classes}')
    # num_neurons = len(neuron_classes)
    # behavior_index_dict = {
    #         'velocity': num_neurons + 1,
    #         'pumping': num_neurons + 2,
    #         'head_angle': num_neurons + 3
    # }
    # write_raw_data(ds_name, neuron_classes, behavior_index_dict)
    # split_train_valid_test(ds_name, random_seed)


def create_ds_sanity(random_seed):
    ds_name = 'sanity'
    neuron_classes = ['AVA']
    behavior_index_dict = {
        'velocity': 1,
        'pumping': 2,
        'head_angle': 3,
    }
    write_raw_data(ds_name, neuron_classes, behavior_index_dict)
    split_train_valid_test(
        ds_name,
        random_seed,
        split_ratio=[0.7, 0, 0.3]
    )


if __name__ == "__main__":
    # create_ds_data0108(2025)
    # create_ds_sanity(2025)
    # create_ds_data0310(1985)
    # create_ds_data0327(2001)
    # create_ds_data0409(2025)
    create_ds_data0410(2025)
