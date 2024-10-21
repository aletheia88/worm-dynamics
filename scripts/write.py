from load import (
        assemble, load_behaviors, load_single_neuron_class,
        assemble_data_with_missing_neurons, shuffle)
from copy import deepcopy
import numpy as np


def split_train_valid_test(
        ds_name,
        split_ratio=[0.8, 0.1, 0.1],
        random_seed=42,
        save=True
):
    data_dir = '/home/alicia/store1/alicia/attention_predict/data'

    data = np.load(f'{data_dir}/{ds_name}.npy')
    datasets = np.load(f'{data_dir}/{ds_name}_datasets.npy')
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
        np.save(f'{data_dir}/{ds_name}_train.npy', train_set)
        np.save(f'{data_dir}/{ds_name}_valid.npy', valid_set)
        np.save(f'{data_dir}/{ds_name}_test.npy', test_set)
        # Also save the corresponding dataset names
        np.save(f'{data_dir}/{ds_name}_train_ds.npy', train_datasets)
        np.save(f'{data_dir}/{ds_name}_valid_ds.npy', valid_datasets)
        np.save(f'{data_dir}/{ds_name}_test_ds.npy', test_datasets)
        print(f'Train/Valid/Test splits for {ds_name} saved!')
    else:
        return train_set, valid_set, test_set


def write_eval_data(ds_name):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    # find unique datasets from validation and testing that not in training
    # get for their indices in `valid_ds` and `test_ds`
    # use these indices to get traces for each datasets
    # lastly assemble all the datasets
    train_ds = np.load(f'{data_dir}/{ds_name}_train_ds.npy')
    valid_ds = np.load(f'{data_dir}/{ds_name}_valid_ds.npy')
    test_ds = np.load(f'{data_dir}/{ds_name}_test_ds.npy')

    valid_data = np.load(f'{data_dir}/{ds_name}_valid.npy')
    test_data = np.load(f'{data_dir}/{ds_name}_test.npy')

    unique_valid = list(set(valid_ds).difference(train_ds))
    unique_test = list(set(test_ds).difference(train_ds))

    eval_datasets = np.unique(unique_valid + unique_valid).tolist()
    np.save(f'{data_dir}/{ds_name}_eval_ds.npy', eval_datasets)

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

    np.save(f'{data_dir}/{ds_name}_eval.npy', eval_data)
    print(f'Eval data/datasets for {ds_name} are saved!')


def write_shuffled_data(ds_name, num_neurons, random_seed, shuffle_type):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    shuffled_data, shuffled_datasets = shuffle(ds_name, num_neurons, random_seed,
                                               shuffle_type=shuffle_type)
    np.save(f'{data_dir}/{ds_name}_shuffle.npy', shuffled_data)
    np.save(f'{data_dir}/{ds_name}_ds_shuffle.npy', shuffled_datasets)
    print(f'Shuffled {ds_name} saved!')


def write_behaviors(max_len, ds_name):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    std_behaviors, _, datasets = load_behaviors(max_len)
    # Normalize velocity
    std_behaviors[:, :, 0] = 10 * std_behaviors[:, :, 0]
    # Normalize head angle
    head_angle = std_behaviors[:, :, 1]
    min_val = np.min(head_angle)
    max_val = np.max(head_angle)
    std_behaviors[:, :, 1] = 2 * ((head_angle - min_val) / (max_val - min_val)) - 1

    # Normalize pumping
    # std_behaviors[:, :, 2] = std_behaviors[ds][:, 2] / 2 - 1
    np.save(f'{data_dir}/{ds_name}.npy', std_behaviors.transpose(0, 2, 1))
    np.save(f'{data_dir}/{ds_name}_datasets.npy', datasets)
    print(f'{ds_name} saved!')


def write_data_with_missing_neurons(neuron_classes, max_len, file_name):

    # GCaMP traces shape: (n, 1600, m)
    # behavior traces shape: (n, 1600, k)
    # Note: traces are already normalized during assembling
    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    outputs = assemble_data_with_missing_neurons(neuron_classes, max_len)
    # ouputs is a tuple of three arrays:
    # 0: assembled_gcamp_traces
    # 1: assembled_behaviors
    # 2: assembled_datasets
    # 3: dataset_neuron_info
    assembled_gcamp_traces = outputs[0]
    assembled_behaviors = outputs[1]
    assembled_datasets = outputs[2]

    assembled_traces = np.concatenate(
            (assembled_gcamp_traces, assembled_behaviors), axis=2).transpose(0, 2, 1)

    np.save(f'{data_dir}/{file_name}.npy', assembled_traces)
    np.save(f'{data_dir}/{file_name}_datasets.npy', np.array(assembled_datasets))
    print(f'{file_name} written under data folder.')


def write_specific_pairings(neuron_class, max_len, file_name):

    # Let n be the number of worms and k the number of behaviors
    # GCaMP traces shape: (n, 1600)
    # behavior traces shape: (n, 1600, k)
    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    gcamp_traces, std_behaviors, _, datasets = load_single_neuron_class(
            neuron_class,
            max_len=max_len)
    # Normalize GCaMP
    f_mean = np.mean(gcamp_traces)
    gcamp_traces = gcamp_traces / (2 * f_mean) - 1

    # Assuming velocity corresponds to column 0
    std_behaviors[:, :, 0] = 10 * std_behaviors[:, :, 0]
    # Assuming pumping corresponds to column 1
    std_behaviors[:, :, 1] = std_behaviors[:, :, 1] / 2 - 1
    # Assuming head curvature corresponds to column 2
    head_angle = std_behaviors[:, :, 2]
    min_val = np.min(head_angle)
    max_val = np.max(head_angle)
    std_behaviors[:, :, 2] = 2 * ((head_angle - min_val) / (max_val - min_val)) - 1

    assembled_traces = np.concatenate(
            (gcamp_traces[:, np.newaxis, :],
            std_behaviors.transpose(0, 2, 1)),
            axis=1)
    np.save(f'{data_dir}/{file_name}.npy', assembled_traces)
    np.save(f'{data_dir}/{file_name}_datasets.npy', datasets)
    print(f'{file_name} written under data folder!')


def write_to_npy(neuron_classes, max_len, max_animals):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    raw_neuron_traces, raw_behavior_traces = select_traces(
            neuron_classes,
            max_len,
            max_animals)
    normalized_neuron_traces, normalized_behavior_traces = normalize_traces(
            raw_neuron_traces,
            raw_behavior_traces)

    stacked_neuron_traces = []

    for neuron_traces in normalized_neuron_traces.values():
        # append trace by neuron class
        # each neuron trace has shape (num_worms, max_len)
        stacked_neuron_traces.append(neuron_traces)

    stacked_neuron_traces = np.array(stacked_neuron_traces).transpose(1, 0, 2)

    # bahavior trace of each dataset has shape (max_len, 3) 
    stacked_behavior_traces = np.stack(
            list(normalized_behavior_traces.values()), axis=0).transpose(0, 2, 1)

    assembled_traces = np.concatenate((
        stacked_neuron_traces.astype(np.float32),
        stacked_behavior_traces.astype(np.float32)),
        axis=1)

    # write under '../data/'
    file_name = '_'.join(neuron_classes)
    np.save(f'{data_dir}/{file_name}', assembled_traces)

    print(f'{file_name} written under data folder!')


def select_traces(neuron_classes, max_len, max_animals):

    # assemble neural and behavioral traces
    gcamp_traces, std_behaviors, _ = assemble(neuron_classes, max_len, max_animals)

    # select which datasets to include in the final .npy file
    datasets = [list(gcamp_traces[neuron].keys())
                for neuron in gcamp_traces.keys()
                if len(gcamp_traces[neuron].keys()) > 0]

    datasets.append(list(std_behaviors.keys()))
    unique_datasets = list(set().union(*datasets))
    #print(f'number of unique datasets: {len(unique_datasets)}')
    raw_neuron_traces = {neuron_class: [] for neuron_class in neuron_classes}
    raw_behavior_traces = {}

    for unique_ds in unique_datasets:

        remaining_neuron_classes = deepcopy(neuron_classes)
        # keeps track of traces from which neuron classes that have just been appended
        just_added_classes = []

        for neuron, dataset_trace in gcamp_traces.items():

            if unique_ds in dataset_trace.keys():

                neuron_class = get_neuron_class(neuron_classes, neuron)

                if neuron_class in remaining_neuron_classes:
                    #print(f'remain classes: {remaining_neuron_classes}')
                    #print(f'{neuron_class} adds {unique_ds}')
                    raw_neuron_traces[neuron_class].append(dataset_trace[unique_ds])
                    just_added_classes.append(neuron_class)
                    remaining_neuron_classes.remove(neuron_class)

                    if unique_ds not in raw_behavior_traces.keys():
                        #print(f'behavior adds {unique_ds}')
                        raw_behavior_traces[unique_ds] = std_behaviors[unique_ds]

        # the remaining neuron classes should be empty, otherwise we discard the dataset
        #print(f'remain classes after appending: {len(remaining_neuron_classes)}\n')
        if len(remaining_neuron_classes) != 0:
            #print(f'just updated classes: {just_added_classes}\n')
            raw_behavior_traces.popitem()
            #print(f'behavior pops {behavior_traces.popitem()[0]}\n')
            for updated_class in just_added_classes:
                raw_neuron_traces[updated_class].pop()
                #print(f'neurons pops {updated_class}\n')

    for neuron_class in neuron_classes:
        raw_neuron_traces[neuron_class] = np.array(raw_neuron_traces[neuron_class])
        #print(f'{neuron_class} shape: {raw_neuron_traces[neuron_class].shape}')

    #print(f'behaviors: {len(raw_behavior_traces.keys())}, {raw_behavior_traces.keys()}')
    return raw_neuron_traces, raw_behavior_traces


def normalize_traces(raw_neuron_traces, behavior_traces):

    normalized_neuron_traces = {}
    normalized_behavior_traces = deepcopy(behavior_traces)

    for neuron_class, traces in raw_neuron_traces.items():

        f_mean = np.mean(traces)
        # TODO: check if trace is zero-centered
        # otherwise, use (1/2) * (traces / f_mean - 1)
        #normalized_neuron_traces[neuron_class] = (1/2) * (traces / f_mean - 1)
        normalized_neuron_traces[neuron_class] = traces / (2 * f_mean) - 1

    for ds, traces in behavior_traces.items():

        # normalize velocity
        normalized_behavior_traces[ds][:, 0] = 10 * traces[:, 0]
        # normalize pumping
        normalized_behavior_traces[ds][:, 2] = traces[:, 2] / 2 - 1
        # normalize head angle/curvature - pending

    return normalized_neuron_traces, normalized_behavior_traces


def get_neuron_class(neuron_classes, neuron):

    for neuron_class in neuron_classes:

        if neuron.startswith(neuron_class):
            return neuron_class

    return None


if __name__ == "__main__":

    ### Assemble neural and behavioral traces of >1 neuron classes and write data to npy
    # neuron_classes = ['M3']
    # max_len = 1600
    # max_animals = 100
    # write_to_npy(neuron_classes, max_len, max_animals)

    ### Write all behavioral data to npy file
    # max_len = 1600
    # ds_name = 'velocity_headangle'
    # write_behaviors(max_len, ds_name)

    ### Split into training, validation, and testing datasets
    # ds_name = 'RID_AVE_RIV_AVD_AIN'
    # random_seed = 1912
    # split_train_valid_test(ds_name, random_seed=random_seed)

    ### Assemble neural and behavioral trace of 1 neuron class and write to npy
    # neuron_class = 'MC'
    # gcamp_traces, std_behaviors, reversals = load_single_neuron_class(neuron_class, verbose=True)

    ### Assemble specific neural and behavioral pairings
    # neuron_class = 'MC'
    # max_len = 1600
    # file_name = 'MC_pumping'
    # write_specific_pairings(neuron_class, max_len, file_name)

    ### Assemble data with missing neurons
    # neuron_classes = ['RID', 'AVE', 'RIV', 'AVD', 'AIN']
    # max_len = 1600
    # file_name = 'RID_AVE_RIV_AVD_AIN'
    # write_data_with_missing_neurons(neuron_classes, max_len, file_name)

    ### Create and write shuffled data
    # ds_name = 'RID_AVE_RIV_AVD_AIN_train'
    # random_seed = 1912
    # num_neurons = 3
    # shuffle_type = 'all'
    # write_shuffled_data(ds_name, num_neurons, random_seed, shuffle_type)

    ### Create and write 4-column data: interneuron + std behaviors
    # interneurons = ['RID', 'AUA', 'AVJ', 'AVE', 'AIB', 'RIV', 'AVD', 'RIA', 'AIN',
    #                 'AIZ', 'URB']
    # max_len = 1600
    # for neuron_class in interneurons:
    #     file_name = f'{neuron_class}_stdbeh'
    #     write_specific_pairings(neuron_class, max_len, file_name)
    # for neuron_class in interneurons:
    #     file_name = f'{neuron_class}_stdbeh'
    #     split_train_valid_test(file_name, random_seed=1912)

    ### Write evaluation datasets: union of validation and testing datasets
    ds_name = 'RID_AVE_RIV_AVD_AIN'
    write_eval_data(ds_name)

