from load import (
        assemble, load_behaviors_from_all_neuropal, load_single_neuron_class,
        assemble_data_with_missing_neurons)
from copy import deepcopy
import numpy as np


def split_train_valid_test(
        ds_name,
        split_ratio=[0.7, 0.2, 0.1],
        random_seed=42,
        save=True
):
    data = np.load(f'../data/{ds_name}.npy')
    datasets = np.load(f'../data/{ds_name}_datasets.npy')
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
        np.save(f'../data/{ds_name}_train.npy', train_set)
        np.save(f'../data/{ds_name}_valid.npy', valid_set)
        np.save(f'../data/{ds_name}_test.npy', test_set)
        # Also save the corresponding dataset names
        np.save(f'../data/{ds_name}_train_ds.npy', train_datasets)
        np.save(f'../data/{ds_name}_valid_ds.npy', valid_datasets)
        np.save(f'../data/{ds_name}_test_ds.npy', test_datasets)
        print(f'Train/Valid/Test splits for {ds_name} saved!')
    else:
        return train_set, valid_set, test_set


def write_all_behaviors(max_len):

    std_behaviors, _ = load_behaviors_from_all_neuropal(max_len)
    for ds in std_behaviors.keys():
        # normalize velocity
        std_behaviors[ds][:, 0] = 10 * std_behaviors[ds][:, 0]
        # normalize pumping
        std_behaviors[ds][:, 2] = std_behaviors[ds][:, 2] / 2 - 1

    assembled_behaviors = np.stack(list(std_behaviors.values()), axis=0)
    np.save(f'../data/all_behaviors.npy', assembled_behaviors)
    print('All behaviors saved!')


def write_data_with_missing_neurons(neuron_classes, max_len, file_name):

    # GCaMP traces shape: (n, 1600, m)
    # behavior traces shape: (n, 1600, k)
    # Note: traces are already normalized during assembling
    assembled_gcamp_traces, assembled_behaviors, assembled_datasets = \
            assemble_data_with_missing_neurons(neuron_classes, max_len)

    assembled_traces = np.concatenate(
            (assembled_gcamp_traces, assembled_behaviors),
            axis=2).transpose(0, 2, 1)

    np.save(f'../data/{file_name}.npy', assembled_traces)
    np.save(f'../data/{file_name}_datasets.npy', np.array(assembled_datasets))
    print(f'{file_name} written under data folder.')


def write_specific_pairings(neuron_class, max_len, file_name):

    # Let n be the number of worms and k the number of behaviors
    # GCaMP traces shape: (n, 1600)
    # behavior traces shape: (n, 1600, k)
    gcamp_traces, std_behaviors, _ = load_single_neuron_class(
            neuron_class,
            max_len=max_len)
    # normalize traces
    # in this case, k = 1 since we only include the feeding behavior, i.e., pumping
    std_behaviors[:, :, 0] = std_behaviors[:, :, 0] / 2 - 1
    f_mean = np.mean(gcamp_traces)
    gcamp_traces = gcamp_traces / (2 * f_mean) - 1

    assembled_traces = np.concatenate(
            (gcamp_traces[:, np.newaxis, :],
            std_behaviors.transpose(0, 2, 1)),
            axis=1
    )
    np.save(f'../data/{file_name}.npy', assembled_traces)
    print(f'{file_name} written under data folder!')


def write_to_npy(neuron_classes, max_len, max_animals):

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
    np.save(f'../data/{file_name}', assembled_traces)

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
    # write_all_behaviors(max_len, save=True)

    ### Split into training, validation, and testing datasets
    ds_name = 'AVA_MC_all'
    random_seed = 1912 # Alan Turing's random seed
    split_train_valid_test(ds_name, random_seed=random_seed)

    ### Assemble neural and behavioral trace of 1 neuron class and write to npy
    # neuron_class = 'MC'
    # gcamp_traces, std_behaviors, reversals = load_single_neuron_class(neuron_class, verbose=True)

    ### Assemble specific neural and behavioral pairings
    # neuron_class = 'MC'
    # max_len = 1600
    # file_name = 'MC_pumping'
    # write_specific_pairings(neuron_class, max_len, file_name)

    ### Assemble data with missing neurons
    # neuron_classes = ['AVA', 'MC']
    # max_len = 1600
    # file_name = 'AVA_MC_all'
    # write_data_with_missing_neurons(neuron_classes, max_len, file_name)
