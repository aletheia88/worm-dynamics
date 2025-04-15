import numpy as np


# How to determine the sampling weight/probability for each sample
# 1. Count the number of times each neuron appears in all training datasets
# 2. For each dataset A, find the neuron that appears the least number of times
# according to the counts above—suppose this number is K.
# 3. Assign the dataset this number K
# 4. Compute the sampling weight:
# all the 1200 samples generated from dataset A will have the sampling weight:
# 1-K/N, where N is the total number of datasets

def compute_sampling_weights(ds_name):

    """ According to the formula: 1 - K/N. """

    # aggregate the counts for each neuron type across all datasets
    # sort in the descending order

    neuron_classes = [
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

    ds_path = '/store1/alicia/attention_predict/data'
    data = np.load(f'{ds_path}/{ds_name}.npy')
    num_datasets = data.shape[0]
    sampling_weights_map = {}

    aggregated_neuron_counts = count_neurons(
            data, neuron_classes)

    for ds_index in range(num_datasets):

        sequences = data[ds_index, :, :]
        least_count = get_least_common(
                sequences,
                neuron_classes,
                aggregated_neuron_counts
        )
        sampling_weight = 1 - least_count/num_datasets
        sampling_weights_map[ds_index] = sampling_weight

    return sampling_weights_map


def count_neurons(data, neuron_classes):

    """ Aggregate the number of datasets each neuron is recorded in. """

    # count the number of datasets each neuron type is recorded
    aggregated_neuron_counts = {}
    num_neurons = len(neuron_classes)
    num_datasets, num_variables, _ = data.shape
    num_neurons = num_variables - 4

    for neuron_index in range(num_neurons):

        neuron_type = neuron_classes[neuron_index]
        aggregated_neuron_counts[neuron_type] = 0

        for ds_index in range(num_datasets):

            sequence = data[ds_index, neuron_index, :]
            # assume the data is not normalized
            if np.max(sequence) == np.min(sequence) == 0:
                continue
            # this neuron is recorded one (more) time
            else:
                aggregated_neuron_counts[neuron_type] += 1

    return aggregated_neuron_counts


def get_least_common(
        sequences,
        neuron_classes,
        aggregated_neuron_counts
):

    """ For each dataset, find the least common neuron recorded
    according to the aggregated neuron counts and its frequency. """

    # find the recorded neurons
    num_variables = sequences.shape[0]
    recorded_neurons = []

    for neuron_index in range(num_variables - 4):

        trace = sequences[neuron_index, :]

        if np.max(trace) == np.min(trace) == 0:
            continue
        else:
            recorded_neurons.append(neuron_classes[neuron_index])

    # assign each neuron its count in the aggregated data
    ds_neuron_count = {}
    for neuron in recorded_neurons:
        ds_neuron_count[neuron] = aggregated_neuron_counts[neuron]

    # sort recorded neurons according to the aggregated neuron counts
    sorted_neuron_types, sorted_neuron_counts = sort_neuron_count(
            ds_neuron_count)
    # print(f'sorted neuron types:\n{sorted_neuron_types}')
    # print(f'sorted neuron counts:\n{sorted_neuron_counts}')

    return sorted_neuron_counts[-1]


def sort_neuron_count(neuron_counts):

    """ Sort the neurons and their counts in the descending order. """

    sorted_items = sorted(
            neuron_counts.items(),
            key=lambda item: item[1], reverse=True)
    sorted_neuron_types = [item[0] for item in sorted_items]
    sorted_counts = [item[1] for item in sorted_items]

    return sorted_neuron_types, sorted_counts


def weigh_all_samples(ds_name, num_datasets):

    sampling_weights_map = compute_sampling_weights(ds_name)
    sample_weights = []
    for ds_index in range(num_datasets):
        sample_weights += [sampling_weights_map[ds_index]] * 1200

    return sample_weights


if __name__ == "__main__":
    ds_name = 'data0410_raw_train'
    # sampling_weights_map = compute_sampling_weights(ds_name)
    # print(sampling_weights_map)
    sample_weights = weigh_all_samples(ds_name, 15)
    print(f'sample weights:\n{sample_weights}')
