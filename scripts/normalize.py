import copy
import numpy as np


def write_normalized_data(ds_name, num_behaviors, tfm_type='zscore'):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    new_ds_name = ds_name + '_norm'
    tfm_params, _ = get_zscore_tfm_params(ds_name, num_behaviors)

    for ds_type in ['train', 'valid']:

        data = np.load(f'{data_dir}/{ds_name}_raw_{ds_type}.npy')
        num_datasets, num_inputs, length = data.shape
        print(f'raw data: {data.shape}')
        normalized_data = np.zeros((num_datasets, num_inputs, length))
        # data shape: (num_datasets, num_inputs, length)
        num_neurons = num_inputs - num_behaviors
        # count "heat-stim" as one of the neurons
        print(f'num neurons: {num_neurons}')
        datasets = np.load(f'{data_dir}/{ds_name}_raw_{ds_type}_ds.npy')

        if tfm_type == 'zscore':
            for i in range(num_inputs):

                # TODO: to dealt with heat-stim, change back to i > num_neurons - 1
                if i >= num_neurons:
                    mean = tfm_params[i]['mu']
                    stddev = tfm_params[i]['sigma']

                    normalized_data[:, i, :] = (data[:, i, :] - mean) / stddev

                # ignore the heat-stim column indexed at 'num_neurons - 1'
                # TODO: to deal with heat-stim, change back to num_neurons - 1
                elif i < num_neurons:

                    for ds_index in enumerate(range(len(datasets))):
                        # set missing neuron activity to -10
                        if np.min(data[ds_index, i, :]) == \
                                np.max(data[ds_index, i, :]) == 0:
                            normalized_data[ds_index, i, :] = -10
                        else:
                            mean = tfm_params[i]['mu']
                            stddev = tfm_params[i]['sigma']
                            if mean is not None and stddev is not None:
                                normalized_data[ds_index, i, :] = (
                                        data[ds_index, i, :] - mean) / stddev
        else:
            print(f'{tfm_type} normalization not implemented yet.')

        np.save(f'{data_dir}/{new_ds_name}_{ds_type}.npy', normalized_data)
        np.save(f'{data_dir}/{new_ds_name}_{ds_type}_ds.npy', datasets)
        print(f'Normalized data {new_ds_name}_{ds_type} is written!')


def get_zscore_tfm_params(ds_name, num_behaviors):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    raw_data = np.load(f'{data_dir}/{ds_name}_raw_train.npy')

    num_datasets, num_inputs, _ = raw_data.shape
    num_neurons = num_inputs - num_behaviors
    tfm_params = {i: {'mu': None, 'sigma': None} for i in range(num_inputs)}

    void_neuron_indices = []

    for i in range(num_inputs):
        # get traces of variable i
        variable_traces = raw_data[:, i, :]
        filtered_traces = []
        # variable_traces shape: (num_datasets, length)
        # if the variable is a neuron
        # TODO: change back to num_neurons - 1
        if i == num_neurons:
            # TODO: uncomment continue to escape heat-stim
            # continue
            filtered_traces = variable_traces
        # minus 1 to exclude the heat-stim column
        # TODO: change back to num_neurons - 1
        elif i < num_neurons:
            # remove the missing neurons in each animal
            for ds_index in range(num_datasets):
                if (
                    np.max(variable_traces[ds_index, :]) != 0 and
                    np.min(variable_traces[ds_index, :]) != 0
                   ):
                    filtered_traces.append(
                        variable_traces[np.newaxis, ds_index, :]
                    )
            # remove the neuron that is never recorded
            if len(filtered_traces) == 0:
                void_neuron_indices.append(i)
            else:
                filtered_traces = np.concatenate(filtered_traces)
        else:
            filtered_traces = variable_traces

        if len(filtered_traces) != 0:
            print(f'filtered_traces {i}: {filtered_traces.shape}')
            # compute parameters after removing missing neurons
            tfm_params[i]['mu'] = np.mean(filtered_traces)
            tfm_params[i]['sigma'] = np.std(filtered_traces)

    return tfm_params, void_neuron_indices


def get_zscore_params_for_behavior(ds_name, num_behaviors):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    raw_data = np.load(f'{data_dir}/{ds_name}_raw_train.npy')

    num_datasets, num_inputs, _ = raw_data.shape
    num_neurons = num_inputs - num_behaviors
    zscore_params = {}

    for var_index in range(num_neurons + 1, num_inputs):
        zscore_params[var_index] = {}
        zscore_params[var_index]['mu'] = np.mean(raw_data[:, var_index, :])
        zscore_params[var_index]['sigma'] = np.std(raw_data[:, var_index, :])

    return zscore_params


def get_f90_params_for_neurons(ds_name, num_behaviors):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    normalized_data = np.load(f'{data_dir}/{ds_name}-0_train.npy')

    num_datasets, num_inputs, _ = normalized_data.shape
    num_neurons = num_inputs - num_behaviors
    f90_params = {}

    # compute the population 90th percentile
    for var_index in range(num_neurons):

        activity = normalized_data[:, var_index, :]
        filtered_traces = []

        for ds_index in range(num_datasets):
            if (
                np.max(activity[ds_index, :]) != 0 and
                np.min(activity[ds_index, :]) != 0
               ):
                filtered_traces.append(
                    activity[np.newaxis, ds_index, :]
                )
        # excluding the missing neurons
        filtered_traces = np.concatenate(filtered_traces)
        print(f'filtered {var_index}: {filtered_traces.shape}')
        f90_params[var_index] = np.percentile(filtered_traces, 90)

    return f90_params


def get_linear_tfm_params(ds_name, upper_percentile=100, lower_percentile=0):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    data = np.load(f'{data_dir}/{ds_name}.npy')

    num_inputs = data.shape[1]
    # for each variable we need to find 'm' and 'b' of the linear transform
    tfm_params = {i: {'m': None, 'b': None} for i in range(num_inputs)}

    # TODO: remove the missing neurons
    for i in range(num_inputs):

        traces = data[:, i, :]
        upper_bound, lower_bound = np.percentile(
                traces,
                [upper_percentile, lower_percentile]
        )
        tfm_params[i]['b'] = (lower_bound + upper_bound) / (-2)
        tfm_params[i]['m'] = (1 - tfm_params[i]['b']) / upper_bound

    return tfm_params


def normalize0505(ds_name, num_behaviors):

    """ New normalization scheme for raw data0410 and its new splits.

    ** Normalizing Neurons **

    1. For each cell's original flurorescence vector (F_i),
    calculate the 10th percentile (P_10i) and divide all
    values in F_i by P_10i.

    2. Find the 90th percentile of the pooled list of F/F10 values
    across animals.

    3. Liear-transform the activity according to the following
        X_normalized = a * X_orginal + b
        where a = (2 / (f90 - 1)) and b = - (f90 + 1)/(f90 - 1)

    ** Normalizing Behaviors **

    1. For each behavior, compute mean and stddev across animals
    from training data only

    2. Z-score the activity by
        X_normalized = (X_original - mean_population) / stddev_population
    """

    new_ds_name = ds_name + '_norm0505'
    data_dir = '/home/alicia/store1/alicia/attention_predict/data'

    for ds_type in ['train', 'valid', 'test']:

        data = np.load(f'{data_dir}/{ds_name}_raw_{ds_type}.npy')
        datasets = np.load(f'{data_dir}/{ds_name}_raw_{ds_type}_ds.npy')

        num_datasets, num_inputs, length = data.shape
        normalized_data = copy.deepcopy(data)
        print(f'raw data: {data.shape}')

        # note: behavior includes heat-stim
        num_neurons = num_inputs - num_behaviors
        print(f'num neurons: {num_neurons}')

        ### normalizing neurons ###

        # first, baseline-scaling by individual
        for ds_index in range(num_datasets):

            for var_index in range(num_neurons):

                activity = data[ds_index, var_index, :]
                if (
                    np.max(activity) != 0 and
                    np.min(activity) != 0
                ):
                    f10 = np.percentile(activity, 10)
                    normalized_data[ds_index, var_index, :] = activity/f10

        # save the individually normalized training animals
        # for computing the population 90th percentile
        if ds_type == 'train':
            np.save(f'{data_dir}/{new_ds_name}-0_train.npy', normalized_data)
            print(f'individually normalized {new_ds_name}-0_train saved!')

        # second, compute the population 90th percentile from training data
        f90_params = get_f90_params_for_neurons(new_ds_name, num_behaviors)
        print(f'f90 params: {f90_params}')

        # third, do linear-scaling that sets individual f10 to -1
        # and population f90 to 1, while setting missing neuron activity to -10
        for var_index in range(num_neurons):
            f90 = f90_params[var_index]

            for ds_index in range(num_datasets):

                activity = normalized_data[ds_index, var_index, :]

                if (
                    np.max(activity) == 0 and
                    np.min(activity) == 0
                ):
                    normalized_data[ds_index, var_index, :] = -10
                else:
                    normalized_data[ds_index, var_index, :] = \
                        (2 / (f90 - 1)) * activity - (f90 + 1)/(f90 - 1)

        # compute mean and stddev from training data
        zscore_params = get_zscore_params_for_behavior(
                ds_name,
                num_behaviors)
        print(f'zscore params: {zscore_params}')

        # normalize behavior by z-scoring
        # note: we ignore the heat-stim column
        for var_index in range(num_neurons + 1, num_inputs):

            mean = zscore_params[var_index]['mu']
            std = zscore_params[var_index]['sigma']

            for ds_index in range(num_datasets):

                activity = normalized_data[ds_index, var_index, :]
                normalized_data[ds_index, var_index, :] = (activity - mean) / std

        # write normalized data to npy file
        np.save(f'{data_dir}/{new_ds_name}_{ds_type}.npy', normalized_data)
        np.save(f'{data_dir}/{new_ds_name}_{ds_type}_ds.npy', datasets)
        print(f'Normalized data {new_ds_name}_{ds_type} is written!')


if __name__ == '__main__':

    # ds_name = 'data0108'
    # num_behaviors = 34 # 34 = 3 (cepnem beh) + 30 (body angles) + 1 (heat-stim)
    # ds_name = 'data0327'
    # num_behaviors = 3
    # write_normalized_data(ds_name, num_behaviors)
    ds_name = 'data0715'
    num_behaviors = 9
    # tfm_params, void_neuron_indices = get_zscore_tfm_params(
    #         ds_name,
    #         num_behaviors)
    # print(tfm_params)
    # print(f'indices of neurons never recorded: {void_neuron_indices}')
    # write_normalized_data(ds_name, num_behaviors)
    normalize0505(ds_name, num_behaviors)
