import copy
import numpy as np


def write_normalized_data(ds_name, num_behaviors, tfm_type='zscore'):

    tfm_params = get_zscore_tfm_params(ds_name, num_behaviors)

    new_ds_name = ds_name + '_norm'
    data_dir = '/home/alicia/store1/alicia/attention_predict/data'

    for ds_type in ['train', 'eval']:

        data = np.load(f'{data_dir}/{ds_name}_raw_{ds_type}.npy')
        # data shape: (num_datasets, num_inputs, length)
        num_inputs = data.shape[1]
        num_neurons = num_inputs - num_behaviors
        print(f'num neurons: {num_neurons}')
        datasets = np.load(f'{data_dir}/{ds_name}_raw_{ds_type}_ds.npy')

        if tfm_type == 'zscore':
            # ignore the last column that artificially expresses heat-stim
            for i in range(num_inputs - 1):

                if i > num_neurons:
                    mean = tfm_params[i]['mu']
                    stddev = tfm_params[i]['sigma']
                    data[:, i, :] = (data[:, i, :] - mean) / stddev
                else:
                    for ds_index in range(len(datasets)):
                        # set missing neuron activity to -10
                        if np.min(data[ds_index, i, :]) == np.max(data[ds_index, i, :]) == 0:
                            data[ds_index, i, :] = -10
                        else:
                            mean = tfm_params[i]['mu']
                            stddev = tfm_params[i]['sigma']
                            data[ds_index, i, :] = (data[ds_index, i, :] - mean) / stddev
        else:
            print(f'{tfm_type} normalization not implemented yet.')

        np.save(f'{data_dir}/{new_ds_name}_{ds_type}.npy', data)
        np.save(f'{data_dir}/{new_ds_name}_{ds_type}_ds.npy', datasets)

        print(f'Normalized data {new_ds_name}_{ds_type} is written!')


def get_zscore_tfm_params(ds_name, num_behaviors):

    data_dir = '/home/alicia/store1/alicia/attention_predict/data'
    raw_data = np.load(f'{data_dir}/{ds_name}_raw_train.npy')

    num_datasets, num_inputs, _ = raw_data.shape
    num_neurons = num_inputs - num_behaviors
    tfm_params = {i: {'mu': None, 'sigma': None} for i in range(num_inputs - 1)}

    for i in range(num_inputs - 1):

        # get traces of variable i
        variable_traces = raw_data[:, i, :]
        filtered_traces = []
        # variable_traces shape: (num_datasets, length)
        # if the variable is a neuron
        if i < num_neurons:
            # remove the missing neurons in each animal
            for ds_index in range(num_datasets):

                if (np.max(variable_traces[ds_index, :]) != 0 and
                    np.min(variable_traces[ds_index, :]) != 0):
                    filtered_traces.append(variable_traces[np.newaxis, ds_index, :])

            filtered_traces = np.concatenate(filtered_traces)
        else:
            filtered_traces = variable_traces

        print(f'filtered_traces {i}: {filtered_traces.shape}')
        # compute parameters after removing missing neurons
        tfm_params[i]['mu'] = np.mean(filtered_traces)
        tfm_params[i]['sigma'] = np.std(filtered_traces)

    return tfm_params


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
        tfm_fn[i]['b'] = (lower_bound + upper_bound) / (-2)
        tfm_fn[i]['m'] = (1 - tfm_params[i]['b']) / upper_bound

    return tfm_fn


if __name__ == '__main__':

    ds_name = 'data0108'
    num_behaviors = 34 # 34 = 3 (cepnem beh) + 30 (body angles) + 1 (heat-stim)
    tfm_params = get_zscore_tfm_params(ds_name, num_behaviors)
    print(tfm_params)
    write_normalized_data(ds_name, num_behaviors)
