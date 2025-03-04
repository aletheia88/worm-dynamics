from flv_utils import by_class
import numpy as np


def assemble_all(neuron_classes, behavior_index_dict, max_length=1600):

    """ Create high-dimensional data arrays that contain the given neuron classes,
    standard behaviors, body curvature angles and heat-stim.
    """

    all_datasets = []
    noheatstim_datasets = []

    all_outputs = {neuron_class: {} for neuron_class in neuron_classes}
    noheatstim_outputs = {neuron_class: {} for neuron_class in neuron_classes}

    for neuron_class in neuron_classes:
        noheatstim_outputs[neuron_class] = by_class(
                neuron_class,
                behs=['velocity', 'head_angle', 'pumping', 'body_angle'],
                tag_filter=["stim"]
        ) # filter out heat-stim datasets
        all_outputs[neuron_class] = by_class(
                neuron_class,
                behs=['velocity', 'head_angle', 'pumping', 'body_angle']
        ) # contain both heat-stim and no-heat-stim datasets
        all_datasets += all_outputs[neuron_class]['datasets']
        noheatstim_datasets += noheatstim_outputs[neuron_class]['datasets']

    noheatstim_datasets = np.unique(noheatstim_datasets).tolist()
    unique_datasets = np.unique(all_datasets).tolist()
    heatstim_datasets = [ds for ds in unique_datasets if ds not in noheatstim_datasets]

    num_behaviors = 3 + len(behavior_index_dict['body_angles'])
    num_neurons = len(neuron_classes)
    num_states = 1 # heat-stim or not
    num_columns = num_neurons + num_behaviors + num_states

    velocity_index = behavior_index_dict['velocity']
    pumping_index = behavior_index_dict['pumping']
    angle_index = behavior_index_dict['head_angle']
    body_angle_indices = behavior_index_dict['body_angles']

    assembled_data = np.zeros((
        len(unique_datasets),
        num_columns,
        max_length))

    heatstim_encoding = np.ones((max_length,)) * -1
    heatstim_encoding[799:] = np.array(
            [(-1/400) * t + 3 for t in range(800, max_length+1)], dtype=np.float32)
    noheatstim_encoding = np.ones((max_length,)) * -1

    # assemble traces in the order of unique_datasets
    for i, ds in enumerate(unique_datasets):

        for j, neuron_class in enumerate(neuron_classes):

            output = all_outputs[neuron_class]
            # check if neuron_class is contained in this dataset
            if ds in output['datasets']:

                ds_index = output['datasets'].index(ds)
                assembled_data[i, j, :] = output['neuron_traces'][ds_index]

                behaviors = output['behavior_traces']
                assembled_data[i, velocity_index, :] = behaviors['velocity'][ds_index]
                assembled_data[i, pumping_index, :] = behaviors['pumping'][ds_index]
                assembled_data[i, angle_index, :] = behaviors['head_angle'][ds_index]
                # other behaviors: (num_datasets, max_length)
                # body_angles: (num_datasets, max_length, num_body_angles)
                body_angles = behaviors['body_angle'][ds_index, :, :].T
                assembled_data[i, body_angle_indices, :] = body_angles

                if ds in heatstim_datasets:
                    assembled_data[i, -1, :] = heatstim_encoding
                else:
                    assembled_data[i, -1, :] = noheatstim_encoding

    return assembled_data, unique_datasets


if __name__ == '__main__':
    neuron_classes = ['SMDD', 'SAADL', 'SAADR', 'SAAV', 'M3', 'M4', 'MI', 'AVB', 'RIB',
                      'RME', 'RMEV', 'RMED', 'URYD', 'URYV']
    # neuron_classes = ['M3']
    num_neurons = len(neuron_classes)
    num_body_angles = 30
    behavior_index_dict = {
            'velocity': num_neurons,
            'pumping': num_neurons + 1,
            'head_angle': num_neurons + 2,
            'body_angles': list(range(num_neurons+3, num_neurons+3+num_body_angles))
    }
    assembled_data, unique_datasets = assemble_all(neuron_classes, behavior_index_dict)
    print(f'assembled_data: {assembled_data.shape}')
    print(f'unique datasets: {len(unique_datasets)}')

