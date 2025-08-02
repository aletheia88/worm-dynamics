from copy import deepcopy
from flv_utils import by_class
import numpy as np



def assemble_all(neuron_classes, behavior_index_dict, max_length=1600):

    """ Create high-dimensional data arrays that contain the given neuron
    classes, standard behaviors, body curvature angles and heat-stim.
    """

    all_datasets = []
    noheatstim_datasets = []

    all_outputs = {neuron_class: {} for neuron_class in neuron_classes}
    noheatstim_outputs = {neuron_class: {} for neuron_class in neuron_classes}

    for neuron_class in neuron_classes:
        noheatstim_outputs[neuron_class] = by_class(
                neuron_class,
                behs=['velocity', 'head_angle', 'pumping', 'body_angle',
                      'angular_velocity', 'worm_angle'],
                exclude=["stim"],
        )  # filter out heat-stim datasets
        all_outputs[neuron_class] = by_class(
                neuron_class,
                behs=['velocity', 'head_angle', 'pumping', 'body_angle',
                      'angular_velocity', 'worm_angle']
        )  # contain both heat-stim and no-heat-stim datasets
        all_datasets += all_outputs[neuron_class]['datasets']
        noheatstim_datasets += noheatstim_outputs[neuron_class]['datasets']

    noheatstim_datasets = np.unique(noheatstim_datasets).tolist()
    unique_datasets = np.unique(all_datasets).tolist()
    heatstim_datasets = [
            ds for ds in unique_datasets
            if ds not in noheatstim_datasets
    ]

    num_behaviors = 9
    num_neurons = len(neuron_classes)
    num_columns = num_neurons + num_behaviors

    velocity_index = behavior_index_dict['velocity']
    pumping_index = behavior_index_dict['pumping']
    angle_index = behavior_index_dict['head_angle']
    # body_angle_indices = behavior_index_dict['body_angles']
    body_index1 = behavior_index_dict['body_angle1']
    body_index2 = behavior_index_dict['body_angle2']
    body_index3 = behavior_index_dict['body_angle3']
    turning_index = behavior_index_dict['turning_rate']
    heading_index = behavior_index_dict['worm_angle']

    assembled_data = np.zeros((
        len(unique_datasets),
        num_columns,
        max_length))

    heatstim_encoding = np.ones((max_length,)) * -1
    heatstim_encoding[799:] = np.array(
            [(-1/400) * t + 3 for t in range(800, max_length+1)],
            dtype=np.float32)
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
                assembled_data[i, velocity_index, :] = \
                    behaviors['velocity'][ds_index]
                assembled_data[i, pumping_index, :] = \
                    behaviors['pumping'][ds_index]
                assembled_data[i, angle_index, :] = \
                    behaviors['head_angle'][ds_index]
                # other behaviors: (num_datasets, max_length)

                # body_angles: (num_body_angles, max_length)
                body_angles = behaviors['body_angle'][ds_index, :, :].T

                # TODO: average body angles across segment
                # assembled_data[i, body_angle_indices, :] = body_angles
                assembled_data[i, body_index1, :] = np.mean(
                        body_angles[:10, :], axis=0)

                assembled_data[i, body_index2, :] = np.mean(
                        body_angles[10:20, :], axis=0)

                assembled_data[i, body_index3, :] = np.mean(
                        body_angles[20:30, :], axis=0)

                assembled_data[i, turning_index, :] = \
                    behaviors['angular_velocity'][ds_index]

                assembled_data[i, heading_index, :] = \
                    behaviors['worm_angle'][ds_index]

                if ds in heatstim_datasets:
                    assembled_data[i, num_neurons, :] = heatstim_encoding
                else:
                    assembled_data[i, num_neurons, :] = noheatstim_encoding

    return assembled_data, unique_datasets


def assemble_std_beh(neuron_classes, behavior_index_dict, max_length=1600):

    """ Create high-dimensional data arrays that contain
    the given neuron classes, standard behaviors and heat-stim
    (No body angles). """

    all_datasets = []
    noheatstim_datasets = []

    all_outputs = {neuron_class: {} for neuron_class in neuron_classes}
    noheatstim_outputs = {neuron_class: {} for neuron_class in neuron_classes}

    for neuron_class in neuron_classes:
        noheatstim_outputs[neuron_class] = by_class(
                neuron_class,
                behs=['velocity', 'head_angle', 'pumping'],
                exclude=['stim'],
                LR_ops='random'
        ) # filter out heat-stim datasets
        all_outputs[neuron_class] = by_class(
                neuron_class,
                behs=['velocity', 'head_angle', 'pumping'],
                LR_ops='random'
        ) # contain both heat-stim and no-heat-stim datasets
        all_datasets += all_outputs[neuron_class]['datasets']
        noheatstim_datasets += noheatstim_outputs[neuron_class]['datasets']

    noheatstim_datasets = np.unique(noheatstim_datasets).tolist()
    unique_datasets = np.unique(all_datasets).tolist()
    print(f'number of all_datasets: {len(all_datasets)}')
    print(f'number of unique datasets: {len(unique_datasets)}')
    heatstim_datasets = [ds for ds in unique_datasets if ds not in noheatstim_datasets]
    print(f'number of heat-stim datasets: {len(heatstim_datasets)}')

    num_behaviors = 3
    num_neurons = len(neuron_classes)
    num_states = 1 # heat-stim or not
    num_inputs = num_neurons + num_behaviors + num_states

    velocity_index = behavior_index_dict['velocity']
    pumping_index = behavior_index_dict['pumping']
    head_index = behavior_index_dict['head_angle']

    assembled_data = np.zeros((
        len(unique_datasets),
        num_inputs,
        max_length))

    heatstim_encoding = np.ones((max_length,)) * -1
    heatstim_encoding[799:] = np.array(
            [(-1/400) * t + 3 for t in range(800, max_length+1)],
            dtype=np.float32)
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
                assembled_data[i, velocity_index, :] = \
                    behaviors['velocity'][ds_index]
                assembled_data[i, pumping_index, :] = \
                    behaviors['pumping'][ds_index]
                assembled_data[i, head_index, :] = \
                    behaviors['head_angle'][ds_index]

                # put heat-stim column immediately after neural columns
                if ds in heatstim_datasets:
                    assembled_data[i, num_neurons, :] = heatstim_encoding
                else:
                    assembled_data[i, num_neurons, :] = noheatstim_encoding

    return assembled_data, unique_datasets


def assemble_gfp(
    neuron_classes,
    behavior_index_dict,
    max_length=1600,
    num_behaviors=3
):
    """
    Create high-dimensional data arrays that contain
    the GFP signals of given neuron classes and standard behaviors.
    """

    all_datasets = []
    nogfp_datasets = []

    all_outputs = {neuron_class: {} for neuron_class in neuron_classes}
    nogfp_outputs = {neuron_class: {} for neuron_class in neuron_classes}

    for neuron_class in neuron_classes:
        nogfp_outputs[neuron_class] = by_class(
                neuron_class,
                behs=['velocity', 'head_angle', 'pumping'],
                exclude=['gfp'],
                LR_ops='random'
        )  # filter out the GFP datasets
        all_outputs[neuron_class] = by_class(
                neuron_class,
                behs=['velocity', 'head_angle', 'pumping'],
                LR_ops='random'
        )  # contain both gfp and no-gfp datasets
        all_datasets += all_outputs[neuron_class]['datasets']
        nogfp_datasets += nogfp_outputs[neuron_class]['datasets']

    nogfp_datasets = np.unique(nogfp_datasets).tolist()
    all_datasets = np.unique(all_datasets).tolist()
    # subtract to find the GFP datasets
    gfp_datasets = [
        ds for ds in all_datasets
        if ds not in nogfp_datasets and ds != '2025-03-15-15'
    ]
    print(f'number of GFP datasets: {len(gfp_datasets)}')

    num_inputs = len(neuron_classes) + num_behaviors
    velocity_index = behavior_index_dict['velocity']
    pumping_index = behavior_index_dict['pumping']
    head_index = behavior_index_dict['head_angle']

    assembled_data = np.zeros((
        len(gfp_datasets),
        num_inputs,
        max_length))

    # assemble traces in the order of gfp_datasets
    for i, ds in enumerate(gfp_datasets):

        print(f'Exacting from {ds}')
        for j, neuron_class in enumerate(neuron_classes):

            output = all_outputs[neuron_class]
            # check if neuron_class is contained in this dataset
            if ds in output['datasets']:

                ds_index = output['datasets'].index(ds)
                assembled_data[i, j, :] = output['neuron_traces'][ds_index]

                behaviors = output['behavior_traces']
                assembled_data[i, velocity_index, :] = \
                    behaviors['velocity'][ds_index]
                assembled_data[i, pumping_index, :] = \
                    behaviors['pumping'][ds_index]
                assembled_data[i, head_index, :] = \
                    behaviors['head_angle'][ds_index]

    return assembled_data, gfp_datasets


def shuffle(ds_name, num_neurons, shuffle_type, random_seed):

    """ Shuffle behaviors of the existing datasets by mismatching across
    animals. """

    base = '/store1/alicia/attention_predict/data'
    data = np.load(f'{base}/{ds_name}.npy')
    shuffled_data = deepcopy(data)
    num_datasets, num_vars, _ = data.shape
    num_inputs = num_vars // 2
    np.random.seed(random_seed)

    # shuffle behavior across animals
    # E.g. B1, B2, B3 paired with A3, A1, A2
    if shuffle_type == 'behavior-by-animal':
        variable_indices =  [i * 2 for i in list(range(num_neurons, num_inputs))]
        print(f'{len(variable_indices)} variable indices: {variable_indices}')

    # shuffle neurons across animals
    # E.g. N1, N2, N3 paired with A3, A1, A2
    if shuffle_type == 'neuron-by-animal':
        variable_indices = [i * 2 for i in range(num_neurons)]
        print(f'{len(variable_indices)} variable indices: {variable_indices}')

    # shuffle both neurons and behaviors across animals
    if shuffle_type == 'all-by-animal':
        variable_indices = [i * 2 for i in range(num_inputs)]
        print(f'{len(variable_indices)} variable indices: {variable_indices}')

    # shuffle neurons across animals & shuffle the ordering of neurons
    if shuffle_type == 'neuron-by-animal-celltype':
        variable_indices = [i * 2 for i in range(num_neurons)]
        variable_2nd_channel_indices = [i + 1 for i in variable_indices]
        print(f'{len(variable_indices)} variable indices: {variable_indices}')

    # shuffle both neurons & behaviors across aniamls
    # and shuffle how cell types are lined up
    if shuffle_type == 'all-by-animal-celltype':
        variable_indices = [i * 2 for i in range(num_inputs)]
        variable_2nd_channel_indices = [i + 1 for i in variable_indices]
        print(f'{len(variable_indices)} variable indices: {variable_indices}')

    # shuffle across animals
    if shuffle_type in [
            'behavior-by-animal',
            'neuron-by-animal',
            'neuron-by-animal-celltype',
            'all-by-animal',
            'all-by-animal-celltype'
    ]:
        for i in variable_indices:
            shuffle_indices = np.random.choice(
                    list(range(num_datasets)), num_datasets, replace=True)
            shuffled_data[:, i, :] = data[shuffle_indices, i, :]
            shuffled_data[:, i+1, :] = data[shuffle_indices, i+1, :]

    # shuffle across cell types
    if shuffle_type in [
            'neuron-by-animal-celltype',
            'all-by-animal-celltype'
    ]:
        for i in range(num_datasets):
            shuffle_indices = np.random.choice(
                variable_indices, len(variable_indices), replace=True)
            shuffle_2nd_channel_indices = [i + 1 for i in shuffle_indices]
            # print(f'shuffle neuron 1st channel: {shuffle_indices}')
            # print(f'shuffle neuron 2nd channel: {shuffle_2nd_channel_indices}')

            shuffled_data[i, variable_indices, :] = data[i, shuffle_indices, :]
            shuffled_data[i, variable_2nd_channel_indices, :] = \
                data[i, shuffle_2nd_channel_indices, :]

    return shuffled_data


if __name__ == '__main__':
    ### generating data with full set of behavioral variables
    # neuron_classes = ['SMDD', 'SAADL', 'SAADR', 'SAAV', 'M3', 'M4', 'MI',
    # 'AVB', 'RIB', 'RME', 'RMEV', 'RMED', 'URYD', 'URYV']
    # neuron_classes = ['M3']
    # num_neurons = len(neuron_classes)
    # num_body_angles = 30
    # behavior_index_dict = {
    #         # heat-stim goes first
    #         'velocity': num_neurons + 1,
    #         'pumping': num_neurons + 2,
    #         'head_angle': num_neurons + 3,
    #         'body_angles': list(range(num_neurons+3,
    #         num_neurons+3+num_body_angles))
    # }
    ### generating data with only standard behaviors
    # neuron_classes = ['AVA']
    # num_neurons = 1

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
    # num_neurons = len(fig4_neuron_classes)
    # behavior indices if incorporating heat-stim
    # behavior_index_dict = {
    #     'velocity': num_neurons + 1,
    #     'pumping': num_neurons + 2,
    #     'head_angle': num_neurons + 3
    # }
    # assembled_data, unique_datasets = assemble_std_beh(
    #         fig4_neuron_classes,
    #         behavior_index_dict)
    # behavior indices if heat-stim is not incorporated
    # behavior_index_dict = {
    #     'velocity': num_neurons,
    #     'pumping': num_neurons + 1,
    #     'head_angle': num_neurons + 2
    # }
    # assembled_data, all_datasets = assemble_gfp(
    #     fig4_neuron_classes,
    #     behavior_index_dict
    # )

    # test assembling 8 behaviors
    # neuron_classes = [
    #         'RIB', 'RIC', 'RID', 'AUA', 'AVJ', 'AVK', 'AIM', 'AIY',
    #         # 'AVA', 'AVE', 'AIB', 'RIM', 'RIV', 'ADA', 'AVD', 'RIA', 'AVH',
    #         # 'AIN', 'AIZ', 'URB', 'ALA', 'RMG', 'RMD', 'RMDD', 'RMDV', 'RME',
    #         # 'RMEV', 'RMED', 'SAAV', 'SMDV', 'IL1L', 'IL1R', 'IL1D', 'IL1V',
    #         # 'URYD', 'URYV', 'BAG', 'ASG', 'CEPD', 'CEPV', 'OLL', 'OLQD',
    #         # 'OLQV', 'IL2L', 'IL2R', 'IL2D', 'IL2V', 'URAD', 'URAV', 'ADE',
    #         # 'FLP', 'AQR', 'URX', 'ADL', 'ASH', 'ASEL', 'ASER', 'AWA', 'AWB',
    #         # 'AWC', 'I1', 'I2', 'I3', 'NSM', 'M1', 'M3', 'M4', 'M5', 'MC', 'MI'
    # ]
    # num_neurons = len(neuron_classes)
    # behavior_index_dict = {
    #         'velocity': num_neurons + 1,
    #         'pumping': num_neurons + 2,
    #         'head_angle': num_neurons + 3,
    #         'body_angle1': num_neurons + 4,
    #         'body_angle2': num_neurons + 5,
    #         'body_angle3': num_neurons + 6,
    #         'turning_rate': num_neurons + 7,
    #         'worm_angle': num_neurons + 8,
    #     }
    # assembled_data, all_datasets = assemble_all(
    #         neuron_classes,
    #         behavior_index_dict)
    # print(f'assembled_data: {assembled_data.shape}')
    # print(f'all datasets: {len(all_datasets)}')


    ### test shuffling animals
    ds_name = 'data0626/data0626-00e_norm0505_train'
    num_neurons = 70
    random_seed = 729
    shuffle_type = 'all-by-animal'
    shuffle(ds_name, num_neurons, shuffle_type, random_seed)
