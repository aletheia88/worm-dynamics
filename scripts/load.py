from copy import deepcopy
from flv_utils import by_class, by_multiclass
from typing import List
import glob
import h5py
import json
import numpy as np


def assemble_all(neuron_classes, behavior_index_dict, max_len=1600):

    datasets = []
    all_outputs = {neuron_class: {} for neuron_class in neuron_classes}
    for neuron_class in neuron_classes:
        all_outputs[neuron_class] = by_class(neuron_class)
        datasets += all_outputs[neuron_class]['datasets']

    unique_datasets = np.unique(datasets).tolist()
    num_behaviors = len(behavior_index_dict)
    num_neurons = len(neuron_classes)
    num_columns = num_behaviors + num_neurons

    velocity_index = behavior_index_dict['velocity']
    pumping_index = behavior_index_dict['pumping']
    angle_index = behavior_index_dict['head_angle']

    assembled_data = np.zeros((
        len(unique_datasets),
        num_neurons + num_behaviors,
        max_len))

    for i, ds in enumerate(unique_datasets):

        for j, neuron_class in enumerate(neuron_classes):

            output = all_outputs[neuron_class]

            if ds in output['datasets']:

                ds_index = output['datasets'].index(ds)
                assembled_data[i, j, :] = output['neuron_traces'][ds_index]

                behaviors = output['behavior_traces']
                assembled_data[i, velocity_index, :] = behaviors['velocity'][ds_index]
                assembled_data[i, pumping_index, :] = behaviors['pumping'][ds_index]
                assembled_data[i, angle_index, :] = behaviors['head_angle'][ds_index]

    # normalize neural traces across animals
    for i in range(num_neurons):
        all_neural_traces = assembled_data[:, i, :]
        assembled_data[:, i, :] = all_neural_traces / (2 * np.mean(all_neural_traces))-1

    # normalize velocity traces
    all_velocity_traces = assembled_data[:, velocity_index, :]
    assembled_data[:, velocity_index, :] = all_velocity_traces * 10
    # normalize pumping
    all_pumping_traces = assembled_data[:, pumping_index, :]
    assembled_data[:, pumping_index, :] = all_pumping_traces * 2 - 1
    # normalize head angle
    all_head_angle_traces = assembled_data[:, angle_index, :]
    min_val = np.min(all_head_angle_traces)
    max_val = np.max(all_head_angle_traces)
    normalized_head_angle = 2 * ((all_head_angle_traces - min_val) /
                                 (max_val - min_val)) - 1
    assembled_data[:, angle_index, :] = normalized_head_angle

    return assembled_data, unique_datasets


def shuffle(ds_name, num_neurons, random_seed, shuffle_type='behavior'):

    """ Shuffle behaviors of the existing datasets by mismatching across animals. """

    base = '/store1/alicia/attention_predict/data'
    data = np.load(f'{base}/{ds_name}.npy')
    datasets = np.load(f'{base}/{ds_name}_ds.npy')
    # np.random.seed(random_seed)

    num_datasets = len(datasets)
    num_inputs = data.shape[1]

    shuffle_indices = np.random.choice(list(range(num_datasets)), num_datasets)
    shuffled_datasets = datasets[shuffle_indices]
    shuffled_data = deepcopy(data)

    if shuffle_type == 'behavior':
        behavior_indices = list(range(num_neurons, num_inputs))
        for i in behavior_indices:
            shuffled_data[:, i, :] = data[shuffle_indices, i, :]

    elif shuffle_type == 'neuron':
        neuron_indices = list(range(num_neurons))
        for i in neuron_indices:
            shuffled_data[:, i, :] = data[shuffle_indices, i, :]

    elif shuffle_type == 'all':
        for i in range(num_inputs):
            np.random.shuffle(shuffled_data[:, i, :])

    return shuffled_data, shuffled_datasets


def load_behaviors(max_len: int):

    kfc_labels = "/store1/prj_jax/Aggregated_Traces_h5/dict_neuropal_label_prj_kfc.h5"
    kfc_h5 = "/store1/prj_kfc/data/processed_h5"
    kfc_structure = "/store1/prj_kfc/Structured_Data_Info.h5"

    rim_labels = "/store1/prj_rim/Decoding_Data/Aggregated_Traces_h5/dict_neuropal_label_updated.h5"
    rim_h5 = "/store1/prj_rim/processed_h5"
    rim_structure = "/store1/prj_rim/Decoding_Data/Structured_Data_Info.h5"

    prj_data = {}
    prj_data['kfc'] = {}
    prj_data['rim'] = {}

    prj_data['kfc']['labels_path'] = kfc_labels
    prj_data['kfc']['processed_h5'] = kfc_h5
    prj_data['kfc']['structure_path'] = kfc_structure

    prj_data['rim']['labels_path'] = rim_labels
    prj_data['rim']['processed_h5'] = rim_h5
    prj_data['rim']['structure_path'] = rim_structure

    std_behaviors = []
    reversals = []
    datasets = []

    for prj, data in prj_data.items():

        label_data = h5_to_dict(data['labels_path'])
        structure_data = h5_to_dict(data['structure_path'])

        # Find datasets of certain types. Currently only kfc and rim are considered
        ds_of_interest = [
            ds for ds, value in structure_data.items() if
            len(set(["neuropal"] if prj == "kfc" else ["wt"]).intersection(set(value["Tags"]))) > 0
        ]

        print(f'Total NeuroPAL datasets: {len(ds_of_interest)}')

        for ds in ds_of_interest:
            file_path = f"{data['processed_h5']}/{ds}-data.h5"
            loaded_data = h5_to_dict(file_path)
            behaviors = loaded_data['behavior']

            if (len(behaviors['velocity']) >= max_len and
                len(behaviors['head_angle']) >= max_len):
                # len(behaviors['pumping']) >= max_len):

                flip_factor = -1 if structure_data[ds]['Flipped'] else 1
                std_behaviors.append(np.array([
                                behaviors["velocity"][:max_len],
                                # behaviors["pumping"][:max_len],
                                behaviors["head_angle"][:max_len] * flip_factor]).T)
                datasets.append(ds)
                reversals.append(behaviors['reversal_events'].T)

    std_behaviors = np.array(std_behaviors)

    return std_behaviors, reversals, datasets


def assemble(
        neuron_classes: List[str],
        max_len: int,
        max_animals: int
):
    """ Extract neural and behavioral traces from all datasets containing the specified
    neuron classes.

    Args:
        neuron_class (List[str]): A list of neuron classes to extract neural and
            behavioral data.
        max_len (int): Maximum length of neural GCaMP trace.

    Returns:
        gcamp_traces (Dict[str: Dict[str: np.array]]): A dictionary with a collection of
            GCaMP traces from each dataset for each neuron, formatted as follows
            E.g. {
                    'AVAL': {
                        '2023-06-24-11': array([...]),
                        '2023-07-01-01': array([...]), ...
                    },
                    'AVAR': {
                        '2023-06-24-02': array([...]),
                        '2023-06-24-11': array([...]), ...
                    },
                    'MCL': {
                        '2023-06-24-02: array([...]),
                        '2023-06-24-11': array([...]), ...
                    }, ...
                }
        std_behaviors (Dict[str: np.array]): A dictionary that maps each dataset to
            behavior arrays of shape (max_len, 3), where columns 0, 1, 2 each correspond
            to 'velocity', 'head angle', and 'pumping'.
        reversals (Dict[str: np.array]): A dictionary that maps each dataset to
            its recorded reversals events.
    """
    # find datasets from projects KFC and RIM that contain all neuron classes
    filtered_datasets, prj_data, neurons = filter_datasets(neuron_classes)

    std_behaviors = {}
    gcamp_traces = {}
    reversals = {}

    # assemble neural data
    for prj, data in prj_data.items():

        label_data = h5_to_dict(data['labels_path'])
        structure_data = h5_to_dict(data['structure_path'])

        for i, neuron in enumerate(neurons):

            if not neuron in label_data.keys():
                break

            gcamp_traces[neuron] = {}

            for ds, items in label_data[neuron].items():

                # keep only datasets with all neuron classes labeled confidently
                if ds in filtered_datasets and items['confidence'] > 3.5:

                    file_path = f"{data['processed_h5']}/{ds}-data.h5"
                    loaded_data = h5_to_dict(file_path)
                    timing = loaded_data["timing"]

                    # exclude heat-stim datasets
                    if "stim_begin_confocal" not in timing.keys():

                        gcamp_data = loaded_data['gcamp']['trace_array_original']
                        gcamp_data_length = len(gcamp_data[:, 0])

                        if(gcamp_data_length < max_len):
                            continue

                        if(len(gcamp_data) == max_animals):
                            break

                        neuron_trace = gcamp_data[:max_len, items['index'] - 1]
                        gcamp_traces[neuron][ds] = neuron_trace

                        behaviors = loaded_data['behavior']
                        reversals[ds] = behaviors['reversal_events'].T
                        flip_factor = -1 if structure_data[ds]['Flipped'] else 1
                        std_behaviors[ds] = np.array([
                                    behaviors["velocity"][:max_len],
                                    behaviors["head_angle"][:max_len] * flip_factor,
                                    behaviors["pumping"][:max_len]]).T

    return gcamp_traces, std_behaviors, reversals


def filter_datasets(neuron_classes: List[str]):

    kfc_labels = "/store1/prj_jax/Aggregated_Traces_h5/dict_neuropal_label_prj_kfc.h5"
    kfc_h5 = "/store1/prj_kfc/data/processed_h5"
    kfc_structure = "/store1/prj_kfc/Structured_Data_Info.h5"

    rim_labels = "/store1/prj_rim/Decoding_Data/Aggregated_Traces_h5/dict_neuropal_label_updated.h5"
    rim_h5 = "/store1/prj_rim/processed_h5"
    rim_structure = "/store1/prj_rim/Decoding_Data/Structured_Data_Info.h5"

    prj_data = {}
    prj_data['kfc'] = {}
    prj_data['rim'] = {}

    prj_data['kfc']['labels_path'] = kfc_labels
    prj_data['kfc']['processed_h5'] = kfc_h5
    prj_data['kfc']['structure_path'] = kfc_structure

    prj_data['rim']['labels_path'] = rim_labels
    prj_data['rim']['processed_h5'] = rim_h5
    prj_data['rim']['structure_path'] = rim_structure

    for prj, data in prj_data.items():

        label_data = h5_to_dict(data['labels_path'])
        structure_data = h5_to_dict(data['structure_path'])

        ds_of_interest = [
                ds for ds, value in structure_data.items()
                if len(set(["neuropal"]
                if prj=='kfc' else ['wt']).intersection(set(value["Tags"]))) > 0]

        # collect all neurons under the same class
        neuron_list = []
        for neuron_class in neuron_classes:
            neuron_list += [neuron for neuron in label_data.keys()
                       if neuron.startswith(neuron_class)]

        # find all datasets containing each neuron 
        neuron_datasets = {
                neuron: [ds for ds in label_data[neuron].keys()]
                for neuron in neuron_list
        }

        # reformat the dictionary to {<dataset>: [<neuronA>, <neuronB>, ...]}
        dataset_neurons = {}
        for neuron, datasets in neuron_datasets.items():

            for dataset in datasets:

                if dataset not in dataset_neurons.keys():
                    dataset_neurons[dataset] = [neuron]
                else:
                    dataset_neurons[dataset].append(neuron)

        # check if each dataset contains all neuron classes
        contained_neuron_classes = {dataset: [] for dataset in dataset_neurons.keys()}
        for dataset, neurons in dataset_neurons.items():

            for neuron in neurons:

                for neuron_class in neuron_classes:

                    if (neuron.startswith(neuron_class) and
                        neuron_class not in contained_neuron_classes[dataset]):

                        contained_neuron_classes[dataset].append(neuron_class)


        # get datasets that contain all neuron classes and are NeuroPAL
        filtered_datasets = [
                ds for ds, classes in contained_neuron_classes.items()
                if len(classes) == len(neuron_classes) and ds in ds_of_interest
        ]

    return filtered_datasets, prj_data, neuron_list


def filter_dataset_paths(neuron_classes: List[str]):

    dataset_paths = glob.glob(f"/home/alicia/store1/alicia/wormwideweb/*.json")
    # find datasets that contain all neuron classes
    contained_neuron_classes = {ds_path.split("/")[-1].split('.')[0]: []
                                for ds_path in dataset_paths}

    for dataset_path in dataset_paths:

        dataset_name = dataset_path.split("/")[-1].split('.')[0]

        with open(dataset_path, "r") as f:
            data = json.load(f)

        for n_id, info_dict in data["labeled"].items():

            for neuron_class in neuron_classes:

                neuron = info_dict["label"]
                if (neuron.startswith(neuron_class) and
                    neuron not in contained_neuron_classes):

                    contained_neuron_classes[dataset_name].append(neuron_class)

    filtered_dataset_paths = [
            ds_path for ds_path in dataset_paths
            if len(neuron_classes) == \
               len(contained_neuron_classes[ds_path.split("/")[-1].split('.')[0]])
    ]
    print(f'filtered path: {filtered_dataset_paths}')

    return filtered_dataset_paths


def load_single_neuron_class(
        neuron_class,
        source="flv",
        max_len=1600,
        max_animals=-1,
        confidence_threshold=3.5,
        verbose=False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load neuron and behavior data for a specific neuron class.

    Args:
        neuron_class (str): The class of neurons to load.
        source (str, optional): The data source. Defaults to 'flv'.
        max_len (int, optional): The maximum length of the trace. Defaults to 1600.
        max_animals (int, optional): The maximum number of animals to load. Set to -1 to
            load all animals. Defaults to -1.
        confidence_threshold (float, optional): The confidence threshold for selecting
            datasets. Defaults to 3.5.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, list]: The standardized neural
            data, the unnormalized neural data, the standardized behavior data, and the
            reversal events.
    """
    prj_data = {}

    prj_data["om"] = {}
    prj_data["flv"] = {}

    prj_data["om"]["kfc"] = {}
    prj_data["om"]["rim"] = {}
    prj_data["flv"]["kfc"] = {}
    prj_data["flv"]["rim"] = {}

    prj_data["om"]["kfc"]["labels_path"] = \
        "/om/user/hiserale/data/prj_kfc/dict_neuropal_label_prj_kfc.h5"
    prj_data["om"]["kfc"]["processed_h5"] = \
        "/om/user/hiserale/data/prj_kfc/processed_h5"
    prj_data["om"]["kfc"]["structure_path"] = \
        "/om/user/hiserale/data/prj_kfc/Structured_Data_Info.h5"

    prj_data["om"]["rim"]["labels_path"] = \
        "/om/user/hiserale/data/prj_rim/dict_neuropal_label_prj_rim.h5"
    prj_data["om"]["rim"]["processed_h5"] = \
        "/om/user/hiserale/data/prj_rim/processed_h5"
    prj_data["om"]["rim"]["structure_path"] = \
        "/om/user/hiserale/data/prj_rim/Structured_Data_Info.h5"

    prj_data["flv"]["kfc"]["labels_path"] = \
        "/store1/prj_jax/Aggregated_Traces_h5/dict_neuropal_label_prj_kfc.h5"
    prj_data["flv"]["kfc"]["processed_h5"] = "/store1/prj_kfc/data/processed_h5"
    prj_data["flv"]["kfc"]["structure_path"] = "/store1/prj_kfc/Structured_Data_Info.h5"

    prj_data["flv"]["rim"]["labels_path"] = \
        "/store1/prj_rim/Decoding_Data/Aggregated_Traces_h5/dict_neuropal_label_updated.h5"
    prj_data["flv"]["rim"]["processed_h5"] = "/store1/prj_rim/processed_h5"
    prj_data["flv"]["rim"]["structure_path"] = \
        "/store1/prj_rim/Decoding_Data/Structured_Data_Info.h5"

    neuron_classes = [
        "AVB", "RIB", "RIC", "RID", "AUA", "AVJ", "AVK", "AIM", "AIY", "AIA", "AVA", "AVE", "AIB",
        "RIM", "AVL", "RIF", "RIV", "ADA", "AVD", "RMF", "RIA", "AVH", "RIR", "RIS", "RIH", "AIN",
        "RIP", "AIZ", "URB", "ALA", "RMG", "RMD", "RMDD", "RMDV", "RME", "RMEV", "RMED", "SAADL",
        "SAADR", "SAAV", "SMBV", "SMBD", "SMDV", "SMDD", "SIAV", "SIAD", "SIBV", "SIBD", "VB02", "ASJ",
        "IL1L", "IL1R", "IL1D", "IL1V", "URYD", "URYV", "BAG", "ASG", "CEPD", "CEPV", "OLL", "OLQD",
        "OLQV", "IL2L", "IL2R", "IL2D", "IL2V", "URAD", "URAV", "ADE", "FLP", "AQR", "URX", "ADL",
        "ASH", "ASEL", "ASER", "ASI", "AFD", "ASK", "AWA", "AWB", "AWC", "I1", "I2", "I3", "I4", "I5",
        "I6", "NSM", "M1", "M3", "M4", "M5", "MC", "MI"
    ]

    # Global standardization values
    v_STD = 0.06030961137253011
    θh_STD = 0.49429038957075727
    P_STD = 1.2772001409506841

    filter_qs = True # Filter out neurons with '?' in their names

    std_behaviors = []
    gcamp_traces = []
    reversals = []
    datasets = []

    for prj, data in prj_data[source].items():
        # Get neuropal labels
        label_data = h5_to_dict(data["labels_path"])
        # Get animal metadata
        structure_data = h5_to_dict(data["structure_path"])
        # Find datasets of certain types. Currently only kfc and rim are considered
        ds_of_interest = [
            ds for ds, value in structure_data.items() if
            len(set(["neuropal"] if prj == "kfc" else ["wt"]).intersection(set(value["Tags"]))) > 0
        ]
        # Find all neurons of the given class without overlapping with other classes
        neurons = [
            neuron for neuron in label_data.keys() if neuron.startswith(neuron_class) and
            (len(neuron) - len(neuron_class)) <= 1 and not neuron in set(neuron_classes) -
            {neuron_class} and (("?" != neuron[-2]) if filter_qs else True)
        ]
        for neuron in neurons:
            count = 0
            # Loop through all datasets
            for ds, items in label_data[neuron].items():
                # Check if dataset is of interest and confidence is above threshold
                if (ds in ds_of_interest and
                    items["confidence"] > confidence_threshold):

                    file_path = f"{data['processed_h5']}/{ds}-data.h5"
                    loaded_data = h5_to_dict(file_path)
                    gcamp_data = loaded_data["gcamp"]["trace_array_original"]
                    behaviors = loaded_data['behavior']

                    # Skip if the length of the trace is less than max_len
                    if (len(gcamp_data[:, 0]) < max_len):
                        continue
                    # Stop if max_animals is reached (for testing)
                    if (max_animals != -1 and len(gcamp_traces) == max_animals):
                        break

                    count += 1
                    reversals.append(behaviors['reversal_events'].T)
                    flip_factor = -1 if structure_data[ds]['Flipped'] else 1
                    std_behaviors.append(
                            np.array([
                                behaviors["velocity"][:max_len],
                                behaviors["pumping"][:max_len],
                                behaviors["head_angle"][:max_len] * flip_factor]).T)
                    gcamp_traces.append(gcamp_data[:max_len, items["index"] - 1])
                    datasets.append(ds)
            if verbose:
                print(f"{neuron} was found in {count} animals in prj_{prj}")

    std_behaviors = np.array(std_behaviors)
    gcamp_traces = np.array(gcamp_traces)

    if len(gcamp_traces) == 0:
        print(f"{neuron_class} was not found in any animals")
        return None, None, None, None

    report_list = [
            f"{neuron_class} was found in {len(gcamp_traces)} animals",
            f" with confidence > {confidence_threshold}",
            f" in {[f'prj_{prj}' for prj in prj_data[source].keys()]}"]

    if verbose:
        print(" ".join(report_list))

    return gcamp_traces, std_behaviors, reversals, datasets


def assemble_data_with_missing_neurons(neuron_classes, max_len):

    all_data = {neuron_class: {} for neuron_class in neuron_classes}

    for neuron_class in neuron_classes:

        gcamp_traces, std_behaviors, _, datasets = load_single_neuron_class(
                neuron_class, max_len=max_len)

        # Normalize GCaMP
        f_mean = np.mean(gcamp_traces)
        gcamp_traces = gcamp_traces / (2 * f_mean) - 1
        # Normalize behavior here
        # Assuming velocity corresponds to column 0
        std_behaviors[:, :, 0] = 10 * std_behaviors[:, :, 0]
        # Assuming pumping corresponds to column 1
        std_behaviors[:, :, 1] = std_behaviors[:, :, 1] / 2 - 1
        # Assuming head curvature corresponds to column 2
        head_angle = std_behaviors[:, :, 2]
        min_val = np.min(head_angle)
        max_val = np.max(head_angle)
        std_behaviors[:, :, 2] = 2 * ((head_angle - min_val) / (max_val - min_val)) - 1

        all_data[neuron_class]['gcamp_traces'] = gcamp_traces
        all_data[neuron_class]['std_behaviors'] = std_behaviors
        all_data[neuron_class]['datasets'] = datasets

    max_datasets = max([len(all_data[neuron_class]['datasets'])
                        for neuron_class in neuron_classes])

    for neuron_class in neuron_classes:
        if len(all_data[neuron_class]['datasets']) == max_datasets:
            max_neuron_class = neuron_class
            break

    print(f'{max_neuron_class} is found with the maximum number of datasets.')
    # assemble neural and behavior traces following the ordering of datasets from which
    # the most abundant neuron class is found
    assembled_datasets = all_data[max_neuron_class]['datasets']
    assembled_gcamp_traces = np.zeros((max_datasets, max_len, len(neuron_classes)))
    assembled_behaviors = all_data[max_neuron_class]['std_behaviors']
    # `dataset_neuron_info` dictionary is organized as follows:
    # {'ds1': ['neuronA', 'neuronB', 'neuronC'],
    #  'ds2': ['neuronA'], ...}
    dataset_neuron_info = {ds: [max_neuron_class] for ds in assembled_datasets}

    for dataset_index, dataset in enumerate(assembled_datasets):

        for neuron_index, neuron_class in enumerate(neuron_classes):

            neuron_class_datasets = all_data[neuron_class]['datasets']
            if neuron_class != max_neuron_class:

                if dataset in neuron_class_datasets:
                    gcamp_index = all_data[neuron_class]['datasets'].index(dataset)
                    assembled_gcamp_traces[dataset_index, :, neuron_index] = \
                        all_data[neuron_class]['gcamp_traces'][gcamp_index, :]

                    if neuron_class not in dataset_neuron_info[dataset]:
                        dataset_neuron_info[dataset].append(neuron_class)
            else:
                assembled_gcamp_traces[dataset_index, :, neuron_index] = \
                        all_data[max_neuron_class]['gcamp_traces'][dataset_index, :]

    return (assembled_gcamp_traces,
            assembled_behaviors,
            assembled_datasets,
            dataset_neuron_info)


def h5_to_dict(file_path: str) -> dict:
    """ Converts an HDF5 file to a nested dictionary.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        dict: A nested dictionary representing the contents of the HDF5 file.

    Note:
        This code is provided courtesy of Alex Hister.
    """

    def recursively_load_dict(group):
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py._hl.group.Group):
                data[key] = recursively_load_dict(item)
            elif isinstance(item, h5py._hl.dataset.Dataset):
                value = item[()]
                # Convert byte strings to regular strings
                if isinstance(value, np.ndarray):
                    if value.dtype.type is np.bytes_:
                        value = value.astype(str)
                        value = value.tolist()
                    elif value.dtype.type is np.object_:
                        value = np.array([v.decode('utf-8') for v in value])
                        value = value.tolist()
                elif isinstance(value, bytes):
                    value = value.decode('utf-8')
                data[key] = value
        return data

    with h5py.File(file_path, 'r') as file:
        return recursively_load_dict(file)


if __name__ == "__main__":

    ### Assemble neuron and behavior traces from datasets that contain common neurons 
    # neuron_classes = ["M3"]
    # max_len = 1600
    # max_animal = 100
    # gcamp_traces, std_behaviors, reversals = assemble(neuron_classes, max_len,
    # max_animal)
    # print(len(std_behaviors.keys()))
    # print(f'GCaMP traces: {gcamp_traces}')

    ### Load behaviors from all rim and kfc datasets ### 
    # max_len = 1600
    # std_behaviors, reversals, datasets = load_behaviors(max_len)
    # print(f'std behaviors: {std_behaviors.shape}')
    # print(f'reversals: {len(reversals)}')
    # print(f'datasets: {len(datasets)}')

    ### Load datasets that contain a single neuron class ###
    # neuron_class = 'RMG'
    # gcamp_traces, std_behaviors, reversals, datasets = load_single_neuron_class(
    #         neuron_class,
    #         verbose=True)
    # print(f'neural trace shape: {gcamp_traces.shape}')
    # print(f'behavior trace shape: {std_behaviors.shape}')
    # print(f'num datasets: {len(datasets)}')
    # print(f'num unique datasets: {len(np.unique(datasets))}')

    ### Assemble datasets which include missing neurons
    # neuron_classes = ['RID', 'AVE', 'RIV', 'AVD', 'AIN']
    # max_len = 1600
    # outputs = assemble_data_with_missing_neurons(neuron_classes, max_len)
    # assembled_gcamp_traces, assembled_behaviors, assembled_datasets, dataset_neuron_info = outputs
    # print(f'assembled neural traces: {assembled_gcamp_traces.shape}')
    # print(f'assembled behaviors: {assembled_behaviors.shape}')
    # print(f'assembled_datasets: {assembled_datasets}')
    # print(f'datasets found in each neuron class: {dataset_neuron_info}')
    # counts = {neuron_class: 0 for neuron_class in neuron_classes}

    # for dataset, neuron_list in dataset_neuron_info.items():

    #     for neuron_class in neuron_list:
    #         counts[neuron_class] += 1
    # print(f'neuron dataset counts: {counts}')
    # print(f'maximum unique datasets: {len(np.unique(assembled_datasets))}')

    ### Assemble traces for multiple neuron classes from all datasets
    neuron_classes = ['AVA', 'MC', 'SMDV']
    behavior_index_dict = {'velocity': 3, 'pumping': 4, 'head_angle': 5}
    assembled_data, datasets = assemble_all(neuron_classes, behavior_index_dict)
    print(f'num datasets: {len(datasets)}')
    print(f'all datasets: {datasets}')
