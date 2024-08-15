from tqdm import tqdm
from wormdynamics.info import *
import glob
import h5py
import json
import numpy as np


def write_to_json(neuron_class: str, file_path: str):

    trace_dict = assemble_datasets(neuron_class)
    std_beh = ["velocity", "head_angle", "pumping"]

    for n, dataset in tqdm(enumerate(trace_dict["datasets"])):

        # create a new json file
        json_dict = {}
        json_dict["trace_original"] = trace_dict["trace_original"][n].tolist()
        json_dict["velocity"] = trace_dict["behavior"][n, :, 0].tolist()
        json_dict["head_angle"] = trace_dict["behavior"][n, :, 1].tolist()
        json_dict["pumping"] = trace_dict["behavior"][n, :, 2].tolist()

        heatstim_dict = trace_dict["datasets"][dataset]
        json_dict["heatstim"] = heatstim_dict["heatstim"]
        if json_dict["heatstim"]:
            json_dict["stim_begin_confocal"] = heatstim_dict["stim_begin_confocal"]
        else:
            json_dict["stim_begin_confocal"] = -1

    with open(f"{file_path}/{dataset}.json", "w") as f:
        json.dump(json_dict, f, indent=4)


def filter_by_pumping(cutoff):

    output = assemble_all("MC")
    filtered_trace_original = []
    filtered_behavior = []
    filtered_datasets = []

    for idx, dataset_name in enumerate(output["datasets"]):

        pumping = np.percentile(output["behavior"][idx, :, 2], cutoff)
        if pumping == 0:
            filtered_trace_original.append(output['trace_original'][idx])
            filtered_behavior.append(output['behavior'][idx])
            filtered_datasets.append(dataset_name)

    return {
        "trace_original": np.array(filtered_trace_original),
        "behavior": np.array(filtered_behavior),
        "datasets": filtered_datasets
    }


def assemble_all(neuron_class: str, max_len: int = 1600, max_animals: int = 94):
    """ Combine neural trace and behavior data from two output dictionaries based on
    unique datasets.

    This function takes two dictionaries, each containing keys 'trace_original',
    'behavior', and 'datasets'. It identifies unique datasets across both dictionaries,
    and combines the 'trace_original' and 'behavior' data corresponding to these unique
    datasets. The combined data is assembled into a new dictionary.

    Args:
        output1 (dict): First dictionary containing the keys:
            - 'trace_original' (np.ndarray): Neural trace data with shape (n1, 1600)
              where n1 is the number of entries.
            - 'behavior' (np.ndarray): Behavior data with shape (n1, 1600, 3).
            - 'datasets' (list): List of dataset identifiers.
        output2 (dict): Second dictionary similar to output1 but may contain different
        datasets and data lengths.

    Returns:
        dict: A dictionary with the following structure:
            - 'trace_original' (np.ndarray): Combined neural trace data from unique datasets.
            - 'behavior' (np.ndarray): Combined behavior data from unique datasets.
            - 'datasets' (list): List of all unique datasets.

    Assumes:
        - Each dataset identifier in 'datasets' is unique within its respective
          dictionary but may overlap between dictionaries.
        - The 'trace_original' and 'behavior' data for each dataset are aligned by their
          first dimension in the respective dictionaries.
        - The function does not handle duplicate datasets across the dictionaries; it
          assumes dataset identifiers are unique or completely non-overlapping.

    Example:
        >>> output1 = {
            "trace_original": np.random.rand(22, 1600),
            "behavior": np.random.rand(22, 1600, 3),
            "datasets": ['ds1', 'ds2']
        }
        >>> output2 = {
            "trace_original": np.random.rand(25, 1600),
            "behavior": np.random.rand(25, 1600, 3),
            "datasets": ['ds3', 'ds4']
        }
        >>> combined_data = combine_unique_datasets(output1, output2)
        >>> print(combined_data['datasets'])
        ['ds1', 'ds2', 'ds3', 'ds4']
        >>> print(combined_data['trace_original'].shape)
        (47, 1600)
        >>> print(combined_data['behavior'].shape)
        (47, 1600, 3) """

    files_path = "/home/alicia/store1/alicia/transformer/all"
    output1 = assemble_traces_from_wormwideweb(files_path, "MC", max_len)
    output2 = assemble_datasets("MC", max_len, max_animals)

    # find unique datasets
    unique_datasets = set(output1['datasets']).union(set(output2['datasets']))

    # initialize arrays to store combined data
    combined_trace_original = []
    combined_behavior = []
    combined_datasets = []

    # filter and combine data for unique datasets
    for ds in unique_datasets:
        if ds in output1['datasets']:
            idx = output1['datasets'].index(ds)
            combined_trace_original.append(output1['trace_original'][idx])
            combined_behavior.append(output1['behavior'][idx])
            combined_datasets.append(ds)
        if ds in output2['datasets']:
            idx = output2['datasets'].index(ds)
            combined_trace_original.append(output2['trace_original'][idx])
            combined_behavior.append(output2['behavior'][idx])
            combined_datasets.append(ds)

    # convert lists to numpy arrays
    combined_trace_original = np.array(combined_trace_original)
    combined_behavior = np.array(combined_behavior)

    # final combined dictionary
    final_output = {
        "trace_original": combined_trace_original,
        "behavior": combined_behavior,
        "datasets": combined_datasets
    }

    print(f"Found total {len(combined_datasets)} animals!")

    return final_output


def assemble_datasets(neuron_class: str, max_len: int, max_animals: int):
    """ Extract neural and behavioral traces from all datasets containing the specified
    neuron class.

    Args:
        neuron_class (str): The class of the neuron to extract data for, e.g., 'MC' or 'AVA'.

    Returns:
        Dict[str, np.ndarray]: A dictionary with keys 'trace_original' and 'behavior'.
            - 'trace_original': A multidimensional numpy array with shape (max_length,).
            - 'behavior': A multidimensional numpy array with shape (num_animals,
              max_length, 3),
              where each entry corresponds to a different behavioral feature over time.
    Note:
        This code is provided courtesy of Alex Hister. """

    prj_data = {}

    prj_data['kfc'] = {}
    prj_data['rim'] = {}

    prj_data['kfc']['labels_path'] = KFC_LABELS_PATH
    prj_data['kfc']['processed_h5'] = KFC_PROCESSED_H5_PATH
    prj_data['kfc']['structure_path'] = KFC_STRUCTURE_PATH

    prj_data['rim']['labels_path'] = RIM_LABELS_PATH
    prj_data['rim']['processed_h5'] = RIM_PROCESSED_H5_PATH
    prj_data['rim']['structure_path'] = RIM_STRUCTURE_PATH

    std_beh = []
    ys = []
    reversals = []
    ds_included = []

    for prj, data in prj_data.items():

        label_data = h5_to_dict(data['labels_path'])
        structure_data = h5_to_dict(data['structure_path'])

        ds_of_interest = [ds for ds, value in structure_data.items() if
                          len(set(["neuropal"] if prj=='kfc' else
                                  ['wt']).intersection(set(value["Tags"]))) > 0]

        neurons = [neuron for neuron in label_data.keys() if neuron.startswith(neuron_class)]
        print(neurons)
        for neuron in neurons:
            for ds, items in label_data[neuron].items():
                if ds in ds_of_interest and items['confidence'] > 3.5:
                    file_path = f"{data['processed_h5']}/{ds}-data.h5"
                    loaded_data = h5_to_dict(file_path)

                    if(len(loaded_data['gcamp']['trace_array_original'][:, 0]) < max_len):
                        continue
                    if(len(ys) == max_animals):
                        break

                    if ds not in ds_included:

                        timing = loaded_data["timing"]

                        if "stim_begin_confocal" not in timing.keys():

                            ds_included.append(ds)

                            std_beh.append(np.array([
                                loaded_data['behavior']["velocity"][:max_len],
                                loaded_data['behavior']["head_angle"][:max_len] * \
                                        (-1 if structure_data[ds]['Flipped'] else 1),
                                loaded_data['behavior']["pumping"][:max_len]]).T)

                            ys.append(
                                loaded_data['gcamp']['trace_array_original'][:max_len,
                                                                             items['index']-1])

                            reversals.append(loaded_data['behavior']['reversal_events'].T)

    print(f"Found {len(ds_included)} animals")

    return {
            "trace_original": np.array(ys),
            "behavior": np.array(std_beh),
            "datasets": ds_included
    }


def assemble_traces_from_wormwideweb(files_path, neuron):

    # find neuron index in each dataset
    dataset_paths = glob.glob(f"{files_path}/*.json")
    ys = []
    std_beh = []
    ds_included = []

    for dataset_path in dataset_paths:

        dataset_name = dataset_path.split("/")[-1].split('.')[0]

        with open(dataset_path, "r") as f:
            data = json.load(f)

        # remove heatstim datasets
        if "events" in data.keys():
            continue

        for n_id, info_dict in data["labeled"].items():

            if info_dict["label"].startswith(neuron):
                # add neural and behavioral traces
                neuron_id = int(n_id) - 1
                trace_original = np.array(
                        data["trace_original"],
                        dtype=np.float32)[:1600, neuron_id]
                pumping_rates = np.array(data["pumping"], dtype=np.float32)

                ys.append(trace_original)
                std_beh.append(
                        np.array([
                            data["velocity"][:1600],
                            data["head_curvature"][:1600],
                            data["pumping"][:1600]]).T)
                ds_included.append(dataset_name)

    print(f"Found {len(ds_included)} animals!")

    return {
        "trace_original": np.array(ys),
        "behavior": np.array(std_beh),
        "datasets": ds_included
    }


def h5_to_dict(file_path: str) -> dict:
    """ Converts an HDF5 file to a nested dictionary.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        dict: A nested dictionary representing the contents of the HDF5 file.

    Note:
        This code is provided courtesy of Alex Hister. """

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
    files_path = "/home/alicia/store1/alicia/transformer/all"
    output = assemble_traces_from_wormwideweb(files_path, "MC")
    print(output["trace_original"].shape)
    print(output["behavior"].shape)
    print(output["datasets"])
    output = assemble_datasets("MC")
    print(output["trace_original"].shape)
    print(output["behavior"].shape)
    print(output["datasets"])
    """
    mc_output = assemble_datasets("MC")
    mcl_output = assemble_datasets("MCL")
    mcr_output = assemble_datasets("MCR")

    all_datasets = list(mc_output["datasets"].keys()) + \
            list(mcl_output["datasets"].keys()) + \
            list(mcr_output["datasets"].keys())
    print(len(all_datasets))
    print(all_datasets)
    print(len(np.unique(all_datasets)))
    print(np.unique(all_datasets))
    """

