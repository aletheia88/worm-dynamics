from tqdm import tqdm
from wormdynamics.info import *
import glob
import h5py
import json
import numpy as np


def filter_pumping_datasets(remove_heatstim: bool, remove_lowvar: bool):
    """ Filter datasets to retain only those where the difference between the 
    75th percentile and the 25th percentile of the pumping rate exceeds 0.5. """

    # 104 dataset paths in total
    dataset_paths = glob.glob("/storage/fs/store1/alicia/transformer/MC/*.json") + \
            glob.glob("/storage/fs/store1/alicia/transformer/all/*.json")
    filtered_dataset_paths = []
    unique_datasets = []

    for dataset_path in tqdm(dataset_paths):

        # remove heatstim datasets
        if remove_heatstim:

            if "all" in dataset_path:
                if any(dataset in dataset_path for dataset in HEATSTIM):
                    continue
                else:
                    with open(dataset_path, "r") as f:
                        data = json.load(f)
                        pumping_rates = np.array(data["pumping"], dtype=np.float32)
            else:
                with open(dataset_path, "r") as f:
                    data = json.load(f)
                if not data["heatstim"]:
                    pumping_rates = np.array(data["pumping"], dtype=np.float32)

            if remove_lowvar:
                if np.percentile(pumping_rates, 75) - \
                        np.percentile(pumping_rates, 25) > 0.5:

                    dataset_name = dataset_path.split('/')[-1].split('.json')[0]
                    if dataset_name not in unique_datasets:
                        filtered_dataset_paths.append(dataset_path)
                        unique_datasets.append(dataset_name)
            else:
                raise NotImplementedError("filtering not yet implemented")

        # keep the no-stim part in heatstim datasets 
        else:
            raise NotImplementedError("filtering not yet implemented")

    return filtered_dataset_paths


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


def assemble_datasets(neuron_class: str, max_len: int = 1600, max_animals: int = 94):
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
        if "events" not in data.keys():
            ds_included.append(dataset_name)
        else:
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

