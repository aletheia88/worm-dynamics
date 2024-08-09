from wormdynamics.info import *
from tqdm import tqdm
import glob
import json
import numpy as np


def filter_pumping_datasets():
    """ Filter datasets to retain only those where the difference between the 
    75th percentile and the 25th percentile of the pumping rate exceeds 0.5. """

    dataset_paths = glob.glob(f"/storage/fs/store1/alicia/transformer/all/*.json")
    filtered_datasets = []

    for dataset_path in tqdm(dataset_paths):

        with open(dataset_path, "r") as f:
            data = json.load(f)
            pumping_rates = np.array(data["pumping"], dtype=np.float32)

        if any(dataset in dataset_path for dataset in HEATSTIM):
            continue
        if np.percentile(pumping_rates, 75) - np.percentile(pumping_rates, 25) > 0.5:
            filtered_datasets.append(dataset_path)

    return filtered_datasets
