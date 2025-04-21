"""
Data processing for the feedforward neural network for MNIST
"""

import os

import numpy as np
import pandas as pd

from colorama import Fore, Style
from tqdm import tqdm

from torchvision import datasets

color = f'{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}'

# file locations
save_root = os.path.dirname(__file__)
data_root = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            )
        )
    ),
    'data',
    'mnist',
    'data'
)


def process_data(save=True, cases_per_file = 10000, values_file=os.path.join(save_root, 'values'), labels_file=os.path.join(save_root, 'labels')):
    # load dataset
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True)
    raw_values = np.append(train_dataset.data.numpy(), test_dataset.data.numpy(), axis=0)
    raw_labels = np.append(train_dataset.targets.numpy(), test_dataset.targets.numpy(), axis=0)

    file_count = 0

    # process data
    values = []
    labels = []
    for i in tqdm(range(len(raw_values)), ncols=100, desc='processing', bar_format=color):
        values.append(np.array(np.divide(raw_values[i].flatten().tolist(), 255)))
        out = np.zeros(10)
        out[raw_labels[i]] = 1
        out = np.array(out)
        labels.append(out)

        if (i + 1) % cases_per_file == 0 or (i + 1) == len(raw_values):
            # create columns
            value_columns = list(range(len(values[0])))
            label_columns = list(range(len(labels[0])))

            # make list pandas dataframe
            values = pd.DataFrame(np.array(values), columns=value_columns)
            labels = pd.DataFrame(np.array(labels), columns=label_columns)

            file_count += 1

            if save:
                # save processed data
                values.to_csv(f'{values_file}{file_count}.csv', index=False)
                labels.to_csv(f'{labels_file}{file_count}.csv', index=False)
            values = []
            labels = []

if __name__ == '__main__':
    process_data()
