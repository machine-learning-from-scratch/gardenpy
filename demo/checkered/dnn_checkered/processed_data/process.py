r"""
Data processing for the DNN application on checkered or non-checkered pixels.
"""

import os
import shutil

import pandas as pd

from tqdm import tqdm


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
    'checkered',
    'data'
)

if os.listdir(data_root) != ['labels.csv', 'values.csv']:
    # clear root
    shutil.rmtree(data_root)
    os.makedirs(data_root)

    # create data lists
    values = []
    labels = []

    # generate data
    for i in tqdm(range(4), ncols=100, desc='generating'):
        values.append([int(i > 1), int(i % 2 != 0)])
        labels.append([int(values[i][0] != values[i][1]), int(values[i][0] == values[i][1])])

    # make data dataframe
    values = pd.DataFrame(values, columns=['pixel-1', 'pixel-2'])
    labels = pd.DataFrame(labels, columns=['checkered', 'non-checkered'])

    # save data
    values.to_csv(os.path.join(data_root, 'values.csv'), index=False)
    labels.to_csv(os.path.join(data_root, 'labels.csv'), index=False)

for file in os.listdir(data_root):
    shutil.copy(os.path.join(data_root, file), save_root)
