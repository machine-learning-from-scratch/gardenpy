"""
Data generator for MNIST data
"""

import os
import shutil

from torchvision import datasets
from torchvision.transforms import ToTensor

# set root and required files
root = os.path.dirname(__file__)
required_files = [
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte',
    'train-images-idx3-ubyte',
    't10k-images-idx3-ubyte.gz',
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    'train-labels-idx1-ubyte',
    't10k-labels-idx1-ubyte.gz'
]

# list existing files
existing_files = []
if os.path.exists(os.path.join(root, 'MNIST', 'raw')):
    existing_files = os.listdir(os.path.join(root, 'MNIST', 'raw'))

# check if data is valid
if required_files != existing_files:
    # clear root
    # shutil.rmtree(root)
    # os.makedirs(root)

    # load raw data
    training_data = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=ToTensor()
    )
