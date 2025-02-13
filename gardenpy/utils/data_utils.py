r"""
**Data utilities.**

Contains:
    - class`DataLoader`
"""

from typing import Union, Generator, Tuple
import numpy as np

from ..functional.objects import Tensor


class DataLoader:
    def __init__(
            self,
            data: Union[np.ndarray, list],
            labels: Union[np.ndarray, list],
            *,
            batch_size: int,
            shuffle: bool = True
    ):
        # check data and labels
        data, labels = np.array(data), np.array(labels)
        if not np.issubdtype(data.dtype, np.number) or len(data.shape) != 3:
            raise TypeError("Attempted creation with data that wasn't three-dimensional with only real numbers.")
        if not np.issubdtype(data.dtype, np.int_) or len(data.shape) != 3:
            raise TypeError("Attempted creation with labels that weren't three-dimensional with only real numbers.")
        if data.shape[0] != labels.shape[0]:
            raise ValueError("Attempted DataLoader creation with non-matching number of keys.")

        # check hyperparameters
        if not isinstance(batch_size, int) or batch_size < 0:
            raise TypeError("Attempted DataLoader creation with batch size that wasn't a positive integer.")

        # internals
        self._data = [Tensor(pt) for pt in data]
        self._labels = [Tensor(pt) for pt in labels]
        self._batch_size = batch_size
        self._shuffle = bool(shuffle)
        self._indices = np.arange(data.shape[0])

    @property
    def batch_size(self):
        # batch size
        return self._batch_size

    @property
    def data(self):
        return self._data

    def __len__(self) -> int:
        # length
        return (len(self._data) + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Generator[Tuple[Tensor, Tensor], None, None]:
        if self._shuffle:
            # shuffle dataloader
            np.random.shuffle(self._indices)

        for start_idx in range(0, len(self._indices), self._batch_size):
            # iteration
            end_idx = start_idx + self._batch_size
            batch_indices = self._indices[start_idx:end_idx]
            yield self._data[batch_indices], self._labels[batch_indices]
