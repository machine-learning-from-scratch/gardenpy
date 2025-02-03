r"""
**Data utilities.**

Contains:
    - class`DataLoader`
"""

from typing import Generator, Tuple
import numpy as np


class DataLoader:
    def __init__(self, data: np.ndarray, labels: np.ndarray, *, batch_size: int, shuffle: bool = True):
        # check for errors
        assert isinstance(data, np.ndarray), "'data' must be an array"
        assert isinstance(labels, np.ndarray), "'labels' must be an array"
        assert data.shape[0] == labels.shape[0], "'data' and 'labels' must have the same first dimension"
        assert isinstance(batch_size, int) and batch_size > 0, "'batch_size' must be a positive integer"

        # internals
        self._data = data
        self._labels = labels
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
        return (self._data.shape[0] + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if self._shuffle:
            # shuffle dataloader
            np.random.shuffle(self._indices)

        for start_idx in range(0, len(self._indices), self._batch_size):
            # iteration
            end_idx = start_idx + self._batch_size
            batch_indices = self._indices[start_idx:end_idx]
            yield self._data[batch_indices], self._labels[batch_indices]
