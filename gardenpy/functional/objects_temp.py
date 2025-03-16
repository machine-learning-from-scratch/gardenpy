import numpy as np
from typing import List, Union, Dict, Optional, Tuple
from warnings import warn


class Matrix:
    _cache: List['Matrix', None] = []
    _ikwiad: bool = False

    def __init__(self, obj, *, _type: str, _dims: int):
        obj = np.array(obj)
        if (np.issubdtype(obj.dtype, np.number)) or len(obj.shape) != _dims:
            raise TypeError

        # tracking internals
        self._type: Union[str, None] = str(_type)
        self._id: Union[int, None] = None
        self._array: Union[np.ndarray, None] = obj
        # autodiff internals
        self._tracker: Union[Dict[str, Union[Matrix, list, None]], None] = {'drv': [], 'rlt': [], 'org': []}
        # other internals
        self._dims: int = _dims
        self._tags: List[str] = []

        # cache instance
        Matrix._add_cache(itm=self)

    @classmethod
    def _add_cache(cls, itm: 'Matrix') -> None:
        # local cache
        try:
            open_id = cls._cache.index(None)
            itm._global_id = open_id
            cls._cache[open_id] = itm
        except ValueError:
            open_id = len(cls._cache)
            cls._cache.append(itm)
            itm._id = open_id

    @staticmethod
    def _valid_array(itm: 'Matrix') -> bool:
        if itm._type is not None:
            # valid array
            return True
        else:
            # invalid array
            if not Matrix._ikwiad:
                warn("Attempted deleted Tensor reference.", UserWarning)
            return False

    @classmethod
    def _reference_array(cls, itm: Union['Matrix', str, int]) -> Tuple['Matrix', int]:
        # turn on ikwiad
        user_ikwiad = Matrix._ikwiad
        Matrix._ikwiad = True
        # attempt hex conversion
        try:
            itm = int(itm[1:], 16)
        except (ValueError, TypeError):
            pass

        if isinstance(itm, Matrix):
            # use object reference
            itm_id = itm._id
        elif isinstance(itm, int):
            # check index reference
            if len(cls._cache) <= itm:
                raise ValueError(
                    "Attempted reference outside Tensor instance list. "
                    f"Currently, instance list only contains {len(cls._cache)} items. "
                    f"A reference has been made to the {itm} index."
                )
            # use index reference
            itm_id = itm
            itm = cls._cache[itm_id]
        else:
            # invalid reference
            raise TypeError("Attempted Tensor reference with an invalid type.")
        if not cls._valid_array(itm=itm):
            # invalid tensor
            raise TypeError("Attempted reference to a deleted Tensor.")

        # return to user ikwiad
        Matrix._ikwiad = user_ikwiad
        # return items
        return itm, itm_id