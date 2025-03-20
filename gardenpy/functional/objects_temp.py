r"""
**GardenPy objects.**

Core objects for the garden library.

Contains:
    - class:`_Array`
    - class:`Matrix`
    - class:`Gradient`
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self, TypeVar
from warnings import warn
import numpy as np
from numpy.typing import NDArray

from .raw_operators import inf_remove
from ..utils.errors import TrackingError

# array type T
T = TypeVar('T', bound='_Array')


class _Array(ABC):
    # ikwiad
    _ikwiad: bool = False
    # base memory internals
    _cache: list[T | None] = []
    _prefix: str = 'a'

    def __init__(self, obj: any, *, _ndim: int):
        # verify object
        assert isinstance(_ndim, int) and 0 < _ndim, "Internal ndim must be a positive integer."
        obj = np.array(obj)
        if not np.issubdtype(obj.dtype, np.number) or obj.ndim != _ndim:
            # NB: Current implementation forces Matrix to be 2D and Gradient to be 4D.
            # This can be upscaled so Matrix is nD and Gradient is 2nD, as the Gradient is a 2nD Jacobian of the Matrix.
            # However, this falls outside the scope of this project.
            raise TypeError(
                "Failed instantiation: Input object failed to be all integers or "
                f"failed to match the given ndim of {_ndim} (obj ndim of {obj.ndim})."
            )

        # set tracking internals
        self._id: int | None = None
        self._array: NDArray | None = obj
        # set autodiff internals
        self._default_tracker: dict[str, T | list | None] | None = None
        self._tracker: dict[str, T | list | None] | None = None
        # other internals
        self._tags: list[str] = []

        # cache instance
        self._add_cache(itm=self)

    def __repr__(self) -> str:
        self._is_valid_array(itm=self)
        return str(self._array)

    @property
    def id(self) -> str | None:
        if not self._is_valid_array(itm=self):
            return None
        return f"{self.__class__._prefix}{hex(self._id)}"

    @property
    def array(self) -> NDArray | None:
        self._is_valid_array(itm=self)
        return self._array

    @property
    def tags(self) -> list[str]:
        self._is_valid_array(itm=self)
        return self._tags

    @tags.setter
    def tags(self, tag: str) -> None:
        self._is_valid_array(itm=self)
        self._tags.append(str(tag))

    def remove_tag(self, tag: str) -> None:
        if tag in self._tags:
            self._tags.remove(str(tag))
        elif not _Array._ikwiad:
            warn(f"Referenced tag ({tag}) wasn't found in the in the instances tags ({self._tags}).", UserWarning)

    @property
    def shape(self) -> tuple[int, ...] | None:
        if not self._is_valid_array(itm=self):
            return None
        return self._array.shape

    @property
    def tracker(self) -> dict[str, T | list | str | None] | None:
        if not self._is_valid_array(itm=self):
            # invalid array
            return None
        # ikwiad on
        user_ikwiad = _Array._ikwiad
        _Array._ikwiad = True

        # instance track
        alt_tracker = self._track_instance()

        # ikwiad reset
        _Array._ikwiad = user_ikwiad
        # tracking internals
        return {'id': self.id, 'tags': self._tags, **alt_tracker}

    @property
    def internals(self) -> dict[str, T | list | str | None] | None:
        if not self._is_valid_array(itm=self):
            # invalid array
            return None
        # ikwiad on
        user_ikwiad = _Array._ikwiad
        _Array._ikwiad = True

        # instance track
        alt_tracker = self._track_instance()

        # ikwiad reset
        _Array._ikwiad = user_ikwiad
        # full internals
        return {'id': self.id, 'tags': self._tags, 'shape': self.shape, 'prefix': self.__class__._prefix, **alt_tracker}

    @classmethod
    def _add_cache(cls, itm: T) -> None:
        try:
            # unused cache location
            open_id = cls._cache.index(None)
            itm._global_id = open_id
            cls._cache[open_id] = itm
        except ValueError:
            # new cache location
            open_id = len(cls._cache)
            cls._cache.append(itm)
            itm._id = open_id

    @classmethod
    def _reference_array(cls, itm: Self | str | int) -> tuple[T, int]:
        # ikwiad on
        user_ikwiad = _Array._ikwiad
        _Array._ikwiad = True
        _prefix = None
        try:
            # prefix and id reference
            _prefix, itm = itm[0], int(itm[1:], 16)
        except (ValueError, TypeError):
            # failed attempt
            pass

        if isinstance(itm, _Array):
            if cls.__name__ != itm.__class__.__name__:
                # differing classes
                raise TypeError(
                    "Invalid type reference: Reference made outside of this subclass. "
                    f"Reference made to subclass {itm.__class__.__name__} inside of subclass {cls.__name__}. "
                    f"Use the correct subclass to reference {itm.__class__.__name__}."
                )
            # array id
            itm_id = itm._id
        elif isinstance(itm, int):
            # check index reference
            if _prefix is not None and _prefix != cls._prefix:
                # differing classes
                raise TypeError(
                    "Invalid type reference: Reference made outside of this subclass. "
                    f"Reference made to '{_prefix}' in '{cls._prefix}', referencing {cls.__name__}. "
                    f"Use the correct subclass to reference your _Array ('{_prefix}') type."
                )
            if len(cls._cache) <= itm:
                # out of index
                raise ValueError(
                    "Invalid index reference: Reference made outside of the length of this subclass's cache."
                    f"Reference made to index {itm}, more than the {len(cls._cache)} items available."
                )
            # use index reference
            itm_id = itm
            itm = cls._cache[itm_id]
        else:
            # invalid reference
            raise TypeError(
                "Invalid reference type: Reference should be made with an "
                "_Array instance, reference id, or cache index."
            )
        if itm is None or itm_id is None:
            # invalid array
            raise TypeError(
                "Invalid final reference: Referenced location is an empty spot within the subclass's cache."
            )

        # ikwiad reset
        _Array._ikwiad = user_ikwiad
        return itm, itm_id

    @classmethod
    def reset(cls, *args: T | str | None) -> None:
        # saved arrays
        args = list(args)
        arg_ids = []
        for arg in args:
            _, arg_id = cls._reference_array(itm=arg)
        # removed arrays
        removed_arrays = [
            itm for i, itm in enumerate(cls._cache)
            if i not in arg_ids and itm is not None and 'retain' not in itm._tags
        ]
        for instance in removed_arrays:
            # reset removed arrays
            instance.instance_reset()
        return None

    @classmethod
    def reference(cls, idx: str | int) -> T:
        # reference array
        array, _ = cls._reference_array(itm=idx)
        return array

    @classmethod
    def cache(cls) -> list[str | None]:
        # subclass caches
        return [itm.id if itm is not None else None for itm in cls._cache]

    @classmethod
    def cache_debug(cls) -> list[dict | None]:
        # subclass internal caches
        return [itm.internals if itm is not None else None for itm in cls._cache]

    @classmethod
    def ikwiad(cls, ikwiad: bool | None = None) -> None:
        if ikwiad is None:
            # switch ikwiad
            _Array._ikwiad = not _Array._ikwiad
            return None
        # set ikwiad
        _Array._ikwiad = bool(ikwiad)
        return None

    def instance_reset(self) -> None:
        if self._is_valid_array(itm=self):
            # clear cache location
            self.__class__._cache[self._id] = None
            # reset internals
            self._id = None
            self._array = None
            self._tracker = None
            self._tags.append('deleted')
        return None

    @staticmethod
    def _is_valid_array(itm: T) -> bool:
        if itm._id is not None:
            # valid array
            return True
        else:
            # invalid array
            if not _Array._ikwiad:
                warn("A reference was made to an invalid Tensor.", UserWarning)
            return False

    def _unpack_ids(self, itm: T | list | property | None) -> list | str | None:
        if isinstance(itm, _Array):
            # id reference
            return itm.id
        elif isinstance(itm, list):
            # function call
            return [self._unpack_ids(it) for it in itm]
        else:
            if itm is not None:
                # hex id call
                return hex(id(itm))
            else:
                # none spot
                return None

    @abstractmethod
    def _track_instance(self) -> dict:
        raise NotImplementedError("_track_instance must be implemented in subclasses for external tracker references.")


########################################################################################################################


class Matrix(_Array):
    # matrix memory internals
    _cache: list[Matrix | None] = []
    _prefix: str = 'm'

    def __init__(self, obj: any):
        super().__init__(obj=obj, _ndim=2)
        # matrix subclass internals
        self._default_tracker = {'derivative': [], 'relation': [], 'origin': []}
        self._tracker = self._default_tracker.copy()

    def copy(self) -> Matrix | None:
        if self._is_valid_array(itm=self):
            # duplicate array
            return Matrix(obj=self._array)
        return None

    def instance_track_reset(self) -> None:
        if self._is_valid_array(itm=self):
            # reset tracker
            self._tracker = self._default_tracker.copy()
        return None

    @classmethod
    def replace(cls, replaced: Matrix | str, replacer: Matrix | str) -> None:
        # find replaced and replacer information
        replaced_itm, replaced_id = cls._reference_array(itm=replaced)
        replacer_itm, replacer_id = cls._reference_array(itm=replacer)
        # move replacer and delete replaced
        replaced.instance_reset()
        replacer._id = replaced._id
        cls._cache[replaced_id] = replaced_itm
        cls._cache[replacer_id] = None
        return None

    @classmethod
    def track_reset(cls, *args: Matrix | str | None) -> None:
        # saved arrays
        args = list(args)
        for i, arg in enumerate(args):
            _, arg_id = cls._reference_array(arg)
            args[i] = arg_id

        # track removed arrays
        untracked_arrays = [
            itm for i, itm in enumerate(cls._cache)
            if i not in args and itm is not None and 'track retain' not in itm._tags
        ]
        for array in untracked_arrays:
            # track reset track removed arrays
            array.instance_track_reset()
        return None

    def _track_instance(self):
        # repr to hex
        alt_tracker = {
            'derivative': self._tracker.copy()['derivative'],
            'relation': self._unpack_ids(self._tracker.copy()['relation']),
            'origin': self._unpack_ids(self._tracker.copy()['origin'])
        }
        return alt_tracker

    @staticmethod
    def _update_track(obj: Matrix, derivative: any, relation: any) -> None:
        # update tracker
        obj._tracker['derivative'].append(derivative)
        obj._tracker['relation'].append(relation)
        return None

    class _BaseMethod(ABC):
        @staticmethod
        def _ndim(obj: NDArray, ndim: int) -> None:
            assert isinstance(ndim, int) and 0 < ndim, "Internal ndim must be a positive integer."
            if not (isinstance(obj, np.ndarray) and obj.ndim == ndim):
                raise ValueError(
                    "Failed initialization: Passed object was either not a numpy array or didn't have ndim dimensions. "
                    f"Passed object was type {type(obj)} with dimensions "
                    f"{'NULL' if not isinstance(obj, np.ndarray) else obj.ndim}. "
                    f"Expected a numpy array with ndim {ndim}."
                )
            return None

        @staticmethod
        def _dim_match(obj_1: NDArray, obj_2: NDArray) -> None:
            if not (isinstance(obj_1, np.ndarray) and isinstance(obj_2, np.ndarray) and obj_1.shape == obj_2.shape):
                raise ValueError(
                    "Failed matching: Passed objects were either not both numpy arrays or didn't match shapes. "
                    f"Objects were {type(obj_1)} and {type(obj_2)} respectively. "
                    f"Object 1 had dimensions {'NULL' if not isinstance(obj_1, np.ndarray) else obj_1.ndim}. "
                    f"Object 2 had dimensions {'NULL' if not isinstance(obj_2, np.ndarray) else obj_2.ndim}. "
                )
            return None

        @staticmethod
        def _elementwise_broadcast(two_grad: NDArray) -> NDArray:
            assert isinstance(two_grad, np.ndarray) and two_grad.ndim == 2, "two_grad must be a 2D numpy array."
            # 4D identity creation
            eye = np.zeros((*two_grad.shape, *two_grad.shape))
            np.einsum('ijij -> ij', eye, optimize=False)[:] = 1.0
            # 2D to 4D broadcasting
            return eye * two_grad[np.newaxis, np.newaxis, :, :]

        @staticmethod
        def _scalar_broadcast(two_grad: NDArray) -> NDArray:
            assert isinstance(two_grad, np.ndarray) and two_grad.ndim == 2, "two_grad must be a 2D numpy array."
            # extend to 4D
            return two_grad[np.newaxis, np.newaxis, :, :]

        @staticmethod
        @abstractmethod
        def forward(*args: any, **kwargs: any) -> NDArray:
            pass

        @staticmethod
        @abstractmethod
        def backward(*args: any, **kwargs: any) -> NDArray:
            pass

        @staticmethod
        @abstractmethod
        def backward_o(*args: any, **kwargs: any) -> NDArray:
            pass

        @classmethod
        @abstractmethod
        def _forward(cls, *args: any, **kwargs: any) -> NDArray:
            pass

        @classmethod
        @abstractmethod
        def _backward(cls, *args: any, **kwargs: any) -> NDArray:
            pass

        @classmethod
        @abstractmethod
        def _backward_o(cls, *args: any, **kwargs: any) -> NDArray:
            pass

        @abstractmethod
        def main(self, *args: any, **kwargs: any) -> 'Matrix':
            pass

    class _LoneBaseMethod(_BaseMethod):
        @staticmethod
        @abstractmethod
        def forward(*args: any, **kwargs: any) -> NDArray:
            pass

        @staticmethod
        @abstractmethod
        def backward(*args: any, **kwargs: any) -> NDArray:
            pass

        @staticmethod
        def backward_o(*args: any, **kwargs: any) -> NDArray:
            raise NotImplementedError(
                "Invalid call: backward_o is never defined for lone methods."
            )

        @classmethod
        @abstractmethod
        def _forward(cls, *args: any, **kwargs: any) -> NDArray:
            pass

        @classmethod
        @abstractmethod
        def _backward(cls, *args: any, **kwargs: any) -> NDArray:
            pass

        @classmethod
        def _backward_o(cls, *args: any, **kwargs: any) -> NDArray:
            raise NotImplementedError(
                "Invalid call: _backward_o is never defined for lone methods."
            )

        @abstractmethod
        def main(self, *args: any, **kwargs: any) -> 'Matrix':
            pass

    class ElementWiseMethod(_BaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError()

        @staticmethod
        @abstractmethod
        def backward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError()

        @staticmethod
        @abstractmethod
        def backward_o(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError()

        @classmethod
        def _forward(cls, main: NDArray, other: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            cls._ndim(obj=other, ndim=2)
            cls._dim_match(obj_1=main, obj_2=other)
            result = cls.forward(main, other)
            cls._ndim(obj=result, ndim=2)
            return result

        @classmethod
        def _backward(cls, main: NDArray, other: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            cls._ndim(obj=other, ndim=2)
            result = cls.backward(main, other)
            return cls._elementwise_broadcast(two_grad=result)

        @classmethod
        def _backward_o(cls, main: NDArray, other: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            cls._ndim(obj=other, ndim=2)
            result = cls.backward_o(other, main)  # note: this is weird, but might work
            return cls._elementwise_broadcast(two_grad=result)

        def main(self, main: Matrix, other: Matrix | NDArray | float | int) -> Matrix:
            # check main array
            if not isinstance(main, Matrix):
                raise TypeError(
                    "..."
                )

            # set array
            if isinstance(other, Matrix):
                arr = other._array
            elif isinstance(other, np.ndarray | float | int):
                arr = other
            else:
                raise TypeError()

            # calculate result
            result = Matrix(self.forward(main._array, arr))
            result._tracker['origin'] = [main, other]
            # track main
            Matrix._update_track(obj=main, derivative=self._backward, relation=[other, result])
            if isinstance(other, Matrix):
                # track other
                Matrix._update_track(obj=other, derivative=self._backward_o, relation=[main, result])
            return result

    class LoneElementWiseMethod(_LoneBaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray) -> NDArray:
            raise NotImplementedError()

        @staticmethod
        @abstractmethod
        def backward(main: NDArray) -> NDArray:
            raise NotImplementedError()

        @classmethod
        def _forward(cls, main: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            result = cls.forward(main)
            cls._ndim(obj=result, ndim=2)
            return result

        @classmethod
        def _backward(cls, main: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            result = cls.backward(main)
            return cls._elementwise_broadcast(two_grad=result)

        def main(self, main: Matrix) -> Matrix:
            # check array
            if not isinstance(main, Matrix):
                raise TypeError(
                    "Non-matrix call"
                )
            # calculate result
            result = Matrix(self.forward(main._array))
            result._tracker['origin'] = [main, None]
            # track main
            Matrix._update_track(obj=main, derivative=self._backward, relation=[None, result])
            # return result
            return result

    class ScalarMethod(_LoneBaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray) -> NDArray:
            raise NotImplementedError()

        @staticmethod
        @abstractmethod
        def backward(main: NDArray) -> NDArray:
            raise NotImplementedError()

        @classmethod
        def _forward(cls, main: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            result = cls.forward(main)
            cls._ndim(obj=result, ndim=2)
            return result

        @classmethod
        def _backward(cls, main: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            result = cls.backward(main)
            return cls._scalar_broadcast(two_grad=result)

        def main(self, main: Matrix) -> Matrix:
            # check array
            if not isinstance(main, Matrix):
                raise TypeError(
                    "Non-matrix call"
                )
            # calculate result
            result = Matrix(self.forward(main._array))
            result._tracker['origin'] = [main, None]
            # track main
            Matrix._update_track(obj=main, derivative=self._backward, relation=[None, result])
            # return result
            return result

    class CustomMethod(_BaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError()

        @staticmethod
        @abstractmethod
        def backward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError()

        @staticmethod
        @abstractmethod
        def backward_o(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError()

        @classmethod
        def _forward(cls, main: NDArray, other: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            cls._ndim(obj=other, ndim=2)
            result = cls.forward(main, other)
            cls._ndim(obj=result, ndim=2)
            return result

        @classmethod
        def _backward(cls, main: NDArray, other: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            cls._ndim(obj=other, ndim=2)
            result = cls.backward(main, other)
            cls._ndim(obj=result, ndim=4)
            return result

        @classmethod
        def _backward_o(cls, main: NDArray, other: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            cls._ndim(obj=other, ndim=2)
            result = cls.backward_o(other, main)  # note: this is also super weird
            cls._ndim(obj=result, ndim=4)
            return result

        def main(self, main: Matrix, other: Matrix | NDArray | float | int) -> Matrix:
            # check main array
            if not isinstance(main, Matrix):
                raise TypeError(
                    "..."
                )

            # set array
            if isinstance(other, Matrix):
                arr = other._array
            elif isinstance(other, np.ndarray | float | int):
                arr = other
            else:
                raise TypeError()

            # calculate result
            result = Matrix(self.forward(main._array, arr))
            result._tracker['origin'] = [main, other]
            # track main
            Matrix._update_track(obj=main, derivative=self._backward, relation=[other, result])
            if isinstance(other, Matrix):
                # track other
                Matrix._update_track(obj=other, derivative=self._backward_o, relation=[main, result])
            return result

    class LoneCustomMethod(_LoneBaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray) -> NDArray:
            raise NotImplementedError()

        @staticmethod
        @abstractmethod
        def backward(main: NDArray) -> NDArray:
            raise NotImplementedError()

        @classmethod
        def _forward(cls, main: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            result = cls.forward(main)
            cls._ndim(obj=result, ndim=2)
            return result

        @classmethod
        def _backward(cls, main: NDArray) -> NDArray:
            cls._ndim(obj=main, ndim=2)
            result = cls.backward(main)
            cls._ndim(obj=result, ndim=4)
            return result

        def main(self, main: Matrix) -> Matrix:
            # check array
            if not isinstance(main, Matrix):
                raise TypeError(
                    "Non-matrix call"
                )
            # calculate result
            result = Matrix(self.forward(main._array))
            result._tracker['origin'] = [main, None]
            # track main
            Matrix._update_track(obj=main, derivative=self._backward, relation=[None, result])
            # return result
            return result

    class _MatMul(CustomMethod):
        @staticmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            return main @ other

        @staticmethod
        def backward(main: NDArray, other: NDArray) -> NDArray:
            four_concat = other.T[np.newaxis, :, np.newaxis, :]
            eye = np.eye(main.shape[0])[:, np.newaxis, :, np.newaxis]
            return four_concat * eye

        @staticmethod
        def backward_o(main: NDArray, other: NDArray) -> NDArray:
            four_concat = main[:, np.newaxis, :, np.newaxis]
            eye = np.eye(other.shape[1])[np.newaxis, :, np.newaxis, :]
            return four_concat * eye

    class _Pow(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            return main ** other

        @staticmethod
        @inf_remove(inf_val=1e10)
        def backward(main: NDArray, other: NDArray) -> NDArray:
            two_grad = other * (main ** (other - 1.0))
            return two_grad

        @staticmethod
        @inf_remove(inf_val=1e10)
        def backward_o(main: NDArray, other: NDArray) -> NDArray:
            two_grad = np.log(main) * (main ** other)
            return two_grad

    class _Mul(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            return main * other

        @staticmethod
        def backward(main: NDArray, other: NDArray) -> NDArray:
            return other

        @staticmethod
        def backward_o(main: NDArray, other: NDArray) -> NDArray:
            return main

    class _TrueDiv(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            return main / other

        @staticmethod
        @inf_remove(inf_val=1e10)
        def backward(main: NDArray, other: NDArray) -> NDArray:
            return other ** -1.0

        @staticmethod
        @inf_remove(inf_val=1e10)
        def backward_o(main: NDArray, other: NDArray) -> NDArray:
            return -main / other ** 2.0

    class _Add(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            return main + other

        @staticmethod
        def backward(main: NDArray, other: NDArray) -> NDArray:
            return np.ones(main.shape)

        @staticmethod
        def backward_o(main: NDArray, other: NDArray) -> NDArray:
            return np.ones(other.shape)

    class _Sub(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            return main - other

        @staticmethod
        def backward(main: NDArray, other: NDArray) -> NDArray:
            return np.ones(main.shape)

        @staticmethod
        def backward_o(main: NDArray, other: NDArray) -> NDArray:
            return -np.ones(other.shape)

    # internal instance
    _matmul = _MatMul()
    _pow = _Pow()
    _mul = _Mul()
    _truediv = _TrueDiv()
    _add = _Add()
    _sub = _Sub()

    # dunder
    def __matmul__(self, other: Matrix | NDArray) -> Matrix:
        r"""**Matrix multiplication.**"""
        return self._matmul.main(self, other)

    def __pow__(self, other: Matrix | NDArray | float | int) -> Matrix:
        r"""**Hadamard power.**"""
        return self._pow.main(self, other)

    def __mul__(self, other: Matrix | NDArray | float | int) -> Matrix:
        r"""**Hadamard multiplication.**"""
        return self._mul.main(self, other)

    def __truediv__(self, other: Matrix | NDArray | float | int) -> Matrix:
        r"""**Hadamard division.**"""
        return self._truediv.main(self, other)

    def __add__(self, other: Matrix | NDArray | float | int) -> Matrix:
        r"""**Addition.**"""
        return self._add.main(self, other)

    def __sub__(self, other: Matrix | NDArray | float | int) -> Matrix:
        r"""**Subtraction.**"""
        return self._sub.main(self, other)


########################################################################################################################


class Gradient(_Array):
    # gradient memory internals
    _cache: list[Gradient | None] = []
    _prefix: str = 'g'

    def __init__(self, obj: any, *, _override: bool = False):
        assert _override
        super().__init__(obj=obj, _ndim=4)
        self._default_tracker = {'chain': []}
        self._tracker = self._default_tracker.copy()

    def _track_instance(self):
        alt_tracker = {'chain': self._unpack_ids(self._tracker.copy()['chain'])}
        return alt_tracker

    def reduce_grad(self) -> Matrix:
        return Matrix(np.sum(self._array, axis=(0, 1)))

    @staticmethod
    def _chain_opr(down: NDArray, up: NDArray) -> NDArray:
        # 6D downstream expansion
        down = down[:, :, :, :, np.newaxis, np.newaxis]
        # 6D upstream expansion
        up = up[np.newaxis, np.newaxis, :, :, :, :]
        # 6D to 4D manipulation
        return np.sum(down * up, axis=(2, 3))

    @staticmethod
    def nabla(grad: Matrix, wrt: Matrix, *, binary: bool = True) -> Gradient:
        # check tensors
        if not isinstance(grad, Matrix):
            raise TypeError
        if not isinstance(wrt, Matrix):
            raise TypeError

        # set gradient relation
        relation = None
        if not binary:
            relation = []

        def _relate(item, target, trace=None):
            nonlocal relation
            if trace is None:
                # reset trace
                trace = []
            # NB: This only gets origins if the item is a matrix.
            # Tracing gradients through gradients is possible, but requires a lot of modification and significantly
            # increases computational time, even if it's never used.
            if binary and relation is None and isinstance(item, Matrix):
                # get origins
                origins = [Matrix.reference(org) for org in item.tracker['origin']]
                trace.append(item)
                if target in origins:
                    # related
                    trace.append(target)
                    relation = trace.copy()
                else:
                    # relation search
                    [_relate(item=origin, target=target, trace=trace) for origin in origins]
            elif not binary and isinstance(item, Matrix):
                # get origins
                origins = [Matrix.reference(org) for org in item.tracker['origin']]
                trace.append(item)
                if target in origins:
                    # related
                    trace.append(target)
                    relation.append(trace.copy())
                else:
                    # relation search
                    [_relate(item=origin, target=target, trace=trace) for origin in origins]

        # relate tensors
        _relate(wrt, grad)
        if not relation:
            # no relation
            raise TrackingError(grad=grad, wrt=wrt, message=(
                f"No relation could be found between {grad.id} and {wrt.id}.\n"
                "This might be due to:\n"
                "   No clear relation between the Tensors.\n"
                "   Accidental clearing of trackers.\n"
                "   Deletion of Tensors.\n"
                "   Accidental reference to the wrong Tensor."
            ))

        def _derive(down: Matrix, up: Matrix) -> Gradient:
            # get relations
            strm_result = [Matrix.reference(rlt[1]) for rlt in up.tracker['relation']]
            strm_other = [Matrix.reference(rlt[0]) for rlt in up.tracker['relation']]
            # get operation
            drv_operator = up.tracker['derivative'][strm_result.index(down)]
            other = strm_other[strm_result.index(down)]

            if isinstance(other, Matrix):
                # get value
                other = other._array
            # calculate local gradient
            try:
                # pair derivative method
                res = drv_operator(up._array, other)
            except TypeError:
                # lone derivative method
                res = drv_operator(up._array)

            # tensor conversion
            res = Gradient(obj=res, _override=True)

            # local gradient setup
            res._tracker['chain'] += [down, up]
            return res

        # linear connection override
        linear_override = False
        if not binary and len(relation) != 1:
            linear_override = True
        if binary:
            # calculate initial grads
            result = _derive(down=relation[-2], up=relation[-1])
            del relation[-1]
            while 1 < len(relation):
                # chain rule grads
                result = Gradient.chain(down=_derive(down=relation[-2], up=relation[-1]), up=result)
                del relation[-1]
        else:
            # accumulate grads
            grads = []
            grad_itms = []
            track = None
            for itm in relation:
                op_res = _derive(down=itm[-2], up=itm[-1])
                del itm[-1]
                while 1 < len(itm):
                    # chain rule gradients
                    op_res = Gradient.chain(down=_derive(down=itm[-2], up=itm[-1]), up=op_res)
                    del itm[-1]
                grads.append(op_res._array)
                grad_itms.append(op_res)
                track = op_res._tracker
            result = 0
            for grad, itm in zip(grads, grad_itms):
                result += grad
                itm.instance_reset()
            result = Gradient(obj=result, _override=True)
            result._tracker = track

        # return final gradient
        if linear_override:
            result._tags.append('linear override')
        return result

    @staticmethod
    def chain(down: Gradient, up: Gradient) -> Gradient:
        if not isinstance(down, Gradient):
            raise TypeError(
                "Attempted chain-rule calculation with down object that was either "
                "not a Tensor or not a gradient subtype."
            )
        if not isinstance(up, Gradient):
            raise TypeError(
                "Attempted chain-rule calculation with up object that was either "
                "not a Tensor or not a gradient subtype."
            )

        # check relation
        down_relation = down._tracker['chain'][-1]
        up_relation = up._tracker['chain'][0]
        if down_relation != up_relation:
            raise TrackingError(grad=down, wrt=up, message=(
                f"No relation could be found between {down.id} and {up.id}.\n"
                "This might be due to:\n"
                "   No clear relation between the Tensors.\n"
                "   Accidental clearing of trackers.\n"
                "   Deletion of Tensors.\n"
                "   Accidental reference to the wrong Tensor."
            ))

        # chain gradients
        result = Gradient(obj=Gradient._chain_opr(down=down._array, up=up._array), _override=True)
        # set gradient internals
        result._tracker['chain'] = down._tracker['chain'] + up._tracker['chain'][1:]
        # return final gradient
        return result
