r"""
**GardenPy objects.**

Core objects for the garden library.

Contains:
    - class:`_Tensor`
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

# tensor type T
T = TypeVar('T', bound='_Tensor')


class _Tensor(ABC):
    r"""
    **GardenPy's base Tensor.**

    Includes all base operations and subclass structure, including :class:`Matrix` and :class:`Gradient`.
    _Tensor is an abstract base class and should never be instantiated; only subclasses should be instantiated.
    Creation of subclasses will create a reference within the object's cache.

    Note:
        Tensor subclasses will cache instances of themselves.
        These caches don't automatically clear and will cause memory leaks.
        They should be cleared using :func:`_Tensor.reset`.
    """
    # ikwiad
    _ikwiad: bool = False
    # base memory internals
    _cache: list[T | None] = []
    _prefix: str = '_'

    def __init__(self, obj: any, *, _ndim: int):
        r"""
        **Tensor creation.**

        Creates a Tensor subclass with default internals and caches instance.
        Should only be run with subclasses.

        Args:
            obj (any): Object to be turned into a Tensor.
            _ndim (int), 0 < _ndim: Allowed dimensions of Tensor object.

        Raises:
            TypeError: Object wasn't an _ndim-dimensional array consisting of only real numbers.
            AssertionError: Invalid _ndim argument.

        Note:
            All objects will undergo NumPy array conversion.
        """
        # verify object
        assert isinstance(_ndim, int) and 0 < _ndim, "_ndim must be a positive integer."
        obj = np.array(obj)
        if not np.issubdtype(obj.dtype, np.number) or obj.ndim != _ndim:
            # NB: Current implementation forces Matrix to be 2D and Gradient to be 4D.
            # This can be upscaled so Matrix is nD and Gradient is 2nD, as the Gradient is a 2nD Jacobian of the Matrix.
            # However, this falls outside the scope of this project.
            raise TypeError(
                f"Failed instantiation: Input object failed to be all numbers or "
                f"failed to match the given ndim of {_ndim} (obj ndim of {obj.ndim}). "
                f"This could have been caused by inserting two non-array type items into a function that allows one of "
                f"the items to be a non-array type."
            )

        # set tracking internals
        self._id: int | None = None
        self._tensor: NDArray | None = obj
        # set autodiff internals
        self._default_tracker: dict[str, T | list | None] | None = None
        self._tracker: dict[str, T | list | None] | None = None
        # other internals
        self._tags: list[str] = []

        # cache instance
        self._add_cache(itm=self)

    def __repr__(self) -> str:
        self._is_valid_tensor(itm=self)
        return str(self._tensor)

    @property
    def id(self) -> str | None:
        r"""
        **Tensor ID.**

        ID correlating to its position within the class's cache.

        Returns:
            str | None: Current Tensor ID.
                Returns None if the function is used on a deleted Tensor.

        Raises:
            UserWarning: The function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`_Tensor.ikwiad`.
        """
        if not self._is_valid_tensor(itm=self):
            return None
        return f"{self.__class__._prefix}{hex(self._id)}"

    @property
    def tensor(self) -> NDArray | None:
        r"""
        **Tensor's internal NumPy array.**

        Returns:
            np.ndarray | None: Tensor's internal NumPy array.
                Returns None if the function is used on a deleted Tensor.

        Raises:
            UserWarning: The function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`_Tensor.ikwiad`.
        """
        self._is_valid_tensor(itm=self)
        return self._tensor

    @property
    def shape(self) -> tuple[int, ...] | None:
        if not self._is_valid_tensor(itm=self):
            return None
        return self._tensor.shape

    @property
    def tags(self) -> list[str]:
        self._is_valid_tensor(itm=self)
        return self._tags

    @property
    def tracker(self) -> dict[str, T | list | str | None] | None:
        if not self._is_valid_tensor(itm=self):
            # invalid tensor
            return None
        # ikwiad on
        user_ikwiad = _Tensor._ikwiad
        _Tensor._ikwiad = True

        # instance track
        alt_tracker = self._track_instance()

        # ikwiad reset
        _Tensor._ikwiad = user_ikwiad
        # tracking internals
        return {'id': self.id, 'tags': self._tags, **alt_tracker}

    @property
    def internals(self) -> dict[str, T | list | str | None] | None:
        if not self._is_valid_tensor(itm=self):
            # invalid tensor
            return None
        # ikwiad on
        user_ikwiad = _Tensor._ikwiad
        _Tensor._ikwiad = True

        # instance track
        alt_tracker = self._track_instance()

        # ikwiad reset
        _Tensor._ikwiad = user_ikwiad
        # full internals
        return {'id': self.id, 'tags': self._tags, 'shape': self.shape, **alt_tracker}

    def add_tag(self, tag: str) -> None:
        self._is_valid_tensor(itm=self)
        self._tags.append(str(tag))

    def remove_tag(self, tag: str) -> None:
        if tag in self._tags:
            self._tags.remove(str(tag))
        elif not _Tensor._ikwiad:
            warn(f"Referenced tag ({tag}) wasn't found in the in the instances tags ({self._tags}).", UserWarning)

    def instance_reset(self) -> None:
        if self._is_valid_tensor(itm=self):
            # clear cache location
            self.__class__._cache[self._id] = None
            # reset internals
            self._id = None
            self._tensor = None
            self._tracker = None
            self._tags.append('deleted')
        return None

    @classmethod
    def cache(cls) -> list[str | None]:
        # subclass caches
        return [itm.id if itm is not None else None for itm in cls._cache]

    @classmethod
    def cache_debug(cls) -> list[dict | None]:
        # subclass internal caches
        return [itm.internals if itm is not None else None for itm in cls._cache]

    @classmethod
    def reference(cls, idx: str | int) -> T:
        # reference tensors
        tensor, _ = cls._reference_tensor(itm=idx)
        return tensor

    @classmethod
    def reset(cls, *args: T | str | None) -> None:
        # saved tensors
        args = list(args)
        arg_ids = []
        for arg in args:
            _, arg_id = cls._reference_tensor(itm=arg)
        # removed tensors
        removed_tensors = [
            itm for i, itm in enumerate(cls._cache)
            if i not in arg_ids and itm is not None and 'retain' not in itm._tags
        ]
        for instance in removed_tensors:
            # reset removed tensors
            instance.instance_reset()
        return None

    @classmethod
    def ikwiad(cls, ikwiad: bool | None = None) -> None:
        if ikwiad is None:
            # switch ikwiad
            _Tensor._ikwiad = not _Tensor._ikwiad
            return None
        # set ikwiad
        _Tensor._ikwiad = bool(ikwiad)
        return None

    @staticmethod
    def _is_valid_tensor(itm: T) -> bool:
        if itm._id is not None:
            # valid tensor
            return True
        else:
            # invalid tensor
            if not _Tensor._ikwiad:
                warn("Reference was made to a non-valid Tensor type.", UserWarning)
            return False

    @classmethod
    def _add_cache(cls, itm: T) -> None:
        try:
            # unused cache location
            open_id = cls._cache.index(None)
            itm._id = open_id
            cls._cache[open_id] = itm
        except ValueError:
            # new cache location
            open_id = len(cls._cache)
            cls._cache.append(itm)
            itm._id = open_id

    @classmethod
    def _reference_tensor(cls, itm: Self | str | int) -> tuple[T, int]:
        # ikwiad on
        user_ikwiad = _Tensor._ikwiad
        _Tensor._ikwiad = True
        _prefix = None
        try:
            # prefix and id reference
            _prefix, itm = itm[0], int(itm[1:], 16)
        except (ValueError, TypeError):
            # failed attempt
            pass

        if isinstance(itm, _Tensor):
            if cls.__name__ != itm.__class__.__name__:
                # differing classes
                raise TypeError(
                    f"Invalid type reference: Reference made outside of this subclass. "
                    f"Reference made to subclass {itm.__class__.__name__} inside of subclass {cls.__name__}. "
                    f"Use the correct subclass to reference {itm.__class__.__name__}."
                )
            # tensor id
            itm_id = itm._id
        elif isinstance(itm, int):
            # check index reference
            if _prefix is not None and _prefix != cls._prefix:
                # differing classes
                raise TypeError(
                    f"Invalid type reference: Reference made outside of this subclass. "
                    f"Reference made to '{_prefix}' in '{cls._prefix}', referencing {cls.__name__}. "
                    f"Use the correct subclass to reference your Tensor ('{_prefix}') type."
                )
            if len(cls._cache) <= itm:
                # out of index
                raise ValueError(
                    f"Invalid index reference: Reference made outside of the length of this subclass's cache."
                    f"Reference made to index {itm}, more than the {len(cls._cache)} items available."
                )
            # use index reference
            itm_id = itm
            itm = cls._cache[itm_id]
        else:
            # invalid reference
            raise TypeError(
                "Invalid reference type: Reference should be made with a "
                "Tensor instance, reference id, or cache index."
            )
        if itm is None or itm_id is None:
            # invalid tensor
            raise TypeError(
                "Invalid final reference: Referenced location is an empty spot within the subclass's cache "
                "or an invalid object."
            )

        # ikwiad reset
        _Tensor._ikwiad = user_ikwiad
        return itm, itm_id

    def _unpack_ids(self, itm: T | list | property | None) -> list | str | float| int | None:
        if isinstance(itm, _Tensor):
            # id reference
            return itm.id
        elif isinstance(itm, list):
            # function call
            return [self._unpack_ids(it) for it in itm]
        else:
            if itm is None or isinstance(itm, float | int):
                # raw item
                return itm
            else:
                # item hex
                return hex(id(itm))

    @abstractmethod
    def _track_instance(self) -> dict:
        raise NotImplementedError("_track_instance must be implemented in subclasses for external internal references.")


########################################################################################################################


class Matrix(_Tensor):
    # matrix memory internals
    _cache: list[Matrix | None] = []
    _prefix: str = 'm'

    def __init__(self, obj: any):
        super().__init__(obj=obj, _ndim=2)
        # matrix subclass internals
        self._default_tracker = {'derivative': [], 'relation': [], 'origin': []}
        self._tracker = self._default_tracker.copy()

    def instance_track_reset(self) -> None:
        if self._is_valid_tensor(itm=self):
            # reset tracker
            self._tracker = self._default_tracker.copy()
        return None

    def copy(self) -> Matrix | None:
        if self._is_valid_tensor(itm=self):
            # duplicate matrix
            return Matrix(obj=self._tensor)
        return None

    @classmethod
    def replace(cls, replaced: Matrix | str, replacer: Matrix | str) -> None:
        # find replaced and replacer information
        replaced_itm, replaced_id = cls._reference_tensor(itm=replaced)
        replacer_itm, replacer_id = cls._reference_tensor(itm=replacer)
        # move replacer and delete replaced
        replaced.instance_reset()
        replacer._id = replaced._id
        cls._cache[replaced_id] = replaced_itm
        cls._cache[replacer_id] = None
        return None

    @classmethod
    def track_reset(cls, *args: Matrix | str | None) -> None:
        # saved matrices
        args = list(args)
        for i, arg in enumerate(args):
            _, arg_id = cls._reference_tensor(arg)
            args[i] = arg_id

        # track removed matrices
        untracked_matrices = [
            itm for i, itm in enumerate(cls._cache)
            if i not in args and itm is not None and 'track retain' not in itm._tags
        ]
        for matrix in untracked_matrices:
            # track reset track removed matrices
            matrix.instance_track_reset()
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

    class _MethodCollection:
        @staticmethod
        def _ndim(obj: any, ndim: int, force_arr: bool = True) -> None:
            # check ndim
            assert isinstance(ndim, int) and 0 < ndim, "ndim must be a positive integer."
            if force_arr and not (isinstance(obj, np.ndarray) and obj.ndim == ndim):
                # ndim mismatch
                raise ValueError(
                    f"Failed initialization: Passed object was either not a NumPy array"
                    f" or didn't have ndim dimensions. "
                    f"Received object of type {type(obj)} with dimensions "
                    f"{'NULL' if not isinstance(obj, np.ndarray) else obj.ndim}. "
                    f"Expected a NumPy array with ndim {ndim}."
                )
            return None

        @staticmethod
        def _dim_match(obj_1: any, obj_2: any, force_arr: bool = True) -> None:
            if not force_arr and not (isinstance(obj_1, np.ndarray) or isinstance(obj_2, np.ndarray)):
                # not two numpy arrays
                return None
            elif obj_1.shape != obj_2.shape:
                # shape mismatch
                raise ValueError(
                    f"Failed matching: Passed objects were either not both NumPy arrays or didn't match shapes. "
                    f"Received objects {type(obj_1)} and {type(obj_2)} respectively. "
                    f"Object 1 had dimensions {'NULL' if not isinstance(obj_1, np.ndarray) else obj_1.ndim}. "
                    f"Object 2 had dimensions {'NULL' if not isinstance(obj_2, np.ndarray) else obj_2.ndim}. "
                )
            return None

        @staticmethod
        def _req_arr(obj_1: any, obj_2: any) -> None:
            if not (isinstance(obj_1, np.ndarray) or isinstance(obj_2, np.ndarray)):
                # not including two numpy arrays
                raise TypeError(
                    f"Failed objects: One of the two passed objects must be an array. "
                    f"Received objects {type(obj_1)} and {type(obj_2)} respectively."
                )
            return None

        @staticmethod
        def _num_arr(*args: any, force_arr: bool = True) -> None:
            if not force_arr:
                # only numpy array arguments
                args = [arg for arg in args if isinstance(arg, np.ndarray)]
            for arg in args:
                if not np.issubdtype(arg.dtype, np.number):
                    # not number subtype
                    raise TypeError(
                        f"Failed object: A passed object wasn't an array consisting of only real numbers. "
                        f"Identified failed object as {arg}."
                    )
            return None

        @staticmethod
        def _scalar_arr(obj: any, force_arr: bool = True) -> None:
            if not force_arr:
                # not forced numpy array
                return None
            if obj.size != 1:
                # not scalar array
                raise TypeError(
                    f"Failed object: An array meant to be a scalar consisted of dimensions that weren't one. "
                    f"Received object with dims {obj.shape}."
                )
            return None

        @staticmethod
        def _elementwise_broadcast(two_grad: NDArray) -> NDArray:
            assert isinstance(two_grad, np.ndarray) and two_grad.ndim == 2, "two_grad must be a 2D NumPy array."
            # 4D identity creation
            eye = np.zeros((*two_grad.shape, *two_grad.shape))
            np.einsum('ijij -> ij', eye, optimize=False)[:] = 1.0
            # 2D to 4D broadcasting
            return eye * two_grad[np.newaxis, np.newaxis, :, :]

        @staticmethod
        def _scalar_broadcast(two_grad: NDArray) -> NDArray:
            assert isinstance(two_grad, np.ndarray) and two_grad.ndim == 2, "two_grad must be a 2D NumPy array."
            # extend to 4D
            return two_grad[np.newaxis, np.newaxis, :, :]

    class _PairedBaseMethod(_MethodCollection, ABC):
        # todo: docstrings
        @staticmethod
        @abstractmethod
        def forward(main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            r"""
            Args:
                main:
                other:

            Returns:
            """
            pass

        @staticmethod
        @abstractmethod
        def backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            r"""
            Args:
                main:
                other:

            Returns:
            """
            pass

        @staticmethod
        @abstractmethod
        def other_backward(main: NDArray | float | int, other: NDArray) -> NDArray:
            r"""
            Args:
                main:
                other:

            Returns:
            """
            pass

        @staticmethod
        @abstractmethod
        def _forward(main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            pass

        @staticmethod
        @abstractmethod
        def _backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            pass

        @staticmethod
        @abstractmethod
        def _other_backward(main: NDArray | float | int, other: NDArray) -> NDArray:
            pass

        def main(self, main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
            # set main matrix
            if isinstance(main, Matrix) and Matrix._is_valid_tensor(itm=main):
                main_val = main._tensor
            elif isinstance(main, np.ndarray | float | int):
                main_val = main
            else:
                raise TypeError(
                    f"Invalid type: Main object expected valid type Matrix, NumPy array, float, or integer. "
                    f"This method can be run with the Gradient if func:`reduce_grad` is used to convert "
                    f"the Gradient into a Matrix. "
                    f"Received type {type(other)}."
                )

            # set other matrix
            if isinstance(other, Matrix) and Matrix._is_valid_tensor(itm=main):
                other_val = other._tensor
            elif isinstance(other, np.ndarray | float | int):
                other_val = other
            else:
                raise TypeError(
                    f"Invalid type: Other object expected valid type Matrix, NumPy array, float, or integer. "
                    f"This method can be run with the Gradient if func:`reduce_grad` is used to convert "
                    f"the Gradient into a Matrix. "
                    f"Received type {type(other)}."
                )

            # calculate result
            result = Matrix(self._forward(main=main_val, other=other_val))
            result._tracker['origin'] = [main, other]
            if isinstance(other, Matrix):
                # track main
                Matrix._update_track(obj=main, derivative=self._backward, relation=[other, result])
            if isinstance(other, Matrix):
                # track other
                Matrix._update_track(obj=other, derivative=self._other_backward, relation=[main, result])
            return result

    class _LoneBaseMethod(_MethodCollection, ABC):
        # todo: docstrings
        @staticmethod
        @abstractmethod
        def forward(main: NDArray) -> NDArray:
            pass

        @staticmethod
        @abstractmethod
        def backward(main: NDArray) -> NDArray:
            pass

        @staticmethod
        @abstractmethod
        def _forward(main: NDArray) -> NDArray:
            pass

        @staticmethod
        @abstractmethod
        def _backward(main: NDArray) -> NDArray:
            pass

        def main(self, main: Matrix) -> Matrix:
            # check array
            if not isinstance(main, Matrix) and Matrix._is_valid_tensor(itm=main):
                raise TypeError(
                    f"Invalid type: Main object expected valid type Matrix. "
                    f"This method can be run with the Gradient if func:`reduce_grad` is used to convert "
                    f"the Gradient into a Matrix. "
                    f"Received type {type(main)}."
                )
            # calculate result
            result = Matrix(self._forward(main=main._tensor))
            result._tracker['origin'] = [main, None]
            # track main
            Matrix._update_track(obj=main, derivative=self._backward, relation=[None, result])
            return result

    class ElementWiseMethod(_PairedBaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Forward method wasn't implemented and is required for this method type."
            )

        @staticmethod
        @abstractmethod
        def backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Backward method wasn't implemented and is required for this method type."
            )

        @staticmethod
        @abstractmethod
        def other_backward(main: NDArray | float | int, other: NDArray) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Other backward method wasn't implemented and is required for this method type."
            )

        @classmethod
        def _forward(cls, main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            # check input
            cls._ndim(obj=main, ndim=2, force_arr=False)
            cls._ndim(obj=other, ndim=2, force_arr=False)
            cls._dim_match(obj_1=main, obj_2=other, force_arr=False)
            cls._req_arr(obj_1=main, obj_2=other)
            cls._num_arr(main, other, force_arr=False)
            # forward method call
            result = cls.forward(main, other)
            # check output
            cls._ndim(obj=result, ndim=2, force_arr=True)
            cls._num_arr(result, force_arr=True)
            return result

        @classmethod
        def _backward(cls, main: NDArray, other: NDArray | float | int) -> NDArray:
            # check input
            cls._num_arr(main, force_arr=True)
            # backward method call
            result = cls.backward(main, other)
            # check output
            cls._num_arr(result, force_arr=True)
            # broadcast elementwise output
            return cls._elementwise_broadcast(two_grad=result)

        @classmethod
        def _other_backward(cls, main: NDArray | float | int, other: NDArray) -> NDArray:
            # check input
            cls._num_arr(other, force_arr=True)
            # other backward method call
            result = cls.other_backward(other, main)
            # check output
            cls._num_arr(result, force_arr=True)
            # broadcast elementwise output
            return cls._elementwise_broadcast(two_grad=result)

    class LoneElementWiseMethod(_LoneBaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Forward method wasn't implemented and is required for this method type."
            )

        @staticmethod
        @abstractmethod
        def backward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Backward method wasn't implemented and is required for this method type."
            )

        @classmethod
        def _forward(cls, main: NDArray) -> NDArray:
            # check input
            cls._ndim(obj=main, ndim=2, force_arr=True)
            cls._num_arr(main, force_arr=True)
            # forward method call
            result = cls.forward(main)
            # check output
            cls._ndim(obj=result, ndim=2, force_arr=True)
            cls._num_arr(result, force_arr=True)
            return result

        @classmethod
        def _backward(cls, main: NDArray) -> NDArray:
            # backward method call
            result = cls.backward(main)
            # check output
            cls._num_arr(result, force_arr=True)
            # broadcast elementwise output
            return cls._elementwise_broadcast(two_grad=result)

    class ScalarMethod(_LoneBaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Forward method wasn't implemented and is required for this method type."
            )

        @staticmethod
        @abstractmethod
        def backward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Backward method wasn't implemented and is required for this method type."
            )

        @classmethod
        def _forward(cls, main: NDArray) -> NDArray:
            # check input
            cls._ndim(obj=main, ndim=2, force_arr=True)
            cls._num_arr(main, force_arr=True)
            # forward method call
            result = cls.forward(main)
            # check output
            cls._ndim(obj=result, ndim=2, force_arr=True)
            cls._num_arr(result, force_arr=True)
            cls._scalar_arr(obj=result, force_arr=True)
            return result

        @classmethod
        def _backward(cls, main: NDArray) -> NDArray:
            # backward method call
            result = cls.backward(main)
            # check output
            cls._num_arr(result, force_arr=True)
            # broadcast elementwise output
            return cls._scalar_broadcast(two_grad=result)

    class CustomMethod(_PairedBaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Forward method wasn't implemented and is required for this method type."
            )

        @staticmethod
        @abstractmethod
        def backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Backward method wasn't implemented and is required for this method type."
            )

        @staticmethod
        @abstractmethod
        def other_backward(main: NDArray | float | int, other: NDArray) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Other backward method wasn't implemented and is required for this method type."
            )

        @classmethod
        def _forward(cls, main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            # check input
            cls._ndim(obj=main, ndim=2, force_arr=False)
            cls._ndim(obj=other, ndim=2, force_arr=False)
            cls._req_arr(obj_1=main, obj_2=other)
            cls._num_arr(main, other, force_arr=False)
            # forward method call
            result = cls.forward(main, other)
            # check output
            cls._ndim(obj=result, ndim=2, force_arr=True)
            cls._num_arr(result, force_arr=True)
            return result

        @classmethod
        def _backward(cls, main: NDArray, other: NDArray | float | int) -> NDArray:
            # check input
            cls._num_arr(main, force_arr=True)
            # backward method call
            result = cls.backward(main, other)
            # check output
            cls._ndim(obj=result, ndim=4, force_arr=True)
            cls._num_arr(result, force_arr=True)
            return result

        @classmethod
        def _other_backward(cls, main: NDArray | float | int, other: NDArray) -> NDArray:
            # check input
            cls._num_arr(other, force_arr=True)
            # other backward method call
            result = cls.other_backward(other, main)
            # check output
            cls._ndim(obj=result, ndim=4, force_arr=True)
            cls._num_arr(result, force_arr=True)
            return result

    class LoneCustomMethod(_LoneBaseMethod):
        @staticmethod
        @abstractmethod
        def forward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Forward method wasn't implemented and is required for this method type."
            )

        @staticmethod
        @abstractmethod
        def backward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Missing implementation: "
                "Backward method wasn't implemented and is required for this method type."
            )

        @classmethod
        def _forward(cls, main: NDArray) -> NDArray:
            # check input
            cls._ndim(obj=main, ndim=2, force_arr=True)
            cls._num_arr(main, force_arr=True)
            # forward method call
            result = cls.forward(main)
            # check output
            cls._ndim(obj=result, ndim=2, force_arr=True)
            cls._num_arr(result, force_arr=True)
            return result

        @classmethod
        def _backward(cls, main: NDArray) -> NDArray:
            # backward method call
            result = cls.backward(main)
            # check output
            cls._num_arr(result, force_arr=True)
            cls._ndim(obj=result, ndim=4, force_arr=True)
            return result

    # built-in methods
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
        def other_backward(main: NDArray, other: NDArray) -> NDArray:
            four_concat = main[:, np.newaxis, :, np.newaxis]
            eye = np.eye(other.shape[1])[np.newaxis, :, np.newaxis, :]
            return four_concat * eye

    class _Pow(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            return main ** other

        @staticmethod
        @inf_remove(inf_val=1e10)
        def backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            two_grad = other * (main ** (other - 1.0))
            return two_grad

        @staticmethod
        @inf_remove(inf_val=1e10)
        def other_backward(main: NDArray | float | int, other: NDArray) -> NDArray:
            two_grad = np.log(main) * (main ** other)
            return two_grad

    class _Mul(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            return main * other

        @staticmethod
        def backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            return other

        @staticmethod
        def other_backward(main: NDArray | float | int, other: NDArray) -> NDArray:
            return main

    class _TrueDiv(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            return main / other

        @staticmethod
        @inf_remove(inf_val=1e10)
        def backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            return other ** -1.0

        @staticmethod
        @inf_remove(inf_val=1e10)
        def other_backward(main: NDArray | float | int, other: NDArray) -> NDArray:
            return -main / other ** 2.0

    class _Add(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            return main + other

        @staticmethod
        def backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            return np.ones(main.shape)

        @staticmethod
        def other_backward(main: NDArray | float | int, other: NDArray) -> NDArray:
            return np.ones(other.shape)

    class _Sub(ElementWiseMethod):
        @staticmethod
        def forward(main: NDArray | float | int, other: NDArray | float | int) -> NDArray:
            return main - other

        @staticmethod
        def backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            return np.ones(main.shape)

        @staticmethod
        def other_backward(main: NDArray | float | int, other: NDArray) -> NDArray:
            return -np.ones(other.shape)

    # internal instance
    _matmul = _MatMul()
    _pow = _Pow()
    _mul = _Mul()
    _truediv = _TrueDiv()
    _add = _Add()
    _sub = _Sub()

    # raw method calls
    @classmethod
    def rmatmul(cls, main: Matrix | NDArray, other: Matrix | NDArray) -> Matrix:
        r"""**Raw matrix multiplication.**"""
        return cls._matmul.main(main, other)

    @classmethod
    def rpow(cls, main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
        r"""Raw Hadamard power."""
        return cls._pow.main(main, other)

    @classmethod
    def rmul(cls, main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
        r"""Raw Hadamard multiplication."""
        return cls._mul.main(main, other)

    @classmethod
    def rtruediv(cls, main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
        r"""Raw Hadamard division."""
        return cls._truediv.main(main, other)

    @classmethod
    def radd(cls, main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
        r"""Raw addition."""
        return cls._add.main(main, other)

    @classmethod
    def rsub(cls, main: Matrix | NDArray | float | int, other: Matrix | NDArray | float | int) -> Matrix:
        r"""Raw subtraction."""
        return cls._sub.main(main, other)

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


class Gradient(_Tensor):
    # gradient memory internals
    _cache: list[Gradient | None] = []
    _prefix: str = 'g'

    def __init__(self, obj: any, *, _override: bool = False):
        assert _override, "Raw Gradient creation isn't recommended; to create a Gradient object, signal override."
        super().__init__(obj=obj, _ndim=4)
        # gradient subclass internals
        self._default_tracker = {'chain': []}
        self._tracker = self._default_tracker.copy()

    def reduce_grad(self) -> Matrix | None:
        if self._is_valid_tensor(itm=self):
            return Matrix(np.sum(self._tensor, axis=(0, 1)))
        return None

    def _track_instance(self):
        alt_tracker = {'chain': self._unpack_ids(self._tracker.copy()['chain'])}
        return alt_tracker

    @staticmethod
    def _chain_opr(up: NDArray, down: NDArray) -> NDArray:
        # 6D downstream expansion
        down = down[:, :, :, :, np.newaxis, np.newaxis]
        # 6D upstream expansion
        up = up[np.newaxis, np.newaxis, :, :, :, :]
        # 6D to 4D manipulation
        return np.sum(down * up, axis=(2, 3))

    @staticmethod
    def nabla(grad: Matrix, wrt: Matrix, *, binary: bool = True) -> Gradient:
        # check matrices
        if not isinstance(grad, Matrix) and Matrix._is_valid_tensor(itm=grad):
            raise TypeError(
                f"Invalid type: Gradient calculation can only be done with valid Matrix objects. "
                f"Received grad object of type {type(grad)}."
            )
        if not isinstance(wrt, Matrix) and Matrix._is_valid_tensor(itm=wrt):
            raise TypeError(
                f"Invalid type: Gradient calculation can only be done with valid Matrix objects. "
                f"Received wrt object of type {type(grad)}."
            )

        # initialize relation
        relation = None
        if not binary:
            relation = []

        def _relate(item, target, trace=None):
            # NB: This finds relations by going upstream.
            # The final calculation uses this upstream order, while the chain is stored downstream.
            nonlocal relation
            if trace is None:
                # reset trace
                trace = []
            # NB: This only traces through the Matrix object.
            # Non-Matrix objects are still used in computation if necessary.
            if binary and relation is None and isinstance(item, Matrix):
                # get origins
                origins = [Matrix.reference(idx=org) for org in item.tracker['origin']]
                trace.append(item)
                if target in origins:
                    # related
                    trace.append(target)
                    relation = trace.copy()
                else:
                    # continue search
                    [_relate(item=origin, target=target, trace=trace.copy()) for origin in origins]
            elif not binary and isinstance(item, Matrix):
                # get origins
                origins = [Matrix.reference(org) for org in item.tracker['origin']]
                trace.append(item)
                if target in origins:
                    # related
                    trace.append(target)
                    relation.append(trace.copy())
                else:
                    # continue search
                    [_relate(item=origin, target=target, trace=trace.copy()) for origin in origins]

        # relate matrices
        _relate(wrt, grad)
        if not relation:
            # no relation
            raise TrackingError(grad=grad, wrt=wrt, message=(
                f"No relation could be found between {grad.id} and {wrt.id}.\n"
                f"This might be due to:\n"
                f"   No clear relation between the Matrices.\n"
                f"   Accidental clearing of trackers.\n"
                f"   Deletion of intermediate Matrices.\n"
                f"   Accidental reference to the wrong Matrix."
            ))

        def _derive(up: Matrix, down: Matrix) -> NDArray:
            # get relations
            strm_result = [Matrix.reference(idx=rlt_itm[1]) for rlt_itm in up.tracker['relation']]
            strm_other = [Matrix.reference(idx=rlt_itm[0]) for rlt_itm in up.tracker['relation']]
            # get operation
            drv_operator = up.tracker['derivative'][strm_result.index(down)]
            other = strm_other[strm_result.index(down)]

            if isinstance(other, Matrix):
                # get value
                other = other._tensor
            # calculate local gradient
            try:
                # pair derivative method
                res = drv_operator(main=up._tensor, other=other)
            except TypeError:
                # lone derivative method
                res = drv_operator(main=up._tensor)
            return res

        # NB: Automatic chain-ruling occurs back to front.
        # This is fastest with most NN-type architectures.
        if binary:
            # initial and final non-nested
            chain = [grad] + [relation.copy()[-2:0:-1]] + [wrt]
            chain = [chn for chn in chain if chn]
            # calculate initial gradient
            result = _derive(up=relation[1], down=relation[0])
            del relation[0]
            while 1 < len(relation):
                # chain rule gradients
                result = Gradient._chain_opr(up=_derive(up=relation[1], down=relation[0]), down=result)
                del relation[0]
        else:
            # add gradients
            grads = []
            # initial and final non-nested
            chain = [grad] + [[rlt[-2:0:-1] for rlt in relation.copy() if rlt]] + [wrt]
            for rlt in relation:
                # calculate initial local gradient
                op_res = _derive(up=rlt[1], down=rlt[0])
                del rlt[0]
                while 1 < len(rlt):
                    # chain rule local gradients
                    op_res = Gradient._chain_opr(up=_derive(up=rlt[1], down=rlt[0]), down=op_res)
                    del rlt[0]
                grads.append(op_res)
            result = 0
            for grad in grads:
                # accumulate alternate-path gradients
                result += grad

        # return final gradient
        result = Gradient(obj=result, _override=True)
        result._tracker['chain'] = chain
        return result

    @staticmethod
    def chain(up: Gradient, down: Gradient) -> Gradient:
        # check gradients
        if not isinstance(up, Gradient) and Gradient._is_valid_tensor(itm=up):
            raise TypeError(
                f"Invalid type: Chain-ruling can only be done with valid Gradient objects. "
                f"Received up object of type {type(down)}."
            )
        if not isinstance(down, Gradient) and Gradient._is_valid_tensor(itm=up):
            raise TypeError(
                f"Invalid type: Chain-ruling can only be done with valid Gradient objects. "
                f"Received down object of type {type(down)}."
            )

        # check relation
        if up._tracker['chain'][-1] != down._tracker['chain'][0]:
            raise TrackingError(grad=down, wrt=up, message=(
                f"No relation could be found between {down.id} and {up.id}.\n"
                f"This might be due to:\n"
                f"   No immediate link between Gradients.\n"
                f"   Accidental reference to the wrong Gradient."
            ))

        # chain-rule gradients
        result = Gradient(obj=Gradient._chain_opr(up=up._tensor, down=down._tensor), _override=True)
        # set gradient internals
        result._tracker['chain'] = up._tracker['chain'][:-1] + down._tracker['chain']
        # return final gradient
        return result
