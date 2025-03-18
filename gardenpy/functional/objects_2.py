r"""
**GardenPy objects.**

Core objects for the garden library

Contains:
    - class:`_Array`
    - class:`Matrix`
    - class:`Gradient`
"""

from __future__ import annotations
from typing import Self
from warnings import warn
import numpy as np
from numpy.typing import NDArray

from ..utils.errors import TrackingError


class _Array:
    # caches
    _matrix_cache: list[Matrix | None] = []
    _gradient_cache: list[Gradient | None] = []
    _cache: dict[str, list[_Array | Matrix | Gradient | None]] = {'matrix': _matrix_cache, 'gradient': _gradient_cache}
    # ikwiad
    _ikwiad: bool = False

    def __init__(self, obj: any, *, _type: str, _ndim: int):
        # verify object
        obj = np.array(obj)
        if not np.issubdtype(obj.dtype, np.number) or obj.ndim != _ndim:
            # NB: Dimensional verification to check dimensional consistency.
            # Currently, two for matrices and four for gradients.
            raise TypeError(f"Attempted creation with object that wasn't {_ndim}-dimensional with only real numbers.")

        # tracking internals
        assert _type in ('matrix', 'gradient')
        self._type: str | None = str(_type)
        self._id: int | None = None
        self._array: NDArray | None = obj
        # autodiff internals
        self._default_tracker: dict[str, _Array | list | None] | None = None
        self._tracker: dict[str, _Array | list | None] | None = None
        # other internals
        self._ndim: int = _ndim
        self._tags: list[str] = []

        # cache
        _Array._add_cache(itm=self)

    def __repr__(self) -> str:
        self._is_valid_array(itm=self)
        return str(self._array)

    @property
    def id(self) -> str | None:
        if self._is_valid_array(itm=self):
            # indicator id
            return f"{self._type[0]}{hex(self._id)}"
        return None

    @property
    def type(self) -> str | None:
        self._is_valid_array(itm=self)
        return self._type

    @property
    def tracker(self) -> dict[str, _Array | list | str | None] | None:
        if not self._is_valid_array(itm=self):
            # invalid array
            return None
        # ikwiad on
        user_ikwiad = _Array._ikwiad
        _Array._ikwiad = True
        # tracker debug conversion
        alt_tracker = self._tracker.copy()

        # alter alt_tracker
        def _unpack_ids(itm: _Array | list | None) -> property | None:
            if isinstance(itm, _Array):
                return _Array.id
            elif isinstance(itm, list):
                return _unpack_ids(itm)
            else:
                return None
        if self._type == 'matrix':
            alt_tracker['rlt'] = _unpack_ids(alt_tracker['rlt'])
            alt_tracker['org'] = _unpack_ids(alt_tracker['org'])
        elif self._type == 'gradient':
            alt_tracker['rlt'] = _unpack_ids(alt_tracker['rlt'])

        # ikwiad reset
        _Array._ikwiad = user_ikwiad
        return {'id': self._id, 'tags': self._tags, **alt_tracker}

    @property
    def array(self) -> NDArray | None:
        self._is_valid_array(itm=self)
        return self._array

    @property
    def shape(self) -> tuple[int, ...] | None:
        if self._is_valid_array(itm=self):
            return self._array.shape
        return None

    @property
    def tags(self) -> list[str]:
        self._is_valid_array(itm=self)
        return self._tags

    @tags.setter
    def tags(self, tag: str) -> Self:
        self._is_valid_array(itm=self)
        self._tags.append(str(tag))

    def instance_track_reset(self) -> None:
        if self._is_valid_array(itm=self):
            self._tracker = self._default_tracker
        return None

    def instance_reset(self) -> None:
        if self._is_valid_array(itm=self):
            # clear cache
            _Array._cache[self._type][self._id] = None
            # reset internals
            self._id = None
            self._array = None
            self._type = None
            self._tracker = None
            self._tags.append('deleted')
        return None

    def copy(self) -> _Array | None:
        if self._is_valid_array(itm=self):
            # duplicate array
            return _Array(obj=self._array, _type=self._type, _ndim=self._ndim)
        return None

    @classmethod
    def cache(cls) -> dict[str, list[str | None]]:
        # subclass caches
        matrices = [itm.id if itm is not None else None for itm in _Array._matrix_cache.copy()]
        gradients = [itm.id if itm is not None else None for itm in _Array._gradient_cache.copy()]
        # full cache
        return {'matrix': matrices, 'gradient': gradients}

    @classmethod
    def reference(cls, idx: str, atype: str | None = None) -> _Array:
        # reference array
        array, _, _ = _Array._reference_array(itm=idx, atype=atype)
        return array

    @classmethod
    def reset(cls, *args: _Array | str | None) -> None:
        # find saved arrays
        args = list(args)
        mat_args, grad_args = [], []
        for arg in args:
            _, atype, arg = _Array._reference_array(itm=arg)
            if atype == 'matrix':
                mat_args.append(arg)
            elif atype == 'gradient':
                grad_args.append(arg)
        # remove arrays
        removed_mats = [itm for i, itm in enumerate(_Array._matrix_cache) if i not in mat_args and itm is not None]
        removed_grads = [itm for i, itm in enumerate(_Array._gradient_cache) if i not in grad_args and itm is not None]
        for instance in removed_mats:
            instance.instance_reset()
        for instance in removed_grads:
            instance.instance_reset()
        return None

    @classmethod
    def cache_reset(cls, cache: str, *args: _Array | str | None) -> None:
        if cache not in ('matrix', 'gradient'):
            raise ValueError(...)
        fin_args = []
        for arg in args:
            _, atype, arg = _Array._reference_array(itm=arg)
            if atype == cache:
                fin_args.append(arg)
        # remove arrays
        removed_arrays = [itm for i, itm in enumerate(_Array._cache[cache]) if i not in fin_args and itm is not None]
        for instance in removed_arrays:
            instance.instance_reset()
        return None

    @classmethod
    def grad_reset(cls, *args: _Array | str | None) -> None:
        # find saved arrays
        args = list(args)
        fin_args = []
        for arg in args:
            _, atype, arg = _Array._reference_array(arg)
            if atype == 'matrix':
                fin_args.append(arg)
            elif not _Array._ikwiad:
                warn("", UserWarning)

        removed_arrays = [itm for i, itm in enumerate(_Array._matrix_cache) if i not in fin_args and itm is not None]
        _Array.cache_reset(cache='gradient')
        for array in removed_arrays:
            array.instance_track_reset()
        return None

    @classmethod
    def replace(cls, replaced: Matrix | str, replacer: Matrix | str) -> None:
        replaced_itm, replaced_type, replaced_id = _Array._reference_array(itm=replaced)
        replacer_itm, replacer_type, replacer_id = _Array._reference_array(itm=replacer)
        if replaced_type != 'matrix' or replacer_type != 'matrix':
            raise TypeError
        replaced.instance_reset()
        replacer._id = replaced._id
        _Array._matrix_cache[replaced_id] = replaced_itm  # todo: ??????
        _Array._matrix_cache[replacer_id] = None
        return None


    @classmethod
    def zero_grad(cls, *args: Matrix | str) -> None:
        # reset arrays
        _Array.reset(*args)
        # reset trackers
        _Array.grad_reset(*args)
        return None

    @classmethod
    def ikwiad(cls, ikwiad: bool | None = None) -> None:
        if ikwiad is None:
            _Array._ikwiad = not _Array._ikwiad
            return None
        _Array._ikwiad = bool(ikwiad)
        return None

    @staticmethod
    def _is_valid_array(itm: _Array) -> bool:
        if itm._type is not None:
            # valid array
            return True
        else:
            # invalid array
            if not _Array._ikwiad:
                warn("Detected deleted array reference.", UserWarning)
            return False

    @classmethod
    def _reference_array(
            cls,
            itm: _Array | Matrix | Gradient | str | int, atype: str | None = None
    ) -> tuple[_Array | Matrix | Gradient, str, int]:
        # ikwiad on
        user_ikwiad = _Array._ikwiad
        _Array._ikwiad = True

        # attempt pointer reference
        in_atype = None
        try:
            assert isinstance(itm, str)
            itm = int(itm[1:], 16)
            if str(itm)[0] == 'm':
                in_atype = 'matrix'
            elif str(itm)[0] == 'g':
                in_atype = 'gradient'
        except AssertionError:
            pass

        if isinstance(itm, _Array):
            # array type
            itm_id = itm._id
            in_atype = itm._type
        elif isinstance(itm, int):
            # check index reference
            in_atype = in_atype or atype
            if in_atype not in ('matrix', 'gradient'):
                # invalid type
                raise ValueError(
                    "Attempted reference without array type."  # todo: beef up error message
                )
            if len(_Array._cache[in_atype]) <= itm:
                # out of index
                raise ValueError(
                    "Attempted reference outside Array instance list. "
                    f"Currently, instance list only contains {len(_Array._cache[in_atype])} items. "
                    f"A reference has been made to index {itm}."
                )
            # use index reference
            itm_id = itm
            itm = _Array._cache[in_atype][itm_id]
        else:
            # invalid reference
            raise TypeError("Attempted Array reference with an invalid type.")
        if not _Array._is_valid_array(itm=itm):
            # invalid array
            raise TypeError("Attempted reference to a deleted Array.")

        # ikwiad reset
        _Array._ikwiad = user_ikwiad
        return itm, in_atype, itm_id

    @classmethod
    def _add_cache(cls, itm: _Array) -> None:
        try:
            # use unused cache location
            open_id = _Array._cache[itm._type].index(None)
            itm._global_id = open_id
            _Array._cache[itm._type][open_id] = itm
        except ValueError:
            # create new cache location
            open_id = len(_Array._cache[itm._type])
            _Array._cache[itm._type].append(itm)
            itm._id = open_id

    @staticmethod
    def _inf_remove(func: callable) -> callable:
        def wrapper(*args, **kwargs) -> NDArray:
            array = func(*args, **kwargs)
            # replace infinities
            return np.where(np.isposinf(array), 1e10, np.where(np.isneginf(array), -1e10, array))
        return wrapper


########################################################################################################################


class Matrix(_Array):
    def __init__(self, obj: any):
        super().__init__(obj=obj, _ndim=2, _type='matrix')
        # matrix subclass internals
        self._default_tracker = {'drv': [], 'rlt': [], 'org': []}
        self._tracker = self._default_tracker

    @staticmethod
    def _update_track(obj: Matrix, drv: any, rlt: any) -> None:
        # update tracker
        obj._tracker['drv'].append(drv)
        obj._tracker['rlt'].append(rlt)
        return None

    class _BaseMethod:
        @staticmethod
        def _four_broadcast_e(two_grad: NDArray) -> NDArray:
            assert isinstance(two_grad, np.ndarray) and two_grad.ndim == 2
            # 4D identity creation
            eye = np.zeros((*two_grad.shape, *two_grad.shape))
            np.einsum('ijij -> ij', eye, optimize=False)[:] = 1.0
            # 2D to 4D broadcasting
            return eye * two_grad[np.newaxis, np.newaxis, :, :]

        @staticmethod
        def _four_broadcast_s(two_grad: NDArray) -> NDArray:
            assert isinstance(two_grad, np.ndarray) and two_grad.ndim == 2
            # extend to 4D
            return two_grad[np.newaxis, np.newaxis, :, :]

        @classmethod
        def ndim_arg(cls, ndim: int) -> callable:
            def decorator(func: callable) -> callable:
                def wrapper(*args: any, **kwargs: any) -> NDArray:
                    if not all(not isinstance(arg, np.ndarray) or arg.ndim == ndim for arg in args):
                        raise ValueError()
                    if not all(not isinstance(value, np.ndarray) or value.ndim == ndim for value in kwargs.values()):
                        raise ValueError()
                    return func(*args, **kwargs)

                return wrapper

            return decorator

        @classmethod
        def ndim_result(cls, ndim: int) -> callable:
            def decorator(func: callable) -> callable:
                def wrapper(*args: any, **kwargs: any) -> NDArray:
                    result = func(*args, **kwargs)
                    if not (isinstance(result, np.ndarray) and result.ndim == ndim):
                        raise ValueError
                    return result

                return wrapper

            return decorator

        @classmethod
        def dim_match(cls, func: callable) -> callable:
            def wrapper(main: NDArray, other: NDArray | float | int) -> NDArray:
                if isinstance(other, np.ndarray):
                    # check matching dimensions
                    assert main.shape == other.shape
                return func(main, other)

            return wrapper

        @classmethod
        def elementwise_broadcast(cls, func: callable) -> callable:
            def wrapper(*args: any, **kwargs: any) -> NDArray:
                result = func(*args, **kwargs)
                # four broadcast elementwise
                return cls._four_broadcast_e(two_grad=result)

            return wrapper

        @classmethod
        def scalar_broadcast(cls, func: callable) -> callable:
            def wrapper(*args: any, **kwargs: any) -> NDArray:
                result = func(*args, **kwargs)
                # four broadcast scalar
                return cls._four_broadcast_s(two_grad=result)

            return wrapper

        @classmethod
        def inf_remove(cls, *, inf_val: float | int = 1e10) -> callable:
            def decorator(func: callable) -> callable:
                def wrapper(*args: any, **kwargs: any) -> NDArray:
                    array = func(*args, **kwargs)
                    assert isinstance(array, np.ndarray)
                    # inf to inf_val
                    return np.where(np.isposinf(array), inf_val, np.where(np.isneginf(array), -inf_val, array))

                return wrapper

            return decorator

        @staticmethod
        def forward(*args: any, **kwargs: any) -> NDArray:
            ...

        @staticmethod
        def backward(*args: any, **kwargs: any) -> NDArray:
            ...

        @staticmethod
        def backward_o(*args: any, **kwargs: any) -> NDArray:
            ...

        def main(self, *args: any, **kwargs: any) -> 'Matrix':
            ...

    class LoneElementWiseMethod(_BaseMethod):
        @staticmethod
        @super().ndim_arg(ndim=2)
        def forward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        @super().ndim_arg(ndim=2)
        @super().elementwise_broadcast
        def backward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        def main(self, main: Matrix) -> Matrix:
            # check array
            if not isinstance(main, Matrix):
                raise TypeError(
                    "Non-matrix call"  # todo: beef up error message
                )
            # calculate result
            result = Matrix(self.forward(main._array))
            result._tracker['org'] = [main, None]
            # track main
            Matrix._update_track(obj=main, drv=self.backward, rlt=[None, result])
            # return result
            return result

    class ElementWiseMethod(_BaseMethod):
        @staticmethod
        @super().ndim_arg(ndim=2)
        @super().dim_match
        def forward(main: NDArray, other: NDArray | float | int) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        @super().ndim_arg(ndim=2)
        @super().elementwise_broadcast
        def backward(main: NDArray, other: NDArray | float | int) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        def main(self, main: Matrix, other: Matrix | NDArray | float | int) -> Matrix:
            # check main array
            if not isinstance(main, Matrix):
                raise TypeError(
                    "..."  # todo: error message
                )

            # set array
            if isinstance(other, Matrix):
                arr = other._array
            elif isinstance(other, np.ndarray | float | int):
                arr = other
            else:
                raise TypeError()  # todo: error message

            # calculate result
            result = Matrix(self.forward(main._array, arr))
            result._tracker['org'] = [main, other]
            # track main
            Matrix._update_track(obj=main, drv=self.backward, rlt=[other, result])
            if isinstance(other, Matrix):
                # track other
                Matrix._update_track(obj=other, drv=self.backward_o, rlt = [main, result])
            return result

    class ScalarMethod(_BaseMethod):
        @staticmethod
        @super().ndim_arg(ndim=2)
        def forward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        @super().ndim_arg(ndim=2)
        @super().scalar_broadcast
        def backward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        def main(self, main: Matrix) -> Matrix:
            # check array
            if not isinstance(main, Matrix):
                raise TypeError(
                    "Non-matrix call"  # todo: beef up error message
                )
            # calculate result
            result = Matrix(self.forward(main._array))
            result._tracker['org'] = [main, None]
            # track main
            Matrix._update_track(obj=main, drv=self.backward, rlt=[None, result])
            # return result
            return result

    class CustomMethod(_BaseMethod):
        @staticmethod
        @super().ndim_arg(ndim=2)
        def forward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        @super().ndim_arg(ndim=2)
        @super().ndim_result(ndim=4)
        def backward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        def main(self, main: Matrix, other: Matrix | NDArray | float | int) -> Matrix:
            # check main array
            if not isinstance(main, Matrix):
                raise TypeError(
                    "..."  # todo: error message
                )

            # set array
            if isinstance(other, Matrix):
                arr = other._array
            elif isinstance(other, np.ndarray | float | int):
                arr = other
            else:
                raise TypeError()  # todo: error message

            # calculate result
            result = Matrix(self.forward(main._array, arr))
            result._tracker['org'] = [main, other]
            # track main
            Matrix._update_track(obj=main, drv=self.backward, rlt=[other, result])
            if isinstance(other, Matrix):
                # track other
                Matrix._update_track(obj=other, drv=self.backward_o, rlt=[main, result])
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
        @super().inf_remove(inf_val=1e10)
        def backward(main: NDArray, other: NDArray) -> NDArray:
            two_grad = other * (main ** (other - 1.0))
            return two_grad

        @staticmethod
        @super().inf_remove(inf_val=1e10)
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
        @super().inf_remove(inf_val=1e10)
        def backward(main: NDArray, other: NDArray) -> NDArray:
            return other ** -1.0

        @staticmethod
        @super().inf_remove(inf_val=1e10)
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
    def __init__(self, obj: any, *, _signal_override: bool = False):
        if not _signal_override and not _Array._ikwiad:
            warn(f"", UserWarning)  # todo: beef up warning
        super().__init__(obj=obj, _ndim=4, _type='gradient')
        # gradient subclass internals
        self._default_tracker = {'rlt': []}
        self._tracker = self._default_tracker

    def reduce_grad(self) -> Matrix:
        return Matrix(np.sum(self._array, axis=(0, 1)))

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
                origins = [_Array.reference(org) for org in item.tracker['org']]
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
                origins = [_Array.reference(org) for org in item.tracker['org']]
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
            strm_result = [_Array.reference(rlt[1]) for rlt in up.tracker['rlt']]
            strm_other = [_Array.reference(rlt[0]) for rlt in up.tracker['rlt']]
            # get operation
            drv_operator = up.tracker['drv'][strm_result.index(down)]
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
            res = Gradient(obj=res, _signal_override=True)

            # local gradient setup
            res._tracker['rlt'] += [down, up]
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
            result = Gradient(obj=result, _signal_override=True)
            result._tracker = track

        # return final gradient
        if linear_override:
            result._tags.append('linear override')
        return result

    @staticmethod
    def _chain_opr(down: NDArray, up: NDArray) -> NDArray:
        # 6D downstream expansion
        down = down[:, :, :, :, np.newaxis, np.newaxis]
        # 6D upstream expansion
        up = up[np.newaxis, np.newaxis, :, :, :, :]
        # 6D to 4D manipulation
        return np.sum(down * up, axis=(2, 3))

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
        down_relation = down._tracker['rlt'][-1]
        up_relation = up._tracker['rlt'][0]
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
        result = Gradient(obj=Gradient._chain_opr(down=down._array, up=up._array), _signal_override=True)
        # set gradient internals
        result._type = 'gradient'
        result._tracker['rlt'] = down._tracker['rlt'] + up._tracker['rlt'][1:]
        # return final gradient
        return result
