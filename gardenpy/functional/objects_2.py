from typing import Dict, List, Union, Optional, Tuple
from warnings import warn
import numpy as np
from numpy.typing import NDArray

from ..utils.errors import TrackingError


class _Array:
    # caches
    _matrix_cache: List[Union['Matrix', None]] = []
    _gradient_cache: List[Union['Gradient', None]] = []
    _cache: Dict[str, List[Union['_Array', None]]] = {'matrix': _matrix_cache, 'gradient': _gradient_cache}
    # ikwiad
    _ikwiad: bool = False

    def __init__(self, obj: any, *, _type: str, _dims: int):
        # verify object
        obj = np.array(obj)
        if not np.issubdtype(obj.dtype, np.number) or len(obj.shape) != _dims:
            # NB: Dimensional verification to check dimensional consistency.
            # Currently, two for matrices and four for gradients.
            raise TypeError(f"Attempted creation with object that wasn't {_dims}-dimensional with only real numbers.")

        # tracking internals
        assert _type in ('matrix', 'gradient')
        self._type: Union[str, None] = str(_type)
        self._id: Union[int, None] = None
        self._array: Union[NDArray, None] = obj
        # autodiff internals
        self._default_tracker: Union[Dict[str, Union['_Array', list, None]], None] = None
        self._tracker: Union[Dict[str, Union['_Array', list, None]], None] = None
        # other internals
        self._dims: int = _dims
        self._tags: List[str] = []

        # cache
        _Array._add_cache(itm=self)

    def __repr__(self) -> str:
        self._is_valid_array(itm=self)
        return str(self._array)

    @property
    def id(self) -> Union[str, None]:
        if self._is_valid_array(itm=self):
            # indicator id
            return f"{self._type[0]}{hex(self._id)}"
        return None

    @property
    def type(self) -> Union[str, None]:
        self._is_valid_array(itm=self)
        return self._type

    @property
    def internals(self) -> Union[Dict[str, Union['_Array', list, str, None]], None]:
        if not self._is_valid_array(itm=self):
            # invalid array
            return None
        # ikwiad on
        user_ikwiad = _Array._ikwiad
        _Array._ikwiad = True
        # tracker debug conversion
        alt_tracker = self._tracker.copy()
        # todo: alt tracker
        # ikwiad reset
        _Array._ikwiad = user_ikwiad
        return {'id': self._id, 'tags': self._tags, **alt_tracker}

    @property
    def array(self) -> Union[NDArray, None]:
        self._is_valid_array(itm=self)
        return self._array

    @property
    def shape(self) -> Union[Tuple[int, ...], None]:
        if self._is_valid_array(itm=self):
            return self._array.shape
        return None

    @property
    def tags(self) -> List[str]:
        self._is_valid_array(itm=self)
        return self._tags

    @tags.setter
    def tags(self, tag: str) -> None:
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

    def copy(self) -> Union['_Array', None]:
        if self._is_valid_array(itm=self):
            # duplicate array
            return _Array(obj=self._array, _type=self._type, _dims=self._dims)
        return None

    @classmethod
    def cache(cls) -> Dict[str, List[Union[str, None]]]:
        # subclass caches
        matrices = [itm.id if itm is not None else None for itm in _Array._matrix_cache.copy()]
        gradients = [itm.id if itm is not None else None for itm in _Array._gradient_cache.copy()]
        # full cache
        return {'matrix': matrices, 'gradient': gradients}

    @classmethod
    def reference(cls, idx: str, atype: Optional[str] = None) -> '_Array':
        # reference array
        array, _, _ = _Array._reference_array(itm=idx, atype=atype)
        return array

    @classmethod
    def reset(cls, *args: Optional[Union['_Array', str]]) -> None:
        # find saved arrays
        args = list(args)
        mat_args, grad_args = [], []
        for arg in args:
            _, atype, arg = _Array._reference_array(arg)
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
    def grad_reset(cls, *args: Optional[Union['_Array', str]]) -> None:
        # ikwiad on
        user_ikwiad = _Array._ikwiad
        _Array._ikwiad = True

        # find saved arrays
        args = list(args)
        mat_args, grad_args = [], []
        for arg in args:
            _, atype, arg = _Array._reference_array(arg)
            if atype == 'matrix':
                mat_args.append(arg)
            elif atype == 'gradient':
                grad_args.append(arg)

    @staticmethod
    def _is_valid_array(itm: '_Array') -> bool:
        if itm._type is not None:
            # valid array
            return True
        else:
            # invalid array
            if not _Array._ikwiad:
                warn("Detected deleted array reference.", UserWarning)
            return False

    @classmethod
    def _reference_array(cls, itm: Union['_Array', str, int], atype: Optional[str] = None) -> Tuple['_Array', str, int]:
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
    def _add_cache(cls, itm: '_Array') -> None:
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
        super().__init__(obj=obj, _dims=2, _type='matrix')
        # matrix subclass internals
        self._default_tracker = {'drv': [], 'rlt': [], 'org': []}
        self._tracker = self._default_tracker

    @staticmethod
    def _update_track(obj: 'Matrix', drv: any, rlt: any) -> None:
        # update tracker
        obj._tracker['drv'].append(drv)
        obj._tracker['rlt'].append(rlt)
        return None

    class _BaseMethod:
        @classmethod
        def assert_four(cls, func: callable) -> callable:
            def wrapper(*args, **kwargs) -> NDArray:
                # 4D result assertion
                result = func(*args, **kwargs)
                assert len(result.shape) == 4
                return result
            return wrapper

        @classmethod
        def four_broadcast_e(cls, two_grad: NDArray) -> NDArray:
            assert len(two_grad.shape) == 2
            # 4D identity creation
            eye = np.zeros((*two_grad.shape, *two_grad.shape))
            np.einsum('ijij -> ij', eye, optimize=False)[:] = 1.0
            # 2D to 4D broadcasting
            return eye * two_grad[np.newaxis, np.newaxis, :, :]

        @classmethod
        def four_broadcast_s(cls, two_grad: NDArray) -> NDArray:
            assert len(two_grad.shape) == 2
            # extend to 4D
            return two_grad[np.newaxis, np.newaxis, :, :]

        @classmethod
        def inf_remove(cls, func: callable, *, inf_val: Union[float, int] = 1e10) -> callable:
            def wrapper(*args, **kwargs) -> NDArray:
                array = func(*args, **kwargs)
                # inf to inf_val
                return np.where(np.isposinf(array), inf_val, np.where(np.isneginf(array), -inf_val, array))
            return wrapper

        @staticmethod
        def forward(*args, **kwargs) -> NDArray:
            ...

        @staticmethod
        def backward(*args, **kwargs) -> NDArray:
            ...

        def main(self, *args, **kwargs) -> 'Matrix':
            ...

    class LoneElementWiseMethod(_BaseMethod):
        @staticmethod
        def forward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        @super().four_broadcast_e
        def backward(main: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        def main(self, main: 'Matrix') -> 'Matrix':
            # check tensor
            if not (isinstance(main, Matrix)):
                raise TypeError(
                    "Non-matrix call"  # todo: beef up error message
                )
            # calculate result
            result = Matrix(self.forward(main._array))
            result._tracker['org'] = [main, None]
            # track main
            Matrix._update_track(
                obj=main,
                drv=self.backward,
                rlt=[None, result]
            )
            # return result
            return result

    class ElementWiseMethod(_BaseMethod):
        @staticmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        @super().four_broadcast_e
        def backward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        def main(self, main: 'Matrix', other: any) -> 'Matrix':
            # check tensor
            if not (isinstance(main, Matrix)):
                raise TypeError(
                    "Non-matrix call"  # todo: beef up error message
                )
            # calculate result
            result = Matrix(self.forward(main._array, other._array))
            result._tracker['org'] = [main, None]
            # track main
            Matrix._update_track(
                obj=main,
                drv=self.backward,
                rlt=[None, result]
            )
            # return result
            return result

    class ScalarMethod(_BaseMethod):
        ...

    class CustomMethod(_BaseMethod):
        @staticmethod
        def forward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        @super().assert_four
        def backward(main: NDArray, other: NDArray) -> NDArray:
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        def main(self):
            ...

    @staticmethod
    def nabla(grad: 'Matrix', wrt: 'Matrix', *, binary: bool = True) -> 'Gradient':
        # check tensors
        if not isinstance(grad, Matrix):
            raise TypeError(
                "Attempted gradient calculation with grad object that was either"
                "not a Tensor or not a matrix subtype."
            )
        if not isinstance(wrt, Matrix):
            raise TypeError(
                "Attempted gradient calculation with wrt object that was either"
                "not a Tensor or not a matrix subtype."
            )

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
                origins = item._tracker['org']
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
                origins = item._tracker['org']
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

        def _derive(down: 'Matrix', up: 'Matrix') -> 'Gradient':
            # get relations
            strm_result = [rlt[1] for rlt in up._tracker['rlt']]
            strm_other = [rlt[0] for rlt in up._tracker['rlt']]
            # get operation
            drv_operator = up._tracker['drv'][strm_result.index(down)]
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

    class _MatMul(Custom):
        ...


########################################################################################################################


class Gradient(_Array):
    def __init__(self, obj: any, *, _signal_override: bool = False):
        if not _signal_override and not _Array._ikwiad:
            warn(f"", UserWarning)  # todo: beef up warning
        super().__init__(obj=obj, _dims=4, _type='gradient')
        # gradient subclass internals
        self._default_tracker = {'rlt': []}
        self._tracker = self._default_tracker

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
    def chain(down: 'Gradient', up: 'Gradient') -> 'Gradient':
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
