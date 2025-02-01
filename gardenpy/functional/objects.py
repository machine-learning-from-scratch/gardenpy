r"""
**GardenPy objects.**

Contains:
    - :class:`Tensor`
"""

from warnings import warn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from ..utils.errors import TrackingError


class Tensor:
    r"""
    **GardenPy's Tensor, for automatic gradient calculation and tracking.**

    Tensors are arrays that track objects and operations related to itself, creating a computation graph.
    Tensors utilize this computational graph in tape-based autograd to calculate gradients.
    See :func:`Tensor.nabla` and :func:`Tensor.chain`.

    Note:
        Tensors don't automatically reset.
        Internally, tracks will accumulate references and the instance list will clog with unused Tensors.
        This will cause issues with the search tree for autograd and memory leaks, respectively.
        It's recommended to clear unused Tensors frequently using :func:`Tensor.zero_grad`.

    Note:
        Autograd methods can be created by inheriting a Tensor method class.
        These will automatically modify internals, which shouldn't be edited externally.

    Note:
        Internal autograd function calls are found within this class.
        However, some commonly called functions have shorthand function calls.
        See :module:`gardenpy.functional.operators`
    """
    # instance reference
    _instances: List[Union['Tensor', None]] = []
    # class ikwiad reference
    _ikwiad: bool = False

    def __init__(self, obj: any, *, _gradient_override: bool = False):
        r"""
        **Creates a Tensor.**

        Creates a matrix-type Tensor with a blank tracker.
        Adds Tensor to internal instances at the first open spot.
        Currently only supports two-dimensional arrays that consist of real numbers.

        Args:
            obj (any): Object to be turned into a Tensor.
                Object must contain only numbers and be two-dimensional.

        Raises:
            TypeError: If the object isn't a two-dimensional NumPy array consisting of only numbers.

        Note:
            Objects meant for Tensors will undergo NumPy array conversion.
        """
        # check object
        obj = np.array(obj)
        if (not np.issubdtype(obj.dtype, np.number) or len(obj.shape) != 2) and not _gradient_override:
            # NB: Currently, Tensors can only be two-dimensional.
            # Due to the existence of three-dimensional gradients, this lock can be overwritten, but only internally.
            raise TypeError("Attempted creation with object that wasn't two-dimensional with only real numbers.")

        # set tensor internals
        self._id: Union[int, None] = None
        self._tensor: Union[np.ndarray, None] = obj
        self._type: str = 'mat'
        self._tracker: Dict[str, Optional[list, Tensor, str]] = {'opr': [], 'drv': [], 'chn': [], 'rlt': [], 'org': []}
        self._tags: List[str] = []

        # update instances and id
        Tensor._add_instance(self)

    def __repr__(self):
        self._is_valid_tensor(itm=self)
        return str(self._tensor)

    ####################################################################################################################

    @property
    def id(self) -> Union[str, None]:
        r"""
        **Tensor ID.**

        ID correlating to its position within the class's instance list.

        Returns:
            str | NoneType: Current Tensor ID.
                Returns None if the function is used on a deleted Tensor.

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        if self._is_valid_tensor(itm=self):
            return hex(self._id)
        else:
            return None

    @property
    def type(self) -> str:
        r"""
        **Tensor's type.**

        Returns:
            str: Tensor type.
        """
        return self._type

    @property
    def tracker(self) -> Union[Dict[str, Union[str, None, List[str]]], None]:
        r"""
        **Tensor's internal tracker.**

        Returns:
            dict | NoneType: Dictionary of a Tensor's tracker and ID.

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        # do error messages
        self._is_valid_tensor(itm=self)
        # turn on ikwiad
        user_ikwiad = Tensor._ikwiad
        Tensor._ikwiad = True
        # get tracker reference
        alt_tracker = self._tracker.copy()

        if not self._is_valid_tensor(itm=self):
            return None
        elif self._type == 'mat':
            # matrix tracker
            alt_tracker['rlt'] = [
                [itm[0].id if isinstance(itm[0], Tensor) else hex(id(itm[0])),
                 itm[1].id if isinstance(itm[1], Tensor) else hex(id(itm[1]))]
                for itm in alt_tracker['rlt']
            ]
            alt_tracker['org'] = [itm.id if isinstance(itm, Tensor) else hex(id(itm)) for itm in alt_tracker['org']]
        elif self._type == 'grad':
            # gradient tracker
            alt_tracker['rlt'] = [itm.id if isinstance(itm, Tensor) else hex(id(itm)) for itm in alt_tracker['rlt']]
            alt_tracker['org'] = alt_tracker['org'].id

        # return to user ikwiad
        Tensor._ikwiad = user_ikwiad
        # return tracker
        return {'itm': self.id, 'tags': self._tags, **alt_tracker}

    @property
    def array(self) -> Union[np.ndarray, None]:
        r"""
        **Tensor's internal NumPy array.**

        Returns:
            np.ndarray | NoneType: Tensor's internal NumPy array.

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        self._is_valid_tensor(itm=self)
        return self._tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        r"""
        **Tensor's shape.**

        Returns:
            tuple: Tensor's shape.

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        self._is_valid_tensor(itm=self)
        return self._tensor.shape

    @property
    def tags(self) -> List[str]:
        r"""
        **Tensor's tags.**""

        Returns:
            list: Tensor's tags.
        """
        return self._tags

    @tags.setter
    def tags(self, itm: str) -> None:
        r"""
        **Add an item to a Tensor's tags.**

        Args:
            itm (str): Added tag.
        """
        self._tags.append(str(itm))

    ####################################################################################################################

    def instance_grad_reset(self) -> None:
        r"""
        **Reset a Tensor's internal tracker.**

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        # reset tensor tracker
        if self._is_valid_tensor(itm=self):
            self._tracker = {'opr': [], 'drv': [], 'chn': [], 'rlt': [], 'org': []}
        return None

    def instance_reset(self) -> None:
        r"""
        **Reset a Tensor.**

        Removes all identifying features of a Tensor, including its value, and opens its spot within the instance list.

        Raises:
            UserWarning: If the function is used on a deleted Tensor.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        # reset tensor
        if self._is_valid_tensor(itm=self):
            Tensor._instances[self._id] = None
            self._id = None
            self._tensor = None
            self._type = 'del'
            self._tracker = None
            self._tags.append('deleted')
        return None

    def matrix(self) -> Union['Tensor', None]:
        r"""
        **Creates a new matrix-type Tensor if possible.**

        Raises:
            UserWarning: If matrix conversion was attempted on an invalid Tensor type.
                Turned off by toggling ikwiad.
                See :func:`Tensor.ikwiad`.
        """
        if self._type != 'grad':
            if not self._ikwiad:
                warn("Attempted matrix conversion on a Tensor that wasn't a gradient.", UserWarning)
            return None
        return Tensor(self._tensor)

    ####################################################################################################################

    @classmethod
    def instances(cls) -> List[str]:
        r"""
        **Tensor instances.**

        Returns:
            list: All Tensor instances referenced by ID.
        """
        # return instances
        return [instance.id if instance is not None else None for instance in cls._instances]

    @classmethod
    def instance(cls, idx: Union[str, int]) -> 'Tensor':
        r"""
        **Gets Tensor from an index reference.**

        Args:
            idx (str | int): Index reference to a Tensor in the instances list.

        Returns:
            Tensor: Referenced Tensor.

        Raises:
            ValueError: If there was an invalid Tensor reference.
            TypeError: If there was an invalid reference type or deleted Tensor reference.
        """
        # get reference
        itm, _ = cls._get_tensor_reference(itm=idx)
        # return reference
        return itm

    @classmethod
    def reset(cls, *args: Optional[Union['Tensor', str, int]]) -> None:
        r"""
        **Resets Tensor instances.**

        Args:
            *args (Tensor | str | None, optional): Tensors to save.
        """
        # find non-saved instances
        args = list(args)
        for i, arg in enumerate(args):
            _, args[i] = cls._get_tensor_reference(arg)
        non_saved = [itm for i, itm in enumerate(cls._instances) if i not in args and itm is not None]
        # reset non-saved instance
        for instance in non_saved:
            instance.instance_reset()
        return None

    @classmethod
    def grad_reset(cls, *args: Optional[Union['Tensor', str, int]]) -> None:
        r"""
        **Resets Tensor trackers and deletes gradients.**

        Args:
            *args (Tensor | str | None, optional): Tensors to retain trackers.
        """
        # turn on ikwiad
        user_ikwiad = Tensor._ikwiad
        Tensor._ikwiad = True

        # find non-saved instances
        args = list(args)
        for i, arg in enumerate(args):
            _, args[i] = cls._get_tensor_reference(arg)
        non_saved = [itm for i, itm in enumerate(cls._instances) if i not in args and itm is not None]

        for instance in non_saved:
            if instance is None:
                continue
            if instance.type == 'grad':
                # reset gradients
                instance.instance_reset()
            # reset tensor tracker
            instance.instance_grad_reset()
        # return to user ikwiad
        Tensor._ikwiad = user_ikwiad
        return None

    @classmethod
    def replace(cls, replaced: Union['Tensor', str, int], replacer: Union['Tensor', str, int]) -> None:
        r"""
        **Replaces a Tensor with another Tensor in the reference list

        Args:
            replaced (Tensor, str, int): Replaced Tensor.
            replacer (Tensor, str, int): Replacer Tensor.

        Note:
            Deletes the replaced Tensor using :func:`Tensor.instance_reset`.
        """
        # get tensors and ids
        replaced, replaced_id = cls._get_tensor_reference(itm=replaced)
        replacer, replacer_id = cls._get_tensor_reference(itm=replacer)

        # reset replaced tensor
        replaced.instance_reset()
        # move replacer tensor
        replacer._id = replaced_id
        cls._instances[replaced_id] = replacer
        cls._instances[replacer_id] = None
        return None

    @classmethod
    def zero_grad(cls, *args: Union['Tensor', str, int]) -> None:
        r"""
        **Resets Tensor instances and trackers.**

        Args:
            *args (Tensor | str | None, optional): Tensors to save.
        """
        # reset instances
        cls.reset(*args)
        # reset gradient tracking
        cls.grad_reset()

    @classmethod
    def ikwiad(cls, ikwiad: bool) -> None:
        r"""
        **Turns off warning messages ("I know what I am doing" - ikwiad).**

        Args:
            ikwiad (bool): ikwiad state.
        """
        cls._ikwiad = bool(ikwiad)

    ####################################################################################################################

    @staticmethod
    def _is_valid_tensor(itm: 'Tensor') -> bool:
        if itm._type != 'del':
            # non-deleted Tensor
            return True
        else:
            # deleted Tensor
            if not Tensor._ikwiad:
                warn("Attempted deleted Tensor reference.", UserWarning)
            return False

    @staticmethod
    def _update_track(obj: 'Tensor', opr: any, drv: any, chn: any, rlt: any) -> None:
        # tracker update
        obj._tracker['opr'].append(opr)
        obj._tracker['drv'].append(drv)
        obj._tracker['chn'].append(chn)
        obj._tracker['rlt'].append(rlt)
        return None

    @classmethod
    def _get_tensor_reference(cls, itm: Union['Tensor', str, int]) -> Tuple['Tensor', int]:
        # turn on ikwiad
        user_ikwiad = Tensor._ikwiad
        Tensor._ikwiad = True
        # attempt hex conversion
        try:
            itm = int(itm, 16)
        except TypeError:
            pass
        except ValueError:
            pass

        if isinstance(itm, Tensor):
            # use object reference
            itm_id = itm._id
        elif isinstance(itm, int):
            # check index reference
            if len(cls._instances) <= itm:
                raise ValueError(
                    "Attempted reference outside Tensor instance list."
                    f"Currently, instance list only contains {len(cls._instances)} items."
                    f"A reference has been made to the {itm} index."
                )
            # use index reference
            itm_id = itm
            itm = cls._instances[itm_id]
        else:
            # invalid reference
            raise TypeError("Attempted Tensor reference with an invalid type.")
        if not cls._is_valid_tensor(itm=itm):
            # invalid tensor
            raise TypeError("Attempted reference to a deleted Tensor.")

        # return to user ikwiad
        Tensor._ikwiad = user_ikwiad
        # return items
        return itm, itm_id

    @classmethod
    def _add_instance(cls, itm: 'Tensor') -> None:
        try:
            open_id = cls._instances.index(None)
            itm._id = open_id
            cls._instances[open_id] = itm
        except ValueError:
            open_id = len(cls._instances)
            cls._instances.append(itm)
            itm._id = open_id
        return None

    ####################################################################################################################

    class _TensorMethod:
        r"""
        **Base class for Tensor methods.**
        """
        def __init__(self, prefix):
            self._prefix = str(prefix)

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            r"""
            **Chain rule method.**

            Args:
                down (np.ndarray): Downstream gradient.
                up (np.ndarray): Upstream gradient.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

    class LoneTensorMethod(_TensorMethod):
        r"""
        **Paired Tensor method structure.**

        Used for autograd methods that involve one Tensor.
        Autograd can be called with this function without further modification to Tensors once all methods are defined.
        """
        @staticmethod
        def forward(main: np.ndarray) -> np.ndarray:
            r"""
            **Forward method.**

            Args:
                main (np.ndarray): Main array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        def backward(main: np.ndarray) -> np.ndarray:
            r"""
            **Backward method.**

            Args:
                main (np.ndarray): Main array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        def main(self, main: 'Tensor') -> 'Tensor':
            r"""
            **Main function call.**

            Args:
                main (Tensor): Non-deleted input Tensor.

            Returns:
                Tensor: Output Tensor.
            """
            # check tensor
            if not (isinstance(main, Tensor)) or main._type != 'mat':
                if not (isinstance(main, Tensor)) or main._type != 'mat':
                    raise TypeError(
                        "Attempted call without main object being the right kind of Tensor.\n"
                        "To call this function, try:\n"
                        "   Converting the object into a Tensor if it isn't already.\n"
                        "   Using the .matrix() function to convert a matrix.\n"
                        "   Creating a blank Tensor using the internal array."
                    )
            # calculate result
            result = Tensor(self.forward(main._tensor))
            result._tracker['org'] = [main, None]
            # track main
            Tensor._update_track(
                obj=main,
                opr=f"{self._prefix}",
                drv=self.backward,
                chn=self.chain,
                rlt=[None, result]
            )
            # return result
            return result

    class PairedTensorMethod(_TensorMethod):
        r"""
        **Paired Tensor method structure.**

        Used for autograd methods that involve two Tensors.
        Autograd can be called with this function without further modification to Tensors once all methods are defined.
        """
        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            r"""
            **Forward method.**

            Args:
                main (np.ndarray): Main array.
                other (np.ndarray): Other array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            r"""
            **Backward method.**

            Args:
                main (np.ndarray): Main array.
                other (np.ndarray): Other array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            r"""
            **Secondary backward method.**

            Args:
                main (np.ndarray): Main array.
                other (np.ndarray): Other array.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            r"""
            **Secondary chain rule method.**

            Args:
                down (np.ndarray): Downstream gradient.
                up (np.ndarray): Upstream gradient.

            Returns:
                np.ndarray: Result.
            """
            raise NotImplementedError(
                "Attempted function call without redefinition in subclass.\n"
                "Either define this call, or avoid referencing it."
            )

        def main(self, main: 'Tensor', other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
            r"""
            **Main function call.**

            Args:
                main (Tensor): Non-deleted input Tensor.
                other (Tensor): Second non-deleted input Tensor.

            Returns:
                Tensor: Output Tensor.
            """
            # check tensor
            if not (isinstance(main, Tensor)) or main._type != 'mat':
                raise TypeError(
                    "Attempted call without main object being the right kind of Tensor.\n"
                    "To call this function, try:\n"
                    "   Converting the object into a Tensor if it isn't already.\n"
                    "   Using the .matrix() function to convert a matrix.\n"
                    "   Creating a blank Tensor using the internal array."
                )

            # set array
            if isinstance(other, Tensor):
                arr = other._tensor
            else:
                arr = other

            # calculate result
            result = Tensor(self.forward(main._tensor, arr))
            result._tracker['org'] = [main, other]
            # track main
            Tensor._update_track(
                obj=main,
                opr=f"{self._prefix}",
                drv=self.backward,
                chn=self.chain,
                rlt=[other, result]
            )
            if isinstance(other, Tensor):
                # track _other
                Tensor._update_track(
                    obj=other,
                    opr=f"{self._prefix}_o",
                    drv=self.backward_o,
                    chn=self.chain_o,
                    rlt=[main, result]
                )
            # return result
            return result

    ####################################################################################################################

    @staticmethod
    def nabla(grad: 'Tensor', wrt: 'Tensor', *, binary: bool = True) -> 'Tensor':
        r"""
        **Calculates the gradient of two Tensors.**

        Using the computational graph, a search tree attempts to relate the two Tensors.
        Backward and chain rule methods are called to calculate the gradient if the search tree establishes a
        relationship.

        Args:
            grad (Tensor): Object of the matrix type to calculate the gradient.
            wrt (Tensor): Object of the matrix gradient is calculated with respect to.
            binary (Tensor): Whether to utilize a binary search tree or just a search tree.
                If binary is on, only the first relationship is found.
                Otherwise, all relationships are found and the gradients are added together.

        Returns:
            Tensor: Calculated gradient with set internals.

        Raises:
            TypeError: If the Tensors were of the wrong type.
            TrackingError: If no relationship could be found between the Tensors.

        Note:
            This function automatically calls :func:`Tensor.chain` to chain rule gradients.
            However, it doesn't attempt to find already calculated gradients.
            Computational speed can be increased by using pre-calculated gradients and manually calling
            :func:`Tensor.chain`.

        Note:
            The gradient's tracker won't reference the full path if binary is toggled off.
            This doesn't have an impact on computation and is reflected in the gradient's tags as 'linear override'.
        """
        # check tensors
        if not isinstance(grad, Tensor) or grad._type != 'mat':
            raise TypeError(
                "Attempted gradient calculation with grad object that was either"
                "not a Tensor or not a matrix subtype."
            )
        if not isinstance(wrt, Tensor) or wrt._type != 'mat':
            raise TypeError(
                "Attempted gradient calculation with wrt object that was either"
                "not a Tensor or not a matrix subtype."
            )

        # set gradient relation
        if binary:
            relation = None
        else:
            relation = []

        def _relate(item, target, trace=None):
            nonlocal relation
            if trace is None:
                # reset trace
                trace = []
            # NB: This only gets origins if the item is a matrix.
            # Tracing gradients through gradients is possible, but requires a lot of modification and significantly
            # increases computational time, even if it's never used.
            if binary and relation is None and isinstance(item, Tensor) and item._type == 'mat':
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
            elif not binary and isinstance(item, Tensor) and item._type == 'mat':
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
            raise TrackingError(grad=grad, wrt=wrt)

        def _derive(down: 'Tensor', up: 'Tensor', _identity: bool = False) -> 'Tensor':
            # get relations
            strm_result = [rlt[1] for rlt in up._tracker['rlt']]
            strm_other = [rlt[0] for rlt in up._tracker['rlt']]
            # get operation
            operator = up._tracker['opr'][strm_result.index(down)]
            drv_operator = up._tracker['drv'][strm_result.index(down)]
            other = strm_other[strm_result.index(down)]

            if isinstance(other, Tensor):
                # get value
                other = other._tensor
            # calculate local gradient
            try:
                # pair derivative method
                res = drv_operator(up._tensor, other)
            except TypeError:
                # lone derivative method
                res = drv_operator(up._tensor)

            # tensor conversion
            res = Tensor(obj=res, _gradient_override=True)

            # set local grad internals
            res._type = 'grad'
            res._tracker['opr'].append(f'd_{operator}')
            res._tracker['drv'].append(drv_operator)
            res._tracker['chn'].append(down._tracker['chn'])
            res._tracker['rlt'] += [down, up]
            res._tracker['org'] = down
            # return local gradient
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
                result = Tensor.chain(down=_derive(down=relation[-2], up=relation[-1]), up=result)
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
                    op_res = Tensor.chain(down=_derive(down=itm[-2], up=itm[-1]), up=op_res)
                    del itm[-1]
                grads.append(op_res._tensor)
                grad_itms.append(op_res)
                track = op_res._tracker
            result = 0
            for grad, itm in zip(grads, grad_itms):
                result += grad
                itm.instance_reset()
            result = Tensor(obj=result, _gradient_override=True)
            result._type = 'grad'
            result._tracker = track

        # return final gradient
        if linear_override:
            result._tags.append('linear override')
        return result

    @staticmethod
    def chain(down: 'Tensor', up: 'Tensor') -> 'Tensor':
        r"""
        **Chain-rule gradients.**

        Checks if there is an immediate link between the downstream and upstream gradients.
        If there is a link, the chain-rule operation is called to link the gradients.

        Args:
            down (Tensor): Downstream gradient.
            up (Tensor): Upstream gradient.

        Returns:
            Tensor: Chain ruled gradient with set internals.

        Raises:
            TrackingError: If no relationship could be found between the Tensors.

        Note:
            :func:`Tensor.nabla`automatically chain rules gradients.
            However, it doesn't attempt to find already calculated gradients.
            Computational speed can be increased by using pre-calculated gradients and manually calling this function.
        """
        if not isinstance(down, Tensor) or down._type != 'grad':
            raise TypeError(
                "Attempted chain-rule calculation with down object that was either"
                "not a Tensor or not a gradient subtype."
            )
        if not isinstance(up, Tensor) or up._type != 'grad':
            raise TypeError(
                "Attempted chain-rule calculation with up object that was either"
                "not a Tensor or not a gradient subtype."
            )

        # check relation
        down_relation = down._tracker['rlt'][-1]
        up_relation = up._tracker['org']
        if down_relation == up_relation:
            # chain gradients
            result = Tensor(obj=up._tracker['chn'][0][0](down._tensor, up._tensor), _gradient_override=True)
            # set gradient internals
            result._type = 'grad'
            result._tracker['rlt'] = down._tracker['rlt'] + up._tracker['rlt'][1:]
            result._tracker['opr'] = down._tracker['opr'] + up._tracker['opr']
            result._tracker['drv'] = down._tracker['drv'] + up._tracker['drv']
            result._tracker['chn'] = down._tracker['chn'] + up._tracker['chn']
            result._tracker['org'] = down._tracker['org']
            # return final gradient
            return result
        else:
            # no relation
            raise TrackingError(grad=down, wrt=up)

    ####################################################################################################################

    class _MatMul(PairedTensorMethod):
        # matrix multiplication
        # todo: correct implementation
        def __init__(self):
            super().__init__(prefix='matmul')

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main @ other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return np.array([np.sum(other.T, axis=0)])

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return main.T * (0.0 * other + 1.0)

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

    class _Pow(PairedTensorMethod):
        # hadamard power
        def __init__(self):
            super().__init__(prefix='pow')

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main ** other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return other * (main ** (other - 1))

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return np.log(main) * (main ** other)

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

    class _Mul(PairedTensorMethod):
        # hadamard multiplication
        def __init__(self):
            super().__init__(prefix='mul')

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main * other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return other * (main * 0.0 + 1.0)

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return main * (other * 0.0 + 1.0)

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

    class _TrueDiv(PairedTensorMethod):
        # hadamard division
        def __init__(self):
            super().__init__(prefix='truediv')

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main / other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return other * (main * 0.0 + 1.0)

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return (other * 0.0 + 1.0) / main

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

    class _Add(PairedTensorMethod):
        # addition
        def __init__(self):
            super().__init__(prefix='add')

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main + other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return (main * 0.0 + 1.0) * np.eye(main.shape[1])

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return (other * 0.0 + 1.0) * np.eye(other.shape[1])

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

    class _Sub(PairedTensorMethod):
        # subtraction
        def __init__(self):
            super().__init__(prefix='sub')

        @staticmethod
        def forward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main - other

        @staticmethod
        def backward(main: np.ndarray, other: np.ndarray) -> np.ndarray:
            return main * 0.0 + 1.0

        @staticmethod
        def backward_o(other: np.ndarray, main: np.ndarray) -> np.ndarray:
            return other * 0.0 - 1.0

        @staticmethod
        def chain(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

        @staticmethod
        def chain_o(down: np.ndarray, up: np.ndarray) -> np.ndarray:
            try:
                return down @ up
            except ValueError:
                return down * up

    # internal instance
    _matmul = _MatMul()
    _pow = _Pow()
    _mul = _Mul()
    _truediv = _TrueDiv()
    _add = _Add()
    _sub = _Sub()

    def __matmul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""**Matrix multiplication.**"""
        return self._matmul.main(self, other)

    def __pow__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""**Hadamard power.**"""
        return self._pow.main(self, other)

    def __mul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""**Hadamard multiplication.**"""
        return self._mul.main(self, other)

    def __truediv__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""**Hadamard division.**"""
        return self._truediv.main(self, other)

    def __add__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""**Addition.**"""
        return self._add.main(self, other)

    def __sub__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        r"""**Subtraction.**"""
        return self._sub.main(self, other)
