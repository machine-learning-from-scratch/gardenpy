r"""
**GardenPy data checkers.**

Parameter storage & checkers.

Contains:
    - :class:`Params`
    - :class:`ParamChecker`
"""

from copy import deepcopy
from types import LambdaType
from warnings import warn


class Params:
    r"""
    **Parameter class.**

    Collection of parameters and their properties, including defaults, datatypes, value types, and conversion types.
    Used by :class:`ParamChecker`.
    """
    def __init__(
            self,
            default: dict[str, float | int | str | bool] | None = None,
            dtypes: dict[str, any] | None = None,
            vtypes: dict[str, callable] | None = None,
            ctypes: dict[str, callable] | None = None
    ):
        r"""
        **Parameter creation.**

        Parameters:
            default (dict | None): Default values.
            dtypes (dict | None): Accepted datatypes.
            vtypes (dict | None): Accepted value types.
            ctypes (dict | None): Conversion types.

        Raises:
            TypeError: Invalid parameter setting types.
        """
        self._default = Params._check_types(param=default, name='default', accepted_types=(float, int, str, bool))
        self._dtypes = Params._check_types(param=dtypes, name='dtypes', accepted_types=None)
        self._vtypes = Params._check_types(param=vtypes, name='vtypes', accepted_types=LambdaType)
        self._ctypes = Params._check_types(param=ctypes, name='ctypes', accepted_types=LambdaType)

    def __repr__(self) -> str:
        return (
            f"default: {self._default}\n"
            f"dtypes: {self._dtypes}\n"
            f"vtypes: {self._vtypes}\n"
            f"ctypes: {self._ctypes}"
        )

    @property
    def default(self) -> dict[str, float | int | str | bool] | None:
        r"""
        **Default values.**

        Parameter default values.

        Returns:
            dict[str, float | int | str | bool] | None: Default values.
        """
        return deepcopy(self._default)

    @property
    def dtypes(self) -> dict[str, any] | None:
        r"""
        **Accepted datatypes.**

        Parameter accepted datatypes.

        Returns:
            dict[str, any] | None: Accepted datatypes.
        """
        return deepcopy(self._dtypes)

    @property
    def vtypes(self) -> dict[str, callable] | None:
        r"""
        **Accepted value types.**

        Parameter accepted value ranges.

        Returns:
            dict[str, callable] | None: Conversion types.
        """
        return deepcopy(self._vtypes)

    @property
    def ctypes(self) -> dict[str, callable] | None:
        r"""
        **Conversion types.**

        Lambda parameter conversion call.

        Returns:
            dict[str, callable] | None: Conversion types.
        """
        return deepcopy(self._ctypes)

    @staticmethod
    def _check_types(param: any, name: str | None = None, accepted_types: any = None) -> dict | None:
        if param is not None and not isinstance(param, dict):
            # invalid type
            raise TypeError(
                f"Invalid type: Parameter items must be dictionaries or None. "
                f"Received type {type(param)} for {name}."
            )
        if param is None or accepted_types is None:
            return param
        if isinstance(param, dict) and not all([isinstance(itm, accepted_types) for itm in param.values()]):
            # accepted types
            raise TypeError(
                f"Invalid type: Attempted parameter creation with invalid types. "
                f"Specific parameter {name} accepts types {accepted_types}. "
                f"Received types {[type(itm) for itm in param.values()]}."
            )
        return param


class ParamChecker:
    r"""
    **Parameter checker for any parameters.**

    Uses specified default values, datatypes, value types, and conversion types to create a reusable parameter checker.
    Converts parameters to a final validated state after validating.
    """
    _none_params = Params(default=None, dtypes=None, vtypes=None, ctypes=None)

    def __init__(self, prefix: str = 'Parameters', parameters: Params = _none_params, *, ikwiad: bool = False):
        r"""
        **Set internal parameter settings.**

        Requires the use of :class:`Params` to set parameters.

        Parameters:
            prefix (str): Reference name in error messages.
            parameters (Params), default = _none_params: Parameter collection.
            ikwiad (bool), default = False: Turns off warning messages ("I know what I am doing" - ikwiad).
        """
        self._prefix = str(prefix)
        self._ikwiad = bool(ikwiad)
        self._params = self._validate_params(params=parameters)

    @property
    def parameters(self) -> Params:
        r"""
        **Internal parameter collection.**

        Returns:
            Params: Internal parameter collection.
        """
        return deepcopy(self._params)

    def _validate_dict(self, param: dict[str, any], name: str, is_call: bool = False, is_lambda: bool = False) -> None:
        # validate dictionary
        if not isinstance(param, dict):
            raise TypeError(
                f"Invalid type: Parameters component {name} in {self._prefix} must be a dictionary. "
                f"Received type {type(param)}"
            )
        for key, value in param.items():
            if is_lambda and not isinstance(value, LambdaType):
                raise TypeError(
                    f"Invalid type: Invalid lambda in {name} in {self._prefix}: {key}: {value}. "
                    f"Received type{type(value)}."
                )
            if is_call and callable(value):
                raise TypeError(
                    f"Invalid type: Unexpected callable for {name} in {self._prefix}: {key}: {value}. "
                    f"Expected a non-callable value. Received type {type(value)}."
                )
        return None

    def _validate_params(self, params: Params) -> Params:
        if not isinstance(params, Params):
            raise TypeError(f"Invalid type: Fed in parameters must be Params. Received type {type(params)}.")

        if params.default is None:
            # default none
            self._is_set = True
            return params

        # validate dicts
        self._validate_dict(params.default, 'default', is_call=True)
        self._validate_dict(params.dtypes, 'dtypes')
        self._validate_dict(params.vtypes, 'vtypes', is_lambda=True)
        self._validate_dict(params.ctypes, 'ctypes', is_lambda=True)

        # check for key matching
        keys = [params.default.keys(), params.dtypes.keys(), params.vtypes.keys(), params.ctypes.keys()]
        if not all(k == keys[0] for k in keys):
            raise ValueError(
                f"Invalid keys: Received parameters with mismatching keys for {self._prefix}. "
                f"Received keys {keys}."
            )

        return params

    def __call__(self, params: dict[str, any] | None = None, **kwargs: any) -> dict[str, any] | None:
        r"""
        **Checks parameters.**

        Uses the set internal settings to validate and modify parameters to their final state.

        Parameters:
            params (dict[str, any] | None): Parameters.
            **kwargs (any): Key-word parameters.

        Returns:
            dict[str, any] | NoneType: The checked parameters.
                Returns None of no parameters are taken.

        Raises:
            TypeError: Invalid parameter types.
            ValueError: Invalid parameter values.
        """
        # check for no parameters
        if self._params.default is None:
            return None

        # initialize as default
        final = deepcopy(self._params.default)

        if params is None and kwargs is None:
            # return default
            return final

        # set params
        if params and not isinstance(params, dict):
            raise TypeError(
                f"Invalid type: Parameters in {self._prefix} must be a dictionary. "
                f"Received type {type(params)}"
            )
        params = params if params else {}
        if kwargs:
            params.update(kwargs)

        for key, prm in params.items():
            if key not in self._params.default and self._ikwiad:
                # invalid key and warning
                warn(
                    f"Invalid parameter for {self._prefix}: {key}. "
                    f"Choose from: {[pos for pos in self._params.default]}",
                    UserWarning
                )
                continue
            elif key not in self._params.default:
                # invalid key
                continue

            # datatype check
            if not isinstance(prm, self._params.dtypes[key]):
                raise ValueError(
                    f"Invalid datatype: Invalid datatype for {self._prefix} {key}: {prm}. "
                    f"Choose from: {self._params.dtypes[key]}."
                )
            if not self._params.vtypes[key](prm):
                raise ValueError(
                    f"Invalid value: Invalid value for {self._prefix} {key}: {prm}. "
                    f"Failed conditional: {self._params.vtypes[key]}."
                )
            # set parameter
            final[key] = self._params.ctypes[key](prm)

        # return parameters
        return final
