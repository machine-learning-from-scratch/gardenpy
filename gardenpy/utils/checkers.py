r"""
**Data checkers.**

Contains:
    - :class:`Params`
    - :class:`ParamChecker`
"""
# todo: check logic (might need deepcopy?)

from types import LambdaType
from warnings import warn


class Params:
    r"""
    **General parameter setup.**

    General items required for use in :class:`ParamChecker`, organized in a class.
    """
    def __init__(
            self,
            default: dict[str, int | float | str | bool] | None = None,
            dtypes: dict[str, any] | None = None,
            vtypes: dict[str, callable] | None = None,
            ctypes: dict[str, callable] | None = None
    ):
        r"""
        **Set parameter settings for a class instance**

        Args:
            default (dict | None), default = None: Default values.
            dtypes (dict | None), default = None: Accepted datatypes.
            vtypes (dict | None), default = None: Accepted value types.
            ctypes (dict | None), default = None: Conversion types.

        Raises:
            TypeError: Invalid parameter setting types.
        """
        self._default = self._check_default(default)
        self._dtypes = self._check_dtypes(dtypes)
        self._vtypes = self._check_vtypes(vtypes)
        self._ctypes = self._check_ctypes(ctypes)

    def __repr__(self):
        return (
            f"default: {self._default}\n"
            f"dtypes: {self._dtypes}\n"
            f"vtypes: {self._vtypes}\n"
            f"ctypes: {self._ctypes}"
        )

    @staticmethod
    def _check_type(itm):
        if itm is not None and not isinstance(itm, dict):
            # check itm
            raise TypeError("Each parameter item must be None or a dictionary")
        # return itm
        return itm

    @staticmethod
    def _check_default(default: dict[str, any] | None) -> dict[str, any] | None:
        # check dict
        Params._check_type(default)
        # check default
        if (
                isinstance(default, dict) and
                not all([isinstance(itm, (int, float, str, bool)) for itm in default.values()])
        ):
            raise TypeError("Attempted creating default values with invalid types.")
        # return default
        return default

    @staticmethod
    def _check_dtypes(dtypes: dict[str, any] | None) -> dict[str, any] | None:
        # check dict
        Params._check_type(dtypes)
        # check dtypes
        if (
                isinstance(dtypes, dict) and
                not all([itm is not callable for itm in dtypes.values()])
        ):
            raise TypeError("Attempted creating datatypes with invalid types.")
        # return dtypes
        return dtypes

    @staticmethod
    def _check_vtypes(vtypes: dict[str, any] | None) -> dict[str, any] | None:
        # check dict
        Params._check_type(vtypes)
        # check vtypes
        if (
                isinstance(vtypes, dict) and
                not all([isinstance(itm, LambdaType) for itm in vtypes.values()])
        ):
            raise TypeError("Attempted creating value types with invalid types.")
        # return vtypes
        return vtypes

    @staticmethod
    def _check_ctypes(ctypes: dict[str, any] | None) -> dict[str, any] | None:
        # check dict
        Params._check_type(ctypes)
        # check ctypes
        if (
                isinstance(ctypes, dict) and
                not all([isinstance(itm, LambdaType) for itm in ctypes.values()])
        ):
            raise TypeError("Attempted creating conversion types with invalid types.")
        # return ctypes
        return ctypes

    @property
    def default(self) -> dict[str, int | float | str | bool] | None:
        r"""
        **Default values.**

        Default values for each parameter.

        Returns:
            dict: Default values.
        """
        return self._default

    @property
    def dtypes(self) -> dict[str, any] | None:
        r"""
        **Accepted data types.**

        Specific datatypes accepted for each parameter.

        Returns:
            dict: Accepted data types.
        """
        return self._dtypes

    @property
    def vtypes(self) -> dict[str, callable] | None:
        r"""
        **Accepted value types.**

        Specific values accepted for each parameter.

        Returns:
            dict: Conversion types.
        """
        return self._vtypes

    @property
    def ctypes(self) -> dict[str, callable] | None:
        r"""
        **Conversion types.**

        Final converted returned parameter.

        Returns:
            dict: Conversion types.
        """
        return self._ctypes


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

        Args:
            prefix (str): Parameter name used in error messages.
            parameters (Params), default = _none_params: Parameter settings.
            ikwiad (bool), default = False: Turns off all warning messages ("I know what I am doing" - ikwiad).
        """
        self._prefix = str(prefix)
        self._ikwiad = bool(ikwiad)
        self._params = self._validate_params(params=parameters)

    @property
    def parameters(self) -> Params:
        r"""
        **Returns internal parameter settings.**

        Returns:
            Params: Internal parameter settings.
        """
        return self._params

    def _validate_dict(
            self,
            param_dict: dict[str, any],
            name: str,
            check_callable: bool = False,
            check_lambda: bool = False
    ) -> None:
        # validate dictionary
        if not isinstance(param_dict, dict):
            raise TypeError(f"'{name}' in {self._prefix} must be a dict")
        for key, value in param_dict.items():
            if check_lambda and not isinstance(value, LambdaType):
                raise TypeError(f"Invalid lambda in '{name}' in {self._prefix}: {key} -> {value}")
            if check_callable and callable(value):
                raise TypeError(f"Callable not allowed in '{name}' in {self._prefix}: {key} -> {value}")
        return None

    def _validate_params(self, params: Params) -> Params:
        if not isinstance(params, Params):
            raise TypeError("'params' must be Params")

        if params.default is None:
            # default none
            self._is_set = True
            return params

        # validate dicts
        self._validate_dict(params.default, 'default', check_callable=True)
        self._validate_dict(params.dtypes, 'dtypes')
        self._validate_dict(params.vtypes, 'vtypes', check_lambda=True)
        self._validate_dict(params.ctypes, 'ctypes', check_lambda=True)

        # check for key matching
        keys = [params.default.keys(), params.dtypes.keys(), params.vtypes.keys(), params.ctypes.keys()]
        if not all(k == keys[0] for k in keys):
            raise ValueError(f"Keys don't match for '{self._prefix}'")

        return params

    def __call__(self, params: dict | None = None, **kwargs: any) -> dict | None:
        r"""
        **Checks parameters.**

        Uses the set internal settings to validate and modify parameters to their final state.

        Args:
            params (dict, optional): Parameters to be checked.
            **kwargs: Key-word args of the parameters to be checked.

        Returns:
            dict | NoneType: The checked parameters.
                Returns None of no parameters are taken.

        Raises:
            TypeError: If any parameters were of the wrong type.
            ValueError: If invalid values were passed for any of the parameters.
        """
        # check for no parameters
        if self._params.default is None:
            return None

        # initialize as default
        final = self._params.default.copy()

        if params is None and kwargs is None:
            # return default
            return final

        # set params
        if params and not isinstance(params, dict):
            raise TypeError(f"'params' in {self._prefix} must be a dictionary")
        params = params if params else {}
        if kwargs:
            params.update(kwargs)

        for key, prm in params.items():
            if key not in self._params.default and self._ikwiad:
                # invalid key and warning
                warn(
                    f"\nInvalid parameter for '{self._prefix}': '{key}'\n"
                    f"Choose from: '{[pos for pos in self._params.default]}'",
                    UserWarning
                )
                continue
            elif key not in self._params.default:
                # invalid key
                continue

            # datatype check
            if not isinstance(prm, self._params.dtypes[key]):
                raise ValueError(
                    f"Invalid datatype for '{self._prefix}' '{key}': '{prm}'\n"
                    f"Choose from: {self._params.dtypes[key]}"
                )
            if not self._params.vtypes[key](prm):
                raise ValueError(
                    f"Invalid value for '{self._prefix}' '{key}': '{prm}'\n"
                    f"Failed conditional: {self._params.vtypes[key]}"
                )
            # set parameter
            final[key] = self._params.ctypes[key](prm)

        # return parameters
        return final
