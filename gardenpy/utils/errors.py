r"""
**GardenPy built-in errors.**

Custom errors.

Contains:
    - :class:`TrackingError`
    - :class:`MissingMethodError`
"""


class TrackingError(Exception):
    r"""**Unsuccessful automatic differentiation tracking.**"""
    def __init__(self, grad: any, wrt: any, *, message: str | None = None):
        r"""
        **Error references.**

        Parameters:
            grad (any): First item relating to the tracking error.
            wrt (any): Second item relating to the tracking error.
            message (str | None): Error message.

        Note:
            A built-in error message reports common information about the tracking error if no message is given.
        """
        # error message
        if message is None:
            message = f"No relationship could be found between\n{grad}\nand\n{wrt}"
        super().__init__(str(message))


class MissingMethodError(Exception):
    r"""**Missing method.**"""
    # TODO: Once implemented, most likely in DNN, add base internal logic.
    pass
