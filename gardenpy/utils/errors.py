r"""
**Built-in errors.**

Contains:
    - :class:`MissingMethodError`
    - :class:`TrackingError`
"""


class MissingMethodError(Exception):
    r"""**Missing method.**"""
    pass


class TrackingError(Exception):
    r"""**Unsuccessful autodiff tracking.**"""
    def __init__(self, grad, wrt, *, message: str | None = None):
        r"""
        **Error references.**

        Parameters:
            grad (Tensor): First Tensor relating to the tracking error.
            wrt (Tensor): Second Tensor relating to the tracking error.
            message (str, optional): The error message.

        Note:
            A built-in error message reports common information about the tracking error if no message is given.
        """
        # error message
        if message is None:
            message = (
                f"No relation could be found between {grad} and {wrt}"
            )
        super().__init__(str(message))
