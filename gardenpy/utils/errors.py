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

        Args:
            grad (Tensor): First Tensor relating to the tracking error.
            wrt (Tensor): Second Tensor relating to the tracking error.
            message (str, optional): The error message.

        Note:
            A built-in error message reports common information about the tracking error if no message is given.
        """
        # error message
        if message is None:
            message = (
                f"No relation could be found between\n{grad}\nand\n{wrt}\n"
                "This might be due to:\n"
                "   No clear relation between the Tensors.\n"
                "   Accidental clearing of trackers.\n"
                "   Deletion of Tensors.\n"
                "   Accidental reference to the wrong Tensor."
            )
        super().__init__(str(message))
