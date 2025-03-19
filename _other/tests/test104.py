from __future__ import annotations
from abc import ABC


class InnerParent(ABC):
    @staticmethod
    def did_modify(func: callable) -> callable:
        def wrapper(obj: OuterParent) -> None:
            func(obj)
            obj._modified = True

        return wrapper


class InnerChild(InnerParent):
    @staticmethod
    @InnerParent.did_modify
    def count_up(obj: OuterParent):
        obj._internal_int += 1


class OuterParent:
    def __init__(self, arg1):
        self._modified = False
        self._internal_int = 5

    @property
    def modified(self):
        return self._modified

    @property
    def internal_int(self):
        return self._internal_int
