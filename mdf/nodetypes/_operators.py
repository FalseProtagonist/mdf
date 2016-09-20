"""
This adds standard operators to nodes.
eg:

@evalnode
def A():
    return 1

@evalnode
def B():
    return 2

C = A + B  # C is a node whose result is A() + B()
"""
from ..nodes import MDFNode
import operator
import cython
import sys

__all__ = []


class Op(object):
    op = cython.declare(object)
    lhs = cython.declare(object)

    def __init__(self, op, lhs=None):
        self.op = op
        self.lhs = lhs

    def __get__(self, instance, owner=None):
        if instance is not None:
            return self.__class__(self.op, instance)
        return self.__class__(self.op, owner)

    def __call__(self, rhs=None):
        args = ()
        if rhs is not None:
            args = (rhs,)
        return self.lhs.applynode(func=self.op, args=args)


if sys.version_info[0] <= 2:
    for op in ("__add__", "__sub__", "__mul__", "__div__", "__neg__"):
        MDFNode._additional_attrs_[op] = Op(getattr(operator, op))
else:
    for op in ("__add__", "__sub__", "__mul__", "__truediv__", "__neg__"):
        MDFNode._additional_attrs_[op] = Op(getattr(operator, op))
