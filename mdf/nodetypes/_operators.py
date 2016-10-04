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
from ._nodetypes import nodetype
from ._applynode import MDFApplyNode, _applynode
from ..nodes import MDFNode
import operator
import cython
import sys

# PURE PYTHON START
from ._nodetypes import dict_iteritems
# PURE PYTHON END

__all__ = []


class MDFBinOpNode(MDFApplyNode):
    nodetype_args = ["value_node", "func", "args", "kwargs"]
    nodetype_node_kwargs = ["value_node"]

    def _cn_get_all_values(self, ctx, node_state):
        # if the value node and all args/kwargs nodes have data then we can provide a full set of data.
        value_node = self.value_node
        if self.value_node is None:
            return None

        value_node_df = ctx._get_all_values(value_node)
        if value_node_df is None:
            return

        node_type_kwargs = self.node_type_kwargs
        func = node_type_kwargs["func"]
        args = node_type_kwargs.get("args", ())
        kwargs = node_type_kwargs.get("kwargs", {})

        new_args = []
        for arg in args:
            if isinstance(arg, MDFNode):
                arg = ctx._get_all_values(arg)
                if arg is None:
                    return
            new_args.append(arg)

        new_kwargs = {}
        for key, value in dict_iteritems(kwargs):
            if isinstance(value, MDFNode):
                value = ctx._get_all_values(value)
                if value is None:
                    return
            new_kwargs[key] = value

        # we know that this nodetype is only constructed with binary operators,
        # and so the same function can be used on the full dataframes as on the
        # individual rows.
        return func(value_node_df, *new_args, **new_kwargs)


# decorators don't work on cythoned types
binopnode = nodetype(cls=MDFBinOpNode, node_method="_binopnode")(_applynode)



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
        return self.lhs._binopnode(func=self.op, args=args)


binops = [
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__" if sys.version_info[0] > 2 else "__div__",
    "__neg__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__eq__",
    "__ne__"
]

for op in binops:
    MDFNode._additional_attrs_[op] = Op(getattr(operator, op))
