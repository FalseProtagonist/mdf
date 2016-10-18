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
from ._nodetypes import nodetype, MDFCustomNode
from ..nodes import MDFNode
import pandas as pa
import operator
import cython
import sys


__all__ = []


class MDFBinOpNode(MDFCustomNode):
    nodetype_args = ["value_node", "op", "op_name", "rhs"]
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
        op = node_type_kwargs["op"]
        op_name = node_type_kwargs["op_name"]
        rhs = node_type_kwargs.get("rhs", None)

        if rhs is None:
            return op(value_node_df)

        if isinstance(rhs, MDFNode):
            rhs = ctx._get_all_values(rhs)
            if rhs is None:
                return

        # if one side is a series and the other a dataframe broadcast the series to a dataframe
        if isinstance(value_node_df, pa.DataFrame) and isinstance(rhs, pa.Series):
            df_dicts = dict(zip(value_node_df.columns, [rhs] * len(value_node_df.columns)))
            rhs = pa.DataFrame(df_dicts, columns=value_node_df.columns, index=rhs.index)
        elif isinstance(value_node_df, pa.Series) and isinstance(rhs, pa.DataFrame):
            df_dicts = dict(zip(rhs.columns, [value_node_df] * len(rhs.columns)))
            value_node_df = pa.DataFrame(df_dicts, columns=rhs.columns, index=value_node_df.index)

        # use a pandas function if possible
        pandas_func = pandas_binops.get(op_name)
        if pandas_func is not None:
            return getattr(value_node_df, pandas_func)(rhs, axis="index")

        op = node_type_kwargs["op"]
        return op(value_node_df, rhs)


def _binopnode(value_node, op, op_name, rhs=None):
    """
    Return a new mdf node that applies a binary operator to two other nodes.
    """
    lhs = value_node()

    # it's quicker to switch on the interned operator name than to call the operator
    op_name_str = cython.declare(str)
    op_name_str = op_name

    if op_name_str == "__neg__":
        return -lhs
    elif op_name_str == "__add__":
        return lhs + rhs
    elif op_name_str == "__sub__":
        return lhs - rhs
    elif op_name_str == "__mul__":
        return lhs * rhs
    elif op_name_str == "__div__" or op_name_str == "__truediv__":
        return lhs / rhs
    elif op_name_str == "__lt__":
        return lhs < rhs
    elif op_name_str == "__le__":
        return lhs <= rhs
    elif op_name_str == "__gt__":
        return lhs > rhs
    elif op_name_str == "__ge__":
        return lhs >= rhs
    elif op_name_str == "__eq__":
        return lhs == rhs
    elif op_name_str == "__ne__":
        return lhs != rhs
    elif op_name_str == "__or__":
        return lhs | rhs
    elif op_name_str == "__and__":
        return lhs & rhs

    # we should never get here, but it doesn't hurt if we do
    if rhs is None:
        return op(lhs)
    return op(lhs, rhs)


# decorators don't work on cythoned types
binopnode = nodetype(cls=MDFBinOpNode, node_method="_binopnode")(_binopnode)


class Op(object):
    op = cython.declare(object)
    lhs = cython.declare(object)

    def __init__(self, op, op_name, lhs=None):
        self.op = op
        self.op_name = intern(op_name)
        self.lhs = lhs

    def __get__(self, instance, owner=None):
        if instance is not None:
            return self.__class__(self.op, self.op_name, instance)
        return self.__class__(self.op, self.op_name, owner)

    def __call__(self, rhs=None):
        return self.lhs._binopnode(op=self.op, op_name=self.op_name, rhs=rhs)


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
    "__ne__",
    "__or__",
    "__and__",
]


# for vector operations the pandas methods are used directly instead of the
# operator overloads. This is so 'axis' can be set to 'index' rather than
# the default 'columns'.
pandas_binops = {
    "__add__": "add",
    "__sub__": "sub",
    "__mul__": "mul",
    "__div__": "truediv",
    "__truediv__": "truediv",
}


for op in binops:
    MDFNode._additional_attrs_[op] = Op(getattr(operator, op), op)
