"""
applynode is a way of transforming a plain function into an mdf
node by binding other nodes to its parameters.
This is useful for quick interactive work more than for applications
written using mdf.
"""
from ._nodetypes import MDFCustomNode, nodetype
from ..nodes import MDFNode
import cython

# PURE PYTHON START
from ._nodetypes import dict_iteritems
# PURE PYTHON END


class MDFApplyNode(MDFCustomNode):
    nodetype_args = ["value_node", "func", "func_name", "args", "kwargs"]
    nodetype_node_kwargs = ["value_node"]


def _applynode(value_node, func, func_name=None, args=(), kwargs={}):
    """
    Return a new mdf node that applies `func` to the value of the node
    that is passed in. Extra `args` and `kwargs` can be passed in as
    values or nodes.

    Unlike most other node types this shouldn't be used as a decorator, but instead
    should only be used via the method syntax for node types, (see :ref:`nodetype_method_syntax`)
    e.g.::

        A_plus_B_node = A.applynode(operator.add, args=(B,))

    """
    new_args = []
    for arg in args:
        if isinstance(arg, MDFNode):
            arg = arg()
        new_args.append(arg)

    new_kwargs = {}
    for key, value in dict_iteritems(kwargs):
        if isinstance(value, MDFNode):
            value = value()
        new_kwargs[key] = value

    value = value_node()
    return func(value, *new_args, **new_kwargs)


# decorators don't work on cythoned types
applynode = nodetype(cls=MDFApplyNode, method="apply")(_applynode)

