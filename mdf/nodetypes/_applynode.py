"""
applynode is a way of transforming a plain function into an mdf
node by binding other nodes to its parameters.
This is useful for quick interactive work more than for applications
written using mdf.
"""
from ._nodetypes import MDFCustomNode, nodetype
from ..nodes import MDFNode
from functools import wraps
import pandas as pa
import cython

# PURE PYTHON START
from ._nodetypes import dict_iteritems
# PURE PYTHON END


class MDFApplyNode(MDFCustomNode):
    nodetype_args = ["value_node", "func", "func_name", "args", "kwargs"]
    nodetype_node_kwargs = ["value_node"]

    def _cn_get_all_values(self, ctx, node_state):
        # if the value node has data then we can use pandas.apply
        value_node = self.value_node
        if self.value_node is None:
            return None

        value_node_df = ctx._get_all_values(value_node)
        if value_node_df is None:
            return

        node_type_kwargs = self.node_type_kwargs
        func = node_type_kwargs["func"]
        args = node_type_kwargs.get("args", tuple())
        kwargs = node_type_kwargs.get("kwargs", {})

        has_node_args = cython.declare(cython.bint)
        has_node_args = False

        for arg in args:
            if isinstance(arg, MDFNode):
                has_node_args = True

        for key, value in kwargs.iteritems():
            if isinstance(value, MDFNode):
                has_node_args = True

        # if any of the arguments are nodes we have to get them as dataframes too
        if has_node_args:
            arg_iters = {}
            arg_consts = {}
            for i, arg in enumerate(args):
                if isinstance(arg, MDFNode):
                    arg = ctx._get_all_values(arg)
                    if arg is None:
                        return
                    if isinstance(arg, pa.DataFrame):
                        arg_iters[i] = iter(arg.iterrows())
                    else:
                        arg_iters[i] = iter(arg)
                else:
                    arg_consts[i] = arg

            kwarg_iters = {}
            kwarg_consts = {}
            for key, arg in kwargs.iteritems():
                if isinstance(arg, MDFNode):
                    arg = ctx._get_all_values(arg)
                    if arg is None:
                        return
                    if isinstance(arg, pa.DataFrame):
                        kwarg_iters[i] = iter(arg.iterrows())
                    else:
                        kwarg_iters[i] = iter(arg)
                else:
                    kwarg_consts[i] = arg

            # create a wrapper function that gets the args per iteration
            def wrap_func(func, args, kwargs, arg_consts, kwarg_consts, arg_iters, kwarg_iters):
                unpacked_args = [None] * len(args)
                for i, arg in arg_consts:
                    unpacked_args[i] = arg

                unpacked_kwargs = {}
                for key, arg in kwarg_consts.iteritems():
                    unpacked_kwargs[key] = arg

                @wraps(func)
                def wrapped_func(value):
                    for i, arg_iter in arg_iters.iteritems():
                        unpacked_args[i] = next(arg_iter)

                    unpacked_kwargs = {}
                    for key, arg_iter in kwarg_iters.iteritems():
                        unpacked_kwargs[key] = next(arg_iter)

                    return func(value, *unpacked_args, **unpacked_kwargs)

                return wrapped_func

            func = wrap_func(func, args, kwargs, arg_consts, kwarg_consts, arg_iters, kwarg_iters)
            if isinstance(value_node_df, pa.DataFrame):
                return value_node_df.apply(func, axis=1)
            return value_node_df.apply(func)

        # if there are kwargs we have to wrap the function before we can use DataFrame.apply
        if kwargs:
            def wrap_func(func, **kwargs):
                @wraps(func)
                def wrapped_func(value, *args):
                    return func(value, *args, **kwargs)
                return wrapped_func

            func = wrap_func(func, **kwargs)

        if isinstance(value_node_df, pa.DataFrame):
            return value_node_df.apply(func, axis=1, args=args)
        return value_node_df.apply(func, args=args)


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

