from ._nodetypes import MDFCustomNode, MDFCustomNodeIterator, nodetype
from ._datanode import _rowiternode
from ..nodes import MDFIterator
from ..context import MDFContext, _get_current_context
import datetime as dt
import numpy as np
import pandas as pa
import cython


class MDFForwardFillNode(MDFCustomNode):

    def _cn_get_all_values(self, ctx, node_state):
        # get the iterator from the node state
        iterator = cython.declare(MDFCustomNodeIterator)
        iterator = node_state.generator

        ffilliter = cython.declare(_ffillnode)
        ffilliter = iterator.get_node_type_generator()

        return ffilliter._get_all_values(ctx)

    def _cn_update_all_values(self, ctx, node_state):
        """called when the future data has changed"""
        iterator = cython.declare(MDFCustomNodeIterator)
        ffilliter = cython.declare(_ffillnode)
        iterator = node_state.generator
        if iterator is not None:
            ffilliter = iterator.get_node_type_generator()
            all_values = ctx._get_all_values(self._value_node)
            return ffilliter._setup_rowiter(all_values, self)


class _ffillnode(MDFIterator):
    """
    Decorator that creates an :py:class:`MDFNode` that returns
    the current result of the decoratored function forward
    filled from the previous value where the current value
    is NaN.

    The decorated function may return a float, pandas Series
    or numpy array.

    e.g.::

        @ffillnode
        def node():
            return some_value

    or using the nodetype method syntax (see :ref:`nodetype_method_syntax`)::

        @evalnode
        def some_value():
            return ...

        @evalnode
        def node():
            return some_value.ffill()
    """
    _init_args_ = ["value_node", "filter_node", "owner_node", "initial_value"]

    def __init__(self, value_node, filter_node, owner_node, initial_value=None):
        self.is_float = False
        self.initial_value = initial_value

        self.has_rowiter = False
        self.rowiter = None

        # If we can get all values from the value node then use a rowiter node
        ctx = cython.declare(MDFContext)
        ctx = _get_current_context()
        all_values = ctx._get_all_values(value_node)
        if all_values is not None:
            self._setup_rowiter(all_values, owner_node)
            return

        # Otherwise set up the iterator
        value = value_node()
        filter_node_value = True
        if filter_node is not None:
            filter_node_value = filter_node()

        if isinstance(value, float):
            #
            # floating point fill forward
            #
            self.is_float = True
            self.current_value_f = initial_value if initial_value is not None else np.nan
        else:
            #
            # Series or ndarray fill forward
            #
            if not isinstance(value, (pa.Series, np.ndarray)):
                raise RuntimeError("fillnode expects a float, pa.Series or ndarray")

            if initial_value is not None:
                if isinstance(initial_value, (float, int)):
                    if isinstance(value, pa.Series):
                        self.current_value = pa.Series(initial_value,
                                                       index=value.index,
                                                       dtype=value.dtype)
                    else:
                        self.current_value = np.ndarray(value.shape, dtype=value.dtype)
                        self.current_value.fill(initial_value)
                else:
                    # this ensures the current_value ends up being the same type
                    # as value, even if initial_value is another vector type.
                    self.current_value = value.copy()
                    self.current_value[:] = initial_value[:]
            else:
                if isinstance(value, pa.Series):
                    self.current_value = pa.Series(np.nan, index=value.index, dtype=value.dtype)
                else:
                    self.current_value = np.ndarray(value.shape, dtype=value.dtype)
                    self.current_value.fill(np.nan)

        # update the current value
        if filter_node_value:
            self.send(value_node)

    def _setup_rowiter(self, all_values, owner_node):
        # if all values is None check we didn't previously have an iterator
        if all_values is None:
            if self.has_rowiter:
                raise AssertionError("ffillnode was previously vectorized but now can't get all source values")
            self.rowiter = None
            self.has_rowiter = False
            return

        # forward fill
        all_values = all_values.fillna(method="ffill")
        if self.initial_value is not None:
            all_values = all_values.fillna(value=self.initial_value)

        # create the simple row iterator with no delay or forward filling
        # (filtering is done by MDFCustomNode)
        self.rowiter = _rowiternode(data=all_values,
                                    owner_node=owner_node,
                                    index_node_type=dt.datetime)
        self.has_rowiter = True

    def next(self):
        if self.has_rowiter:
            return self.rowiter.next()

        if self.is_float:
            return self.current_value_f
        return self.current_value.copy()

    def send(self, value_node):
        if self.has_rowiter:
            return self.rowiter.next()

        if self.is_float:
            # update the current value if value is not Nan
            value_f = cython.declare(cython.double)
            value_f = value_node()
            if value_f == value_f:
                self.current_value_f = value_f
            return self.current_value_f

        # update the current value with the non-nan values
        value = value_node()
        mask = ~np.isnan(value)
        self.current_value[mask] = value[mask]
        return self.current_value.copy()

    def _get_all_values(self, ctx):
        if self.has_rowiter:
            return self.rowiter._get_all_values(ctx)


# decorators don't work on cythoned types
ffillnode = nodetype(cls=MDFForwardFillNode, method="ffill")(_ffillnode)
