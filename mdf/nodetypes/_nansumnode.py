from ._nodetypes import MDFCustomNode, MDFCustomNodeIterator, nodetype
from ._datanode import _rowiternode
from ..nodes import MDFIterator
from ..context import MDFContext, _get_current_context
import datetime as dt
import pandas as pa
import numpy as np
import cython


class MDFNanSumNode(MDFCustomNode):

    def _cn_get_all_values(self, ctx, node_state):
        # get the iterator from the node state
        iterator = cython.declare(MDFCustomNodeIterator)
        iterator = node_state.generator

        nansumiter = cython.declare(_nansumnode)
        nansumiter = iterator.get_node_type_generator()

        return nansumiter._get_all_values(ctx)


class _nansumnode(MDFIterator):
    """
    Decorator that creates an :py:class:`MDFNode` that maintains
    the `nansum` of the result of `func`.

    Each time the context's date is advanced the value of this
    node is calculated as the nansum of the previous value
    and the new value returned by `func`.

    e.g.::

        @nansumnode
        def node():
            return some_value

    or using the nodetype method syntax (see :ref:`nodetype_method_syntax`)::

        @evalnode
        def some_value():
            return ...

        @evalnode
        def node():
            return some_value.nansum()
    """
    _init_args_ = ["value_node", "owner_node", "filter_node"]

    def __init__(self, value_node, owner_node, filter_node):
        self.is_float = False
        self.accum_initialized = False
        self.has_rowiter = False
        self.rowiter = None

        # If we can get all values from the value node then use a rowiter node
        ctx = cython.declare(MDFContext)
        ctx = _get_current_context()
        all_values = ctx._get_all_values(value_node)
        if all_values is not None:
            # calculate the cumulative sum, skipping nans
            summed_values = all_values.cumsum(axis=0, skipna=True)

            # create the simple row iterator with no delay or forward filling
            # (filtering is done by MDFCustomNode)
            self.rowiter = _rowiternode(data=summed_values,
                                        owner_node=owner_node,
                                        index_node_type=dt.datetime)
            self.has_rowiter = True
            return

        # Otherwise setup the iterator
        value = value_node()
        filter_node_value = True
        if filter_node is not None:
            filter_node_value = filter_node()

        if isinstance(value, pa.Series):
            self.accum = pa.Series(np.nan, index=value.index, dtype=value.dtype)
        elif isinstance(value, np.ndarray):
            self.accum = np.ndarray(value.shape, dtype=value.dtype)
            self.accum.fill(np.nan)
        else:
            self.is_float = True
            self.accum_f = np.nan

        if filter_node_value:
            self.send(value_node)

    def _send_vector(self, value):
        # set any nans in the accumulator where the value is not NaN to zero
        if not self.accum_initialized:
            accum_mask = np.isnan(self.accum) & ~np.isnan(value)
            if accum_mask.any():
                self.accum[accum_mask] = 0.0

            # if the accumulator has no nans then don't check again next time
            if not np.isnan(self.accum.values).any():
                self.accum_initialized = True

        np_mask = cython.declare(np.ndarray)
        np_value = cython.declare(np.ndarray)
        np_accum = cython.declare(np.ndarray)
        np_value_index = cython.declare(np.ndarray)
        np_accum_index = cython.declare(np.ndarray)

        np_value_index = value.index.values
        np_accum_index = self.accum.index.values

        # use numpy to do the mask if the index is the same (as it is in many cases)
        if (np_value_index == np_accum_index).all():
            np_value = value.values
            np_accum = self.accum.values
            np_mask = ~np.isnan(np_value)
            np_accum[np_mask] += np_value[np_mask]
        else:
            mask = ~np.isnan(value)
            if mask.all():
                self.accum += value
            else:
                self.accum[mask] += value[mask]

        return self.accum.copy()

    def _send_float(self, value):
        if value == value:
            if self.accum_f != self.accum_f:
                self.accum_f = 0.0
            self.accum_f += value
        return self.accum_f

    def next(self):
        if self.has_rowiter:
            return self.rowiter.next()

        if self.is_float:
            return self.accum_f
        return self.accum.copy()

    def send(self, value_node):
        if self.has_rowiter:
            return self.rowiter.next()

        value = value_node()
        if self.is_float:
            return self._send_float(value)
        return self._send_vector(value)

    def _get_all_values(self, ctx):
        if self.has_rowiter:
            return self.rowiter._get_all_values(ctx)


# decorators don't work on cythoned types
nansumnode = nodetype(cls=MDFNanSumNode, method="nansum")(_nansumnode)
