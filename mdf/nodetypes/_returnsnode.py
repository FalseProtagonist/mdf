from ._nodetypes import MDFCustomNode, MDFCustomNodeIterator, nodetype, apply_filter
from ._datanode import _rowiternode
from ..nodes import MDFIterator
from ..context import _get_current_context
import datetime as dt
import numpy as np
import pandas as pa
import cython


class MDFReturnsNode(MDFCustomNode):
    nodetype_args = ["value_node", "filter_node", "owner_node", "use_diff"]
    nodetype_node_kwargs = ["value_node"]

    def _cn_get_all_values(self, ctx, node_state):
        # get the iterator from the node state
        iterator = cython.declare(MDFCustomNodeIterator)
        iterator = node_state.generator

        returnsiter = cython.declare(_returnsnode)
        returnsiter = iterator.get_node_type_generator()

        return returnsiter._get_all_values(ctx)

    def _cn_update_all_values(self, ctx, node_state):
        """called when the future data has changed"""
        iterator = cython.declare(MDFCustomNodeIterator)
        returnsiter = cython.declare(_returnsnode)
        iterator = node_state.generator
        if iterator is not None:
            returnsiter = iterator.get_node_type_generator()
            all_values = ctx._get_all_values(self._value_node)

            all_filter_values = None
            filter = self.get_filter()
            if filter is not None:
                all_filter_values = ctx._get_all_values(filter)

            return returnsiter._setup_rowiter(all_values, all_filter_values, self)


class _returnsnode(MDFIterator):
    """
    Decorator that creates an :py:class:`MDFNode` that returns
    the returns of a price series.

    NaN prices are filled forward.
    If there is a NaN price at the beginning of the series, we set
    the return to zero.
    The decorated function may return a float, pandas Series
    or numpy array.

    e.g.::

        @returnsnode
        def node():
            return some_price

    or using the nodetype method syntax (see :ref:`nodetype_method_syntax`)::

        @evalnode
        def some_price():
            return ...

        @evalnode
        def node():
            return some_price.returns()

    The value at any timestep is the return for that timestep, so the methods
    ideally would be called 'return', but that's a keyword and so returns is
    used.

    If use_diff=True then the difference between the prices is computed instead of the rate of return.
    """
    _init_args_ = ["value_node", "use_diff"]

    def __init__(self, value_node, filter_node, owner_node, use_diff=False):
        value = value_node()
        self.is_float = isinstance(value, float)
        self.use_diff = use_diff or False
        self.has_rowiter = False

        # check if we can get the dataframe for the value node
        ctx = _get_current_context()
        all_values = ctx._get_all_values(value_node)
        if all_values is not None:
            all_filter_values = None
            if filter_node:
                all_filter_values = ctx._get_all_values(filter_node)
            if all_filter_values is not None or filter_node is None:
                self._setup_rowiter(all_values, all_filter_values, owner_node)
                return

        if self.is_float:
            # floating point returns
            self.prev_value_f = np.nan
            self.current_value_f = np.nan
            self.return_f = 0.0
        else:
            # Series or ndarray returns
            if not isinstance(value, (pa.Series, np.ndarray)):
                raise RuntimeError("returns node expects a float, pa.Series or ndarray")

            if isinstance(value, pa.Series):
                self.prev_value = pa.Series(np.nan, index=value.index)
                self.current_value = pa.Series(np.nan, index=value.index)
            else:
                self.prev_value = np.ndarray(value.shape, dtype=value.dtype)
                self.current_value = np.ndarray(value.shape, dtype=value.dtype)
                self.returns = np.ndarray(value.shape, dtype=value.dtype)
                self.prev_value.fill(np.nan)
                self.current_value.fill(np.nan)
                self.returns.fill(0.0)

        filter_node_value = True
        if filter_node is not None:
            filter_node_value = filter_node()

        # update the current value
        if filter_node_value:
            self.send(value_node)

    def _setup_rowiter(self, all_values, all_filter_values, owner_node):
        if all_values is None:
            if self.has_rowiter:
                raise AssertionError("returnsnode was previously vectorized but now can't get all source values")
            self.rowiter = None
            self.has_rowiter = False
            return

        returns_df = self.__calculate_returns(all_values, all_filter_values, self.use_diff)
        self.rowiter = _rowiternode(data=returns_df,
                                    owner_node=owner_node,
                                    index_node_type=dt.datetime)
        self.has_rowiter = True

    def __calculate_returns(self, df, filter_mask, use_diff):
        """calculate returns for a dataframe"""
        if filter_mask is not None:
            df = apply_filter(df, filter_mask)

        # forward fill so returns for missing time points are zero
        df = df.fillna(method="ffill")

        if self.use_diff:
            returns = df - df.shift(1)
        else:
            returns = (df / df.shift(1)) -1.0

        returns.fillna(value=0.0, inplace=True)
        return returns

    def next(self):
        if self.has_rowiter:
            return self.rowiter.next()
        if self.is_float:
            return self.return_f
        return self.returns

    def send(self, value_node):
        if self.has_rowiter:
            return self.rowiter.next()

        value = value_node()
        if self.is_float:
            value_f = cython.declare(cython.double, value)

            # advance previous to the current value and update current
            # value with the new value unless it's nan (in which case we
            # leave it as it is - ie fill forward).
            self.prev_value_f = self.current_value_f
            if value_f == value_f:
                self.current_value_f = value_f

            if self.use_diff:
                self.return_f = self.current_value_f - self.prev_value_f
            else:
                self.return_f = (self.current_value_f / self.prev_value_f) - 1.0
            if self.return_f != self.return_f:
                self.return_f = 0.0
            return self.return_f

        # advance prev_value and update current value with any new
        # non-nan values
        mask = ~np.isnan(value)
        self.prev_value = self.current_value.copy()
        self.current_value[mask] = value[mask]

        if self.use_diff:
            self.returns = self.current_value - self.prev_value
        else:
            self.returns = (self.current_value / self.prev_value) - 1.0
        self.returns[np.isnan(self.returns)] = 0.0
        return self.returns

    def _get_all_values(self, ctx):
        if self.has_rowiter:
            return self.rowiter._get_all_values(ctx)


# decorators don't work on cythoned types
returnsnode = nodetype(cls=MDFReturnsNode, method="returns")(_returnsnode)
