from ._nodetypes import MDFCustomNode, nodetype
from ..nodes import MDFIterator
import numpy as np
import pandas as pa
import cython


class MDFReturnsNode(MDFCustomNode):
    pass


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
    """
    _init_kwargs_ = ["filter_node_value"]

    def __init__(self, value, filter_node_value):
        self.is_float = False
        if isinstance(value, float):
            # floating point returns
            self.is_float = True
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

        # update the current value
        if filter_node_value:
            self.send(value)

    def next(self):
        if self.is_float:
            return self.return_f
        return self.returns

    def send(self, value):
        if self.is_float:
            value_f = cython.declare(cython.double, value)

            # advance previous to the current value and update current
            # value with the new value unless it's nan (in which case we
            # leave it as it is - ie fill forward).
            self.prev_value_f = self.current_value_f
            if value_f == value_f:
                self.current_value_f = value_f

            self.return_f = (self.current_value_f / self.prev_value_f) - 1.0
            if np.isnan(self.return_f):
                self.return_f = 0.0
            return self.return_f

        # advance prev_value and update current value with any new
        # non-nan values
        mask = ~np.isnan(value)
        self.prev_value = self.current_value.copy()
        self.current_value[mask] = value[mask]

        self.returns = (self.current_value / self.prev_value) - 1.0
        self.returns[np.isnan(self.returns)] = 0.0
        return self.returns


# decorators don't work on cythoned types
returnsnode = nodetype(cls=MDFReturnsNode, method="returns")(_returnsnode)
