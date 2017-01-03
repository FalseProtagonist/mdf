from ._nodetypes import MDFCustomNode, nodetype
from ..nodes import MDFIterator
import pandas as pa
import numpy as np


class MDFCumulativeProductNode(MDFCustomNode):
    pass


class _cumprodnode(MDFIterator):
    """
    Decorator that creates an :py:class:`MDFNode` that maintains
    the cumulative product of the result of `func`.

    Each time the context's date is advanced the value of this
    node is calculated as the previous value muliplied by
    the new value returned by `func`.

    e.g.::

        @cumprodnode
        def node():
            return some_value

    or using the nodetype method syntax (see :ref:`nodetype_method_syntax`)::

        @evalnode
        def some_value():
            return ...

        @evalnode
        def node():
            return some_value.cumprod()

    TODO: That node needs a test for the argument skipna, since it is not entirely clear what it should do if the first value is na.
    It would be nice to be able to specify an initial value.
    """
    _init_args_ = ["value", "filter_node_value", "skipna"]

    def __init__(self, value, filter_node_value, skipna=True):
        self.is_float = False
        self.skipna = skipna
        if isinstance(value, pa.Series):
            self.accum = pa.Series(np.nan, index=value.index, dtype=value.dtype)
            self.nan_mask = np.isnan(self.accum)
        elif isinstance(value, np.ndarray):
            self.accum = np.ndarray(value.shape, dtype=value.dtype)
            self.accum.fill(np.nan)
            self.nan_mask = np.isnan(self.accum)
        else:
            self.is_float = True
            self.accum_f = np.nan
            self.nan_mask_f = True

        if filter_node_value:
            self.send(value)

    def _send_vector(self, value):
        # we keep track of a nan mask rather than re-evalute it each time
        # because if accum became nan after starting we wouldn't want to
        # start it from 1 again
        if self.nan_mask.any():
            self.nan_mask = self.nan_mask & np.isnan(value)
            self.accum[~self.nan_mask & ~np.isnan(value)] = 1.0

        if self.skipna:
            mask = ~np.isnan(value)
            self.accum[mask] *= value[mask]
        else:
            self.accum *= value

        return self.accum.copy()

    def _send_float(self, value):
        if self.nan_mask_f:
            if value == value:
                self.nan_mask_f = False
                self.accum_f = 1.0

        if not self.skipna \
                or value == value:
            self.accum_f *= value

        return self.accum_f

    def next(self):
        if self.is_float:
            return self.accum_f
        return self.accum.copy()

    def send(self, value):
        if self.is_float:
            return self._send_float(value)
        return self._send_vector(value)


# decorators don't work on cythoned types
cumprodnode = nodetype(cls=MDFCumulativeProductNode, method="cumprod")(_cumprodnode)
