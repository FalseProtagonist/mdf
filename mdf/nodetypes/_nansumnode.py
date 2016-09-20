from ._nodetypes import MDFCustomNode, nodetype
from ..nodes import MDFIterator
import pandas as pa
import numpy as np


class MDFNanSumNode(MDFCustomNode):
    pass


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
    _init_args_ = ["value", "filter_node_value"]

    def __init__(self, value, filter_node_value):
        self.is_float = False
        if isinstance(value, pa.Series):
            self.accum = pa.Series(np.nan, index=value.index, dtype=value.dtype)
        elif isinstance(value, np.ndarray):
            self.accum = np.ndarray(value.shape, dtype=value.dtype)
            self.accum.fill(np.nan)
        else:
            self.is_float = True
            self.accum_f = np.nan

        if filter_node_value:
            self.send(value)

    def _send_vector(self, value):
        mask = ~np.isnan(value)

        # set an nans in the accumulator where the value is not
        # NaN to zero
        accum_mask = np.isnan(self.accum)
        if accum_mask.any():
            self.accum[accum_mask & mask] = 0.0

        self.accum[mask] += value[mask]
        return self.accum.copy()

    def _send_float(self, value):
        if value == value:
            if self.accum_f != self.accum_f:
                self.accum_f = 0.0
            self.accum_f += value
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
nansumnode = nodetype(cls=MDFNanSumNode, method="nansum")(_nansumnode)
