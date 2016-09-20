from ._nodetypes import MDFCustomNode, nodetype
from ..nodes import MDFIterator
import numpy as np
import pandas as pa
import cython


class MDFForwardFillNode(MDFCustomNode):
    pass


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
    _init_kwargs_ = ["filter_node_value", "initial_value"]

    def __init__(self, value, filter_node_value, initial_value=None):
        self.is_float = False
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
            self.send(value)

    def next(self):
        if self.is_float:
            return self.current_value_f
        return self.current_value.copy()

    def send(self, value):
        if self.is_float:
            # update the current value if value is not Nan
            value_f = cython.declare(cython.double, value)
            if value_f == value_f:
                self.current_value_f = value
            return self.current_value_f

        # update the current value with the non-nan values
        mask = ~np.isnan(value)
        self.current_value[mask] = value[mask]
        return self.current_value.copy()


# decorators don't work on cythoned types
ffillnode = nodetype(cls=MDFForwardFillNode, method="ffill")(_ffillnode)
