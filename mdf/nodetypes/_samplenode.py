from ._nodetypes import MDFCustomNode, nodetype
from ..nodes import MDFIterator, now
import pandas as pa
import numpy as np


class MDFSampleNode(MDFCustomNode):
    # always pass date_node as the node rather than evaluate it
    nodetype_node_kwargs = ["date_node"]


class _samplenode(MDFIterator):
    """
    samples value on the given date offset and yields that value
    until the next date offset.

    offset is a pandas.datetools.DateOffset instance,
    eg pandas.datetools.BMonthEnd()
    """
    _init_args_ = ["value", "filter_node_value", "offset", "date_node", "initial_value"]

    def __init__(self, value, filter_node_value, offset, date_node=now, initial_value=None):
        self._offset = offset
        self._date_node = date_node

        # if the initial value is a scalar but the value is a vector
        # broadcast the initial value
        if isinstance(initial_value, (int, float)):
            if isinstance(value, pa.Series):
                initial_value = pa.Series(initial_value, index=value.index, dtype=value.dtype)
            elif isinstance(value, np.ndarray):
                tmp = np.ndarray(value.shape, dtype=value.dtype)
                tmp.fill(initial_value)
                initial_value = tmp

        self._sample = initial_value

        if filter_node_value:
            self.send(value)

    def next(self):
        return self._sample

    def send(self, value):
        date = self._date_node()
        if date is not None and self._offset.onOffset(date):
            self._sample = value
        return self._sample


# decorators don't work on cythoned classes
samplenode = nodetype(_samplenode, cls=MDFSampleNode, method="sample")
