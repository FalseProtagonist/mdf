import datetime
import numpy as np
import pandas as pa
import cython
from ._nodetypes import MDFCustomNode, MDFCustomNodeIterator, nodetype
from ..common import DIRTY_FLAGS
from ..context import MDFContext, _get_current_context
from ..parser import get_assigned_node_name
from ..nodes import (
    MDFNode,
    MDFIterator,
    MDFCallable,
    NodeState,
    now,
)

# needed in pure-python mode
#from ._nodetypes import dict_iteritems


__all__ = [
    "rowiternode",
    "datanode",
    "filternode"
]


DIRTY_FLAGS_FUTURE_DATA = cython.declare(int, DIRTY_FLAGS.FUTURE_DATA)


#
# datarownode is used to construct nodes from either DataFrames, WidePanels or
# TimeSeries.
#
class MDFRowIteratorNode(MDFCustomNode):
    # always pass index_node as the node rather than evaluate it
    nodetype_node_kwargs = ["index_node"]


    def append(self, data, ctx=None):
        """
        Appends new data to the dataframe/series.

        This can be called at any time, but the data added must only be effective
        strictly after the current value of 'now'.
        """
        if ctx is None:
            ctx = _get_current_context()

        # make sure the node has been evaluated before getting the node state
        _unused = ctx[self]
        alt_ctx = cython.declare(MDFContext)
        alt_ctx = self._get_alt_context(ctx)

        node_state = cython.declare(NodeState)
        node_state = self._states[alt_ctx._id_obj]

        # get the row iterator from the node state
        iterator = cython.declare(MDFCustomNodeIterator)
        iterator = node_state.generator

        rowiter = cython.declare(_rowiternode)
        rowiter = iterator.node_type_generator

        # add the data to the iterator
        rowiter._append(data, alt_ctx)

        # mark the node as dirty for future data
        self.set_dirty(alt_ctx, DIRTY_FLAGS_FUTURE_DATA)


class _rowiternode(MDFIterator):
    """
    Decorator that creates an :py:class:`MDFNode` that returns
    the current row of item of a pandas DataFrame, WidePanel
    or Series returned by the decorated function.

    What row is considered current depends on the `index_node`
    parameter, which by default is `now`.

    `missing_value` may be specified as the value to use when
    the index_node isn't included in the data's index. The
    default is NaN.

    `delay` can be a number of timesteps to delay the index_node
    by, effectively shifting the data.

    `ffill` causes the value to get forward filled if True, default is False.

    e.g.::

        @rowiternode
        def datarow_node():
            # construct a dataframe indexed by date
            return a_dataframe

        @evalnode
        def another_node():
            # the rowiternode returns the row from the dataframe
            # for the current date 'now'
            current_row = datarow_node()

    or using the nodetype method syntax (see :ref:`nodetype_method_syntax`)::

        @evalnode
        def dataframe_node():
            # construct a dataframe indexed by date
            return a_dataframe

        @evalnode
        def another_node():
            # get the row from dataframe_node for the current_date 'now'
            current_row = dataframe_node.rowiter()
    """
    _init_kwargs_ = ["owner_node", "index_node", "missing_value", "delay", "ffill"]

    def __init__(self, data, owner_node, index_node=now, missing_value=np.nan, delay=0, ffill=False):
        """data should be a dataframe, widepanel or timeseries"""
        self._owner_node = owner_node
        self._current_index = None
        self._current_value = None
        self._prev_value = None
        self._missing_value_orig = missing_value
        self._index_to_date = False
        self._ffill = ffill

        # data can be appended to - keep track of the first data block and any subsequent blocks
        self._initial_data = data
        self._appended_data = []
        self._appended_data_index = 0

        # call the index node to make sure this node depends on it and remember the type
        index_value = index_node()
        self._index_node_type = type(index_value)

        # store the node and delay it if necessary
        self._index_node = index_node
        if delay > 0:
            self._index_node = self._index_node.delaynode(periods=delay,
                                                          filter=owner_node.get_filter())

        self._set_data(data)

    def _get_iterator(self, data):
        """return a tuple of (iterator, current index, current value for a dataframe, panel or series"""
        if self._is_dataframe:
            iterator = iter(data.index)
            current_index = next(iterator)
            current_value = data.xs(current_index)
        if self._is_widepanel:
            iterator = iter(data.major_axis)
            current_index = next(iterator)
            current_value = data.major_xs(current_index)
        if self._is_series:
            iterator = iter(sorted(dict_iteritems(data)))
            current_index, current_value = next(iterator)

        return iterator, current_index, current_value

    def _set_data(self, data):
        self._data = data

        self._is_dataframe = False
        self._is_widepanel = False
        self._is_series = False

        # this may get updated (e.g. to be a series corresponding to the columns
        # of a dataframe) so restore it to the original value.
        self._missing_value = self._missing_value_orig

        if isinstance(data, pa.DataFrame):
            self._is_dataframe = True

            # convert missing value to a row with the same columns as the dataframe
            if not isinstance(self._missing_value, pa.Series):
                dtype = object
                if data.index.size > 0:
                    dtype = data.xs(data.index[0]).dtype
                self._missing_value = pa.Series(self._missing_value,
                                                index=data.columns,
                                                dtype=dtype)

        elif isinstance(data, pa.WidePanel):
            self._is_widepanel = True

            # convert missing value to a dataframe with the same dimensions as the panel
            if not isinstance(self._missing_value, pa.DataFrame):
                if not isinstance(self._missing_value, dict):
                    self._missing_value = dict([(c, self._missing_value) for c in data.items])
                self._missing_value = pa.DataFrame(self._missing_value,
                                                   columns=data.items,
                                                   index=data.minor_axis,
                                                   dtype=data.dtype)

        elif isinstance(data, pa.Series):
            self._is_series = True

        else:
            clsname = type(data)
            if hasattr(data, "__class__"):
                clsname = data.__class__.__name__
            raise AssertionError("datanode expects a DataFrame, WidePanel or Series; "
                                 "got '%s'" % clsname)

        # set up the iterator
        try:
            self._iter, self._current_index, self._current_value = self._get_iterator(data)
        except StopIteration:
            self._current_index = None
            self._current_value = self._missing_value

        # reset _prev_value to the missing value, it will get set as the iterator is advanced.
        self._prev_value = self._missing_value

        # does the index need to be converted from datetime to date?
        # (use the stored index_node_type as the current value may be delayed
        # and therefore be None instead of it usual type)
        self._index_to_date = type(self._current_index) is datetime.date \
                              and self._index_node_type is datetime.datetime

    def send(self, data):
        if data is not self._initial_data:
            # clear the previous data and any appended data and restart with
            # the new dataframe/series etc
            self._set_data(data)
            self._appended_data = []
            self._appended_data_index = 0
        return self.next()

    def next(self):
        # switching this way cythons better than having a python
        # function object and calling that, because it doesn't have
        # the overhead of doing a python object call.
        if self._is_dataframe:
            return self._next_dataframe()
        if self._is_widepanel:
            return self._next_widepanel()
        if self._is_series:
            return self._next_series()
        return self._missing_value

    def _next_dataframe(self):
        i = self._index_node()
        if self._current_index is None \
                or i is None:
            return self._missing_value

        if self._index_to_date:
            i = i.date()

        while i > self._current_index:
            # advance to the next item in the series
            try:
                # TODO: once we upgrade pandas use the iterrows method
                self._prev_value = self._current_value
                self._current_index = next(self._iter)
                self._current_value = self._data.xs(self._current_index)
            except StopIteration:
                # If another block of data has been appended switch to that
                if self._appended_data_index < len(self._appended_data):
                    self._set_data(self._appended_data[self._appended_data_index])
                    self._appended_data_index += 1
                    continue

                # otherwise we're at the end of the data
                if self._ffill:
                    return self._current_value
                return self._missing_value

        if self._current_index == i:
            return self._current_value

        if self._ffill and self._current_index > i:
            return self._prev_value

        return self._missing_value

    def _next_widepanel(self):
        i = self._index_node()
        if self._current_index is None \
                or i is None:
            return self._missing_value

        if self._index_to_date:
            i = i.date()

        while i > self._current_index:
            # advance to the next item in the series
            try:
                # TODO: once we upgrade pandas use the iterrows method
                self._prev_value = self._current_value
                self._current_index = next(self._iter)
                self._current_value = self._data.major_xs(self._current_index)
            except StopIteration:
                # If another block of data has been appended switch to that
                if self._appended_data_index < len(self._appended_data):
                    self._set_data(self._appended_data[self._appended_data_index])
                    self._appended_data_index += 1
                    continue

                # otherwise we're at the end of the data
                if self._ffill:
                    return self._prev_value
                return self._missing_value

        if self._current_index == i:
            return self._current_value

        if self._ffill and self._current_index > i:
            return self._prev_value

        return self._missing_value

    def _next_series(self):
        i = self._index_node()
        if self._current_index is None \
                or i is None:
            return self._missing_value

        if self._index_to_date:
            i = i.date()

        while i > self._current_index:
            # advance to the next item in the series
            try:
                self._prev_value = self._current_value
                self._current_index, self._current_value = next(self._iter)
            except StopIteration:
                # If another block of data has been appended switch to that
                if self._appended_data_index < len(self._appended_data):
                    self._set_data(self._appended_data[self._appended_data_index])
                    self._appended_data_index += 1
                    continue

                # otherwise we're at the end of the data
                if self._ffill:
                    return self._prev_value
                return self._missing_value

        if self._current_index == i:
            return self._current_value

        if self._ffill and self._current_index > i:
            return self._prev_value

        return self._missing_value

    def _append(self, data, ctx):
        # check the type of the data matches the original data
        if self._is_dataframe:
            assert isinstance(data, pa.DataFrame), \
                "Expected a DataFrame when appending to data node '%s'" % self._owner_node.name
        elif self._is_widepanel:
            assert isinstance(data, pa.WidePanel), \
                "Expected a WidePanel when appending to data node '%s'" % self._owner_node.name
        elif self._is_series:
            assert isinstance(data, pa.Series), \
                "Expected a Series when appending to data node '%s'" % self._owner_node.name

        # check we've not advanced past the first item in the data
        _unused, first_index, _unused = self._get_iterator(data)

        current_index = ctx[self._index_node]
        if type(first_index) is datetime.date and type(current_index) is datetime.datetime:
            current_index = current_index.date()

        if current_index >= first_index:
            raise Exception("Cannot append data as the index <= current index (i.e. new data is in the past).")

        # add the data
        self._appended_data.append(data)


# decorators don't work on cythoned types
rowiternode = nodetype(cls=MDFRowIteratorNode, method="rowiter")(_rowiternode)


#
# helper function for creating a row iterator node, but without having
# to write the function just to return a dataframe/series etc...
#
def datanode(name=None,
             data=None,
             index_node=now,
             missing_value=np.nan,
             delay=0,
             ffill=False,
             filter=None,
             category=None):
    """
    Return a new mdf node for iterating over a dataframe, panel or series.

    `data` is indexed by another node `index_node`, (default is :py:func:`now`),
    which can be any node that evaluates to a value that can be used to index
    into `data`.

    If the `index_node` evaluates to a value that is not present in
    the index of the `data` then `missing_value` is returned.

    `missing_value` can be a scalar, in which case it will be converted
    to the same row format used by the data object with the same value
    for all items.

    `delay` can be a number of timesteps to delay the index_node
    by, effectively shifting the data.

    `ffill` causes the value to get forward filled if True, default is False.

    `data` may either be a data object itself (DataFrame, WidePanel or
    Series) or a node that evaluates to one of those types.

    e.g.::

        df = pa.DataFrame({"A" : range(100)}, index=date_range)
        df_node = datanode(data=df)

        ctx[df_node] # returns the row from df where df == ctx[now]

    A datanode may be explicitly named using the name argument, or
    if left as None the variable name the node is being assigned to
    will be used.
    """
    assert data is not None, "Must specify data as a DataFrame, Series or node"

    if name is None:
        name = get_assigned_node_name("datanode", 0 if cython.compiled else 1)

    if isinstance(data, MDFNode):
        func = MDFCallable(name, data)
    else:
        func = MDFCallable(name, lambda: data)

    node = MDFRowIteratorNode(name=name,
                              func=func,
                              node_type_func=_rowiternode,
                              category=category,
                              filter=filter,
                              nodetype_func_kwargs={
                                  "index_node": index_node,
                                  "delay": delay,
                                  "missing_value": missing_value,
                                  "ffill": ffill,
                              })
    return node


def filternode(name=None,
               data=None,
               index_node=now,
               delay=0,
               filter=None,
               category=None):
    """
    Return a new mdf node for using as a filter for other nodes
    based on the index of the data object passed in (DataFrame,
    Series or WidePanel).

    The node value is True when the index_node (default=now)
    is in the index of the data, and False otherwise.

    This can be used to easily filter other nodes so that
    they operate at the same frequency of the underlying data.

    `delay` can be a number of timesteps to delay the index_node
    by, effectively shifting the data.

    A filternode may be explicitly named using the name argument, or
    if left as None the variable name the node is being assigned to
    will be used.
    """
    assert data is not None, "Must specify data as a DataFrame, Series or node"

    # the filter is always True for points on the data's index,
    # and False otherwise.
    func = lambda: pa.Series(True, index=data.index)
    if isinstance(data, MDFNode):
        func = lambda: pa.Series(True, index=data().index)

    if name is None:
        name = get_assigned_node_name("filternode", 0 if cython.compiled else 1)

    if isinstance(data, MDFNode):
        func = MDFCallable(name, data, lambda x: pa.Series(True, index=x.index))
    else:
        func = MDFCallable(name, lambda: pa.Series(True, index=data.index))

    node = MDFRowIteratorNode(name=name,
                              func=MDFCallable(name, func),
                              node_type_func=_rowiternode,
                              category=category,
                              filter=filter,
                              nodetype_func_kwargs={
                                  "index_node": index_node,
                                  "missing_value": False,
                                  "delay": delay,
                              })
    return node
