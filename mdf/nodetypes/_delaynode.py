from ._nodetypes import MDFCustomNode, MDFCustomNodeIteratorFactory, nodetype
from ..nodes import MDFIterator, _isgeneratorfunction
from ..context import MDFContext, _get_current_context
from collections import deque, namedtuple
import numpy as np
import pandas as pa
import cython


class MDFDelayNode(MDFCustomNode):
    """
    MDFDelayedNode is different from other MDFCustomNodes.
    The value passed to the node function is actually the
    value *from the previous day*.

    This breaks the recursion for nodes that want to access
    a delayed version of themself, eg::

        @delaynode(periods=1, initial_value=0, lazy=True)
        def delayed_foo():
            return foo()

        @evalnode
        def foo():
            return 1 + delayed_foo()

    Here calling delayed_foo doesn't causes a recursive call to
    foo since MDFDelayedNode doesn't call the function
    immediately, it waits until the timestep is about to be
    advanced.
    """
    PerCtxData = namedtuple("PerCtxData", ["value", "generator", "date", "is_valid"])

    def __init__(self,
                 func,
                 node_type_func=None,
                 name=None,
                 short_name=None,
                 fqname=None,
                 cls=None,
                 category=None,
                 filter=None,
                 nodetype_func_kwargs={},
                 **kwargs):
        if isinstance(func, MDFCustomNodeIteratorFactory):
            node_type_func = func.node_type_func
            func = func.func
        self._dn_func = func
        self._dn_per_ctx_data = {}
        self._dn_is_generator = _isgeneratorfunction(func)
        self._dn_lazy = nodetype_func_kwargs.get("lazy", False)

        #
        # the node is initialized either just with the plain node function
        # (if lazy is False), or with the _dn_get_prev_value method if
        # lazy is true. The previous value is evaluated during
        # MDFContext.set_date by the _on_set_date callback.
        #
        MDFCustomNode.__init__(self,
                               self._dn_get_prev_value if self._dn_lazy else func,
                               node_type_func,
                               name=name or self._get_func_name(func),
                               short_name=short_name,
                               fqname=fqname,
                               cls=cls,
                               category=category,
                               filter=filter,
                               nodetype_func_kwargs=nodetype_func_kwargs,
                               **kwargs)

    @property
    def func(self):
        return self._dn_func

    def clear_value(self, ctx):
        try:
            del self._dn_per_ctx_data[ctx._id]
        except KeyError:
            pass

        MDFCustomNode.clear_value(self, ctx)

    def clear(self, ctx):
        self.clear_value(ctx)
        MDFCustomNode.clear(self, ctx)

    def _bind(self, other_node, owner):
        other = cython.declare(MDFDelayNode)
        other = other_node
        MDFCustomNode._bind(self, other, owner)
        self._dn_func = self._bind_function(other._dn_func, owner)
        self._dn_is_generator = other._dn_is_generator
        self._dn_lazy = other._dn_lazy

        func = self._dn_get_prev_value if self._dn_lazy else self._dn_func
        self._set_func(func)

        # set the docstring for the bound node to the same as the unbound one
        self.func_doc = other.func_doc

    def _dn_get_prev_value(self):
        # The value returned on date 'now' is the value for the previous day.
        # This means that the value for now doesn't have to be calculated
        # until just before now is advanced. This breaks the recursion
        # of functions that want to call a node that is a delayed version
        # of the same node.
        ctx = _get_current_context()
        alt_ctx = self.get_alt_context(ctx)
        try:
            data = self._dn_per_ctx_data[alt_ctx._id]
            if data.is_valid:
                return data.value
        except KeyError:
            pass

        kwargs = self._get_kwargs()
        return kwargs["initial_value"]

    def on_set_date(self, ctx_, date):
        """called just before 'now' is advanced"""
        ctx = cython.declare(MDFContext)
        ctx = ctx_

        # if not lazy there's nothing to do
        if not self._dn_lazy:
            return False

        # grab the original alt ctx before it's modified by calling the node func
        orig_alt_ctx = self.get_alt_context(ctx)

        if date > ctx._now:
            # if this node hasn't been valued before, clear any previous
            # alt context set as the dependencies wouldn't have been set
            # up when the alt ctx was determined
            if ctx._id not in self._dn_per_ctx_data:
                self._reset_alt_context(ctx)

            # if there's a filter set don't update the previous value unless
            # the filter returns True
            filter = self.get_filter()
            if filter is not None and not filter():
                return False

            # if there's already a value for this date then don't do anything
            alt_ctx = self.get_alt_context(ctx)
            alt_data = self._dn_per_ctx_data.get(alt_ctx._id)
            if alt_data and alt_data.is_valid and alt_data.date == date:
                return False

            # get the current value of the node function/generator
            generator = None
            if self._dn_is_generator:
                if alt_data:
                    generator = alt_data.generator
                if not generator:
                    generator = self._dn_func()
                value = next(generator)
            else:
                value = self._dn_func()

            # get the alt context again as it could have changed because of new
            # dependencies added when calling the node function/generator
            alt_ctx = self.get_alt_context(ctx)

            # store the generator and value in the alt_ctx, and put an invalid entry
            # in the original context so this context doesn't get its alt ctx reset next time
            self._dn_per_ctx_data[alt_ctx._id] = self.PerCtxData(value, generator, date, True)
            if alt_ctx is not ctx:
                self._dn_per_ctx_data[ctx._id] = self.PerCtxData(None, None, date, False)

        elif date < ctx._now:
            self.clear_value(ctx)
            alt_ctx = self.get_alt_context(ctx)
            if alt_ctx is not ctx:
                self.clear_value(alt_ctx)

        # return True to indicate the value of this node will change after the date has
        # finished being changed.
        return True


class _delaynode(MDFIterator):
    """
    Decorator for creating an :py:class:`MDFNode` that delays
    values for a number of periods corresponding to each time
    the context's date is advanced.

    The values that are delayed are the results of the function
    `func`. `func` is a node function and takes no arguments.

    ``periods`` is the number of timesteps to delay the value by.

    ``initial_value`` is the value of the node to be used before
    the specified number of periods have elapsed.

    `periods`, `initial_value` and `filter` can either be values
    or callable objects (e.g. a node or a function)::

        @delaynode(periods=5)
        def node():
            return x

    or::

        # could be an evalnode also
        periods = varnode("periods", 5)

        @delaynode(periods=periods)
        def node():
            return x

    If ``lazy`` is True the node value is calculated after any calling
    nodes have returned. This allows nodes to call delayed version of
    themselves without ending up in infinite recursion.

    The default for ``lazy`` is False as in most cases it's not
    necessary and can cause problems because the dependencies aren't
    all discovered when the node is first evaluated.

    e.g.::

        @delaynode(periods=10)
        def node():
            return some_value

    or using the nodetype method syntax (see :ref:`nodetype_method_syntax`)::

        @evalnode
        def some_value():
            return ...

        @evalnode
        def node():
            return some_value.delay(periods=5)
    """
    _init_kwargs_ = ["filter_node_value", "periods", "initial_value", "lazy", "ffill"]

    def __init__(self, value, filter_node_value, periods=1,
                 initial_value=None, lazy=False, ffill=False):
        self.lazy = lazy
        self.skip_nans = ffill
        max_queue_size = 0

        # if the initial value is a scalar but the value is a vector
        # broadcast the initial value
        if isinstance(initial_value, (int, float)):
            if isinstance(value, pa.Series):
                initial_value = pa.Series(initial_value, index=value.index, dtype=value.dtype)
            elif isinstance(value, np.ndarray):
                tmp = np.ndarray(value.shape, dtype=value.dtype)
                tmp.fill(initial_value)
                initial_value = tmp

        if lazy:
            # NOTE: when lazy the value is *already delayed by 1* (see MDFDelayNode)
            assert periods is not None and periods > 0, "lazy delay nodes must have 'periods' set to > 0"
            max_queue_size = periods


        else:
            # max size is periods+1 (if a value is delayed 0 periods the length of the queue must be 1)
            assert periods is not None and periods >= 0, "delay nodes must have 'periods' set to >= 0"
            max_queue_size = periods + 1

        # create the queue and fill it with the initial value
        self.queue = deque([initial_value] * max_queue_size, max_queue_size)

        # send the current value if the filter value is True, or if the node
        # is lazy. If it's lazy the filtering is done by the on_set_date callback
        # since it needs to be filtered based on the previous filter value.
        if filter_node_value or lazy:
            self.send(value)

    def next(self):
        return self.queue[0]

    def send(self, value):
        skip = False
        if self.skip_nans:
            if isinstance(value, float):
                if np.isnan(value):
                    skip = True
            elif isinstance(value, np.ndarray):
                if np.isnan(value).all():
                    skip = True
        if not skip:
            self.queue.append(value)
        return self.queue[0]


# decorators don't work on cythoned classes
delaynode = nodetype(_delaynode, cls=MDFDelayNode, method="delay")
