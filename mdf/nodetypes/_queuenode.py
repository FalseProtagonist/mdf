from ._nodetypes import MDFCustomNode, nodetype
from ..nodes import MDFIterator
from collections import deque
import cython


class MDFQueueNode(MDFCustomNode):
    pass


class _queuenode(MDFIterator):
    """
    Decorator for creating an :py:class:`MDFNode` that accumulates
    values in a `collections.deque` each time the context's date
    is advanced.

    The values that are accumulated are the results of the function
    `func`. `func` is a node function and takes no arguments.

    If `size` is specified the queue will grow to a maximum of that size
    and then values will be dropped off the queue (FIFO).

    `size` may either be a value or a callable (i.e. a function or a node)::

        @queuenode(size=10)
        def node():
            return x

    or::

        # could be an evalnode also
        queue_size = varnode("queue_size", 10)

        @queunode(size=queue_size)
        def node():
            return x

    or using the nodetype method syntax (see :ref:`nodetype_method_syntax`)::

        @evalnode
        def some_value():
            return ...

        @evalnode
        def node():
            return some_value.queue(size=5)
    """
    _init_args_ = ["value", "filter_node_value", "size", "as_list"]

    def __init__(self, value, filter_node_value, size=None, as_list=False):
        if size is not None:
            size = max(size, 1)

        self.as_list = as_list

        # create the queue used for the queue data
        self.queue = deque([], size)

        # only include the current value if the filter is
        # True (or if there's no filter being applied)
        if filter_node_value:
            self.queue.append(value)

    def __reduce__(self):
        """support for pickling"""
        return (
            _unpickle_queuenode,
            _pickle_queuenode(self),
            None,
            None,
            None,
        )

    def next(self):
        if self.as_list:
            return list(self.queue)
        return self.queue

    def send(self, value):
        self.queue.append(value)
        if self.as_list:
            return list(self.queue)
        return self.queue


def _unpickle_queuenode(queue, as_list):
    self = cython.declare(_queuenode)
    self = _queuenode(None, False, queue.maxlen, as_list)
    self.queue.extend(queue)
    return self


def _pickle_queuenode(self_):
    self = cython.declare(_queuenode)
    self = self_
    return (
        self.queue,
        self.as_list
    )


# decorators don't work on cythoned classes
queuenode = nodetype(_queuenode, cls=MDFQueueNode, method="queue")
