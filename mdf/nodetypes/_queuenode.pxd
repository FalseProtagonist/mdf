from ._nodetypes cimport MDFCustomNode
from ..nodes cimport MDFIterator


cdef class MDFQueueNode(MDFCustomNode):
    pass


cdef class _queuenode(MDFIterator):
    cdef object queue
    cdef int as_list

    cpdef next(self)
    cpdef send(self, value)
