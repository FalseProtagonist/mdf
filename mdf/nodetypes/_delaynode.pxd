from ._nodetypes cimport MDFCustomNode
from ..nodes cimport MDFIterator
from ..context cimport MDFContext


cdef class MDFDelayNode(MDFCustomNode):
    cdef object _dn_func
    cdef dict _dn_per_ctx_data
    cdef int _dn_is_generator
    cdef int _dn_lazy

    cpdef _dn_get_prev_value(self)


cdef class _delaynode(MDFIterator):
    cdef int lazy
    cdef int skip_nans
    cdef object queue

    cpdef next(self)
    cpdef send(self, value)
