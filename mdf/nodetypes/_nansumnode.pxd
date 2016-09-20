from ._nodetypes cimport MDFCustomNode
from ..nodes cimport MDFIterator


cdef class MDFNanSumNode(MDFCustomNode):
    pass


cdef class _nansumnode(MDFIterator):
    cdef object accum
    cdef double accum_f
    cdef int is_float

    cpdef next(self)
    cpdef send(self, value)

    cdef inline _send_vector(self, value)
    cdef inline double _send_float(self, double value)
