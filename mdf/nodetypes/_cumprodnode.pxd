from ._nodetypes cimport MDFCustomNode
from ..nodes cimport MDFIterator, MDFNode


cdef class MDFCumulativeProductNode(MDFCustomNode):
    pass


cdef class _cumprodnode(MDFIterator):
    cdef object accum
    cdef double accum_f
    cdef object nan_mask
    cdef int nan_mask_f
    cdef int is_float
    cdef int skipna

    cpdef next(self)
    cpdef send(self, value)

    cdef inline _send_vector(self, value)
    cdef inline double _send_float(self, double value)
