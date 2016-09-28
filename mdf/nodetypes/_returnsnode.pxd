from ._nodetypes cimport MDFCustomNode
from ..nodes cimport MDFIterator


cdef class _returnsnode(MDFIterator):
    cdef int is_float
    cdef double current_value_f
    cdef double prev_value_f
    cdef double return_f
    cdef object current_value
    cdef object prev_value
    cdef object returns
    cdef bint use_diff

    cpdef next(self)
    cpdef send(self, value)
