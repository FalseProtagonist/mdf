from ._nodetypes import MDFCustomNode
from ..nodes cimport MDFIterator


cdef class _ffillnode(MDFIterator):
    cdef int is_float
    cdef double current_value_f
    cdef object current_value

    cpdef next(self)
    cpdef send(self, value)
