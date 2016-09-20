from ..nodes cimport MDFIterator, MDFNode


cdef class _samplenode(MDFIterator):
    cdef object _offset
    cdef MDFNode _date_node
    cdef object _sample

    cpdef next(self)
    cpdef send(self, value)
