from ..nodes cimport MDFNode, MDFIterator
from ._nodetypes cimport dict_iteritems


cdef class _rowiternode(MDFIterator):
    cdef object _data
    cdef MDFNode _index_node
    cdef object _index_node_type
    cdef object _iter
    cdef object _current_index
    cdef object _current_value
    cdef object _prev_value
    cdef object _missing_value_orig
    cdef object _missing_value
    cdef int _ffill
    cdef int _is_dataframe
    cdef int _is_widepanel
    cdef int _is_series
    cdef int _index_to_date

    cdef _set_data(self, data)
    cdef _next_dataframe(self)
    cdef _next_widepanel(self)
    cdef _next_series(self)

    cpdef next(self)
    cpdef send(self, value)
