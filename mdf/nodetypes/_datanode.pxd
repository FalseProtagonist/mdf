from ..nodes cimport MDFNode, MDFIterator, NodeState
from ..context cimport MDFContext
from ._nodetypes cimport MDFCustomNode, MDFCustomNodeIterator, dict_iteritems


cdef class MDFRowIteratorNode(MDFCustomNode):
    cpdef append(self, data, MDFContext ctx=?)

    # MDFCustomNode overrides
    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)


cdef class _rowiternode(MDFIterator):
    cdef MDFNode _owner_node
    cdef MDFNode _index_node
    cdef object _data
    cdef object _initial_data
    cdef object _appended_data
    cdef int _appended_data_index
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

    cdef _set_data(self, data, bint reset)
    cdef bint _advance_block(self)
    cdef _get_iterator(self, data, bint advance)
    cdef _next_dataframe(self)
    cdef _next_widepanel(self)
    cdef _next_series(self)

    cdef _append(self, data, MDFContext ctx)
    cdef _get_all_values(self, MDFContext ctx)

    cpdef next(self)
    cpdef send(self, value)
