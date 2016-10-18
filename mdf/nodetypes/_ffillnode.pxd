from ._nodetypes cimport MDFCustomNode, MDFCustomNodeIterator
from ._datanode cimport _rowiternode
from ..nodes cimport MDFNode, MDFIterator, NodeState
from ..context cimport MDFContext, _get_current_context


cdef class MDFForwardFillNode(MDFCustomNode):
    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)
    cpdef _cn_update_all_values(self, MDFContext ctx, NodeState node_state)


cdef class _ffillnode(MDFIterator):
    cdef int is_float
    cdef object initial_value
    cdef double current_value_f
    cdef object current_value
    cdef bint has_rowiter
    cdef _rowiternode rowiter

    cpdef next(self)
    cpdef send(self, value)

    cdef _get_all_values(self, MDFContext ctx)
    cdef _setup_rowiter(self, all_values, MDFNode owner_node)
