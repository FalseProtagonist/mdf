from ._nodetypes cimport MDFCustomNode, MDFCustomNodeIterator
from ._datanode cimport _rowiternode
from ..nodes cimport MDFNode, MDFIterator, NodeState
from ..context cimport MDFContext, _get_current_context
cimport numpy as np


cdef class MDFNanSumNode(MDFCustomNode):
    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)
    cpdef _cn_update_all_values(self, MDFContext ctx, NodeState node_state)


cdef class _nansumnode(MDFIterator):
    cdef object accum
    cdef double accum_f
    cdef bint accum_initialized
    cdef int is_float
    cdef bint has_rowiter
    cdef _rowiternode rowiter

    cpdef next(self)
    cpdef send(self, value)

    cdef _setup_rowiter(self, all_values, MDFNode owner_node)
    cdef _get_all_values(self, MDFContext ctx)

    cdef inline _send_vector(self, value)
    cdef inline double _send_float(self, double value)
