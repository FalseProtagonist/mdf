from ._nodetypes cimport MDFCustomNode, MDFCustomNodeIterator
from ._datanode cimport _rowiternode
from ..nodes cimport MDFIterator, NodeState
from ..context cimport MDFContext, _get_current_context


cdef class MDFNanSumNode(MDFCustomNode):
    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)


cdef class _nansumnode(MDFIterator):
    cdef object accum
    cdef double accum_f
    cdef bint accum_initialized
    cdef int is_float
    cdef bint has_rowiter
    cdef _rowiternode rowiter

    cpdef next(self)
    cpdef send(self, value)

    cdef inline _send_vector(self, value)
    cdef inline double _send_float(self, double value)
