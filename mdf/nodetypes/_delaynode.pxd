from ._nodetypes cimport MDFCustomNode, MDFCustomNodeIterator
from ._datanode cimport _rowiternode
from ..nodes cimport MDFIterator, NodeState
from ..context cimport MDFContext


cdef class MDFDelayNode(MDFCustomNode):
    cdef object _dn_func
    cdef dict _dn_per_ctx_data
    cdef int _dn_is_generator
    cdef int _dn_lazy

    cpdef _dn_get_prev_value(self)

    # MDFCustomNode overrides
    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)
    cpdef dict _get_bind_kwargs(self, owner)

cdef class _delaynode(MDFIterator):
    cdef int lazy
    cdef int skip_nans
    cdef object queue
    cdef bint has_rowiter
    cdef _rowiternode rowiter

    cpdef next(self)
    cpdef send(self, value)

    cdef _get_all_values(self, MDFContext ctx)