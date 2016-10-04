from ._nodetypes cimport MDFCustomNode
from ..context cimport MDFContext, _get_current_context
from ..nodes cimport MDFNode


cdef class MDFLookAheadNode(MDFCustomNode):
    pass


cpdef _lookaheadnode(value_unused,
                     MDFLookAheadNode owner_node,
                     periods=?,
                     MDFNode until=?,
                     MDFNode filter_node=?,
                     offset=?,
                     strict_until=?)
