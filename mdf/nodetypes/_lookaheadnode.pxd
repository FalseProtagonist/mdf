from ._nodetypes cimport MDFCustomNode
from ..context cimport MDFContext, _get_current_context
from ..nodes cimport MDFNode


cdef class MDFLookAheadNode(MDFCustomNode):
    cpdef on_set_date(self, MDFContext ctx, date)


cpdef _lookaheadnode(value_unused, MDFLookAheadNode owner_node, periods, MDFNode filter_node=?, offset=?)
