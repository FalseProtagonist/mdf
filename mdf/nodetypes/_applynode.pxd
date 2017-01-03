from ._nodetypes cimport MDFCustomNode, dict_iteritems
from ..nodes cimport MDFNode, NodeState
from ..context cimport MDFContext


cdef class MDFApplyNode(MDFCustomNode):
    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)
