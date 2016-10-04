from ._nodetypes cimport dict_iteritems
from ._applynode cimport MDFApplyNode
from ..nodes cimport MDFNode, NodeState
from ..context cimport MDFContext


cdef class MDFBinOpNode(MDFApplyNode):
    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)
