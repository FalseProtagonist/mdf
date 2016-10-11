from ._nodetypes cimport MDFCustomNode
from ..nodes cimport MDFNode, NodeState
from ..context cimport MDFContext


cdef class MDFMaskNode(MDFCustomNode):

    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)


cpdef _masknode(value, mask, mask_value=?)
