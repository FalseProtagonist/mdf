from ._nodetypes cimport MDFCustomNode, dict_iteritems
from ..nodes cimport MDFNode, NodeState
from ..context cimport MDFContext
cimport numpy as np


cdef class MDFBinOpNode(MDFCustomNode):
    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)


cpdef _binopnode(MDFNode value_node, op, str op_name, rhs=?)
