from ..nodes cimport MDFNode, MDFEvalNode, MDFIterator, NodeState
from ..context cimport MDFContext, _get_current_context

cdef dict_iteritems(d)

cdef class MDFCustomNode(MDFEvalNode):
    cdef MDFNode _base_node
    cdef object _base_node_method_name
    cdef object _node_type_func
    cdef MDFNode _value_node
    cdef object _cn_func
    cdef object _category
    cdef int _call_with_filter_node
    cdef int _call_with_filter
    cdef int _call_with_self
    cdef int _call_with_node
    cdef int _call_with_no_value
    cdef object _nodetype_node_kwargs  # set of arg names that should be passed as nodes
    cdef dict _kwargs
    cdef dict _kwnodes
    cdef dict _kwfuncs

    # internal C methods
    cdef inline dict _get_kwargs(self)
    cdef _get_nodetype_func_args(self)

    # MDFNode overrides
    cdef _get_all_values(self, MDFContext ctx, NodeState node_state)
    cdef _update_all_values(self, MDFContext ctx, NodeState node_state)
    cpdef dict _get_bind_kwargs(self, owner)

    # protected python methods
    cpdef _cn_eval_func(self)
    cpdef _cn_get_all_values(self, MDFContext ctx, NodeState node_state)
    cpdef _cn_update_all_values(self, MDFContext ctx, NodeState node_state)


cdef class MDFCustomNodeIterator(MDFIterator):
    cdef MDFCustomNode custom_node
    cdef object func
    cdef object node_type_func
    cdef object _value_generator
    cdef int is_generator
    cdef int node_type_is_generator
    cdef object _node_type_generator
    cdef bint _node_type_generator_called

    cdef object _get_input_value(self)
    cdef object _get_node_type_generator(self, value)

    cdef object get_value_generator(self)
    cdef object get_node_type_generator(self)
