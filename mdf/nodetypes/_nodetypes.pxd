from ..nodes cimport MDFNode, MDFEvalNode, MDFIterator
from ..context cimport MDFContext

cdef dict_iteritems(d)

cdef class MDFCustomNode(MDFEvalNode):
    cdef MDFNode _base_node
    cdef object _base_node_method_name
    cdef object _node_type_func
    cdef object _cn_func
    cdef object _category
    cdef int _call_with_filter_node
    cdef int _call_with_filter
    cdef int _call_with_self
    cdef int _call_with_no_value
    cdef dict _kwargs
    cdef dict _kwnodes
    cdef dict _kwfuncs

    # internal C methods
    cdef inline dict _get_kwargs(self)
    cdef _get_nodetype_func_kwargs(self, int remove_special=?)

    # protected python methods
    cpdef _cn_eval_func(self)

cdef class MDFCustomNodeIterator(MDFIterator):
    cdef MDFCustomNode custom_node
    cdef object func
    cdef object node_type_func
    cdef object value_generator
    cdef int is_generator
    cdef int node_type_is_generator
    cdef object node_type_generator

