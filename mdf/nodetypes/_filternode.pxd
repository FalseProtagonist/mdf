from ._nodetypes cimport MDFCustomNode


cdef class FilteredNode(MDFCustomNode):

    # overridden to just call underlying eval function without modification
    cpdef _cn_eval_func(self)

