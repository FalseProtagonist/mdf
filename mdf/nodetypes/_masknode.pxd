from ._nodetypes cimport MDFCustomNode


cdef class MDFMaskNode(MDFCustomNode):
    pass


cpdef _masknode(value, mask, mask_value=?)
