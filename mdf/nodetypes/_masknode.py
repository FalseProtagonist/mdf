from ._nodetypes import MDFCustomNode, nodetype
import numpy as np



class MDFMaskNode(MDFCustomNode):
    nodetype_args = ["value", "mask", "mask_value"]



def _masknode(value, mask, mask_value=np.nan):
    """
    Masked nodes evaluate to 'mask_value' when the specified 'mask'
    is True and the value of the function or node otherwise.

    e.g.

    @evalnode
    def masked_node():
        return value_node.mask(is_weekend, mask_value=np.nan)

    @evalnode
    def value_node():
        return x

    Returns 'x' on weekdays and np.nan on weekends.
    """
    if mask:
        return mask_value
    return value



# decorators don't work on cythoned classes
masknode = nodetype(_masknode, cls=MDFMaskNode, method="mask")
