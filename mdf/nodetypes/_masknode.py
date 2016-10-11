from ._nodetypes import MDFCustomNode, nodetype
from ..nodes import MDFNode
import numpy as np
import pandas as pa
import cython


class MDFMaskNode(MDFCustomNode):
    nodetype_args = ["value_node", "mask", "mask_value"]

    def _cn_get_all_values(self, ctx, node_state):
        # if the value node and mask node have data then we can provide a full set of data.
        value_node = self.value_node
        if self.value_node is None:
            return None

        value_node_df = ctx._get_all_values(value_node)
        if value_node_df is None:
            return

        node_type_kwargs = self.node_type_kwargs
        mask = node_type_kwargs["mask"]
        if not isinstance(mask, MDFNode):
            return

        mask_df = ctx._get_all_values(mask)
        if mask_df is None:
            return

        mask_value = node_type_kwargs.get("mask_value", np.nan)
        if isinstance(mask_value, MDFNode):
            mask_value = ctx._get_all_values(mask_value)
            if mask_value is None:
                return
            mask_value = mask_value[mask_df]

        masked_df = value_node_df.copy()
        masked_df[mask_df] = mask_value
        return masked_df


def _masknode(value_node, mask, mask_value=np.nan):
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
    np_mask = cython.declare(np.ndarray)
    np_value = cython.declare(np.ndarray)
    np_mask_index = cython.declare(np.ndarray)
    np_value_index = cython.declare(np.ndarray)

    value = value_node()
    if isinstance(value, pa.Series):
        if isinstance(mask, pa.Series):
            # if they share the same index use numpy to do the mask (faster than pandas)
            np_mask_index = mask.index.values
            np_value_index = value.index.values
            if (np_mask_index == np_value_index).all():
                np_mask = mask.values
                np_values = value.values.copy()
                np_values[np_mask] = mask_value
                return pa.Series(np_values, index=mask.index)

            # if the mask is fully un-set return the whole value
            if not mask.any():
                return value

            # finally revert to using the pandas mask
            value_copy = value.copy()
            value_copy[mask] = mask_value
            return value_copy

        if mask:
            return pa.Series(mask_value, index=value.index)

    if mask:
        return mask_value
    return value


# decorators don't work on cythoned classes
masknode = nodetype(_masknode, cls=MDFMaskNode, method="mask")
