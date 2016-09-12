from ._nodetypes import MDFCustomNode, nodetype

#
# This is a convenience nodetype for applying a filter
# to an existing node.
#
# Not to be confused with 'filternode' which turns a dataframe into a node
# suitable for filtering data, ie. the filter kwarg to evalnode(filter=...)
#
class FilteredNode(MDFCustomNode):
    nodetype_kwargs = []


    def _cn_eval_func(self):
        # return the wrapper node value.
        # Filtering will be taken care of by MDFEvalNode._get_value.
        return self._cn_func()


# this never gets called because of the _cn_eval_func override above
def _unused(value):
    raise AssertionError("This function should never be called.")


# this isn't used directly, only to add "filter" and "filternode" methods to nodes
_applyfilternode = nodetype(_unused, cls=FilteredNode, method="filter")
