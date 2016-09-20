"""
lookaheadnode evaluates a node over a date range or for a number
of periods in the future and returns a pandas series of values.
When looking for a number of periods in the future it does that
for the first timestep only and is constant thereafter.
It's intended use is for small look aheads for seeding moving
average type calculations.
"""
from ._nodetypes import MDFCustomNode, nodetype
from ..context import MDFContext, _get_current_context
from ..nodes import now
from ..common import DIRTY_FLAGS
import pandas as pa
import cython


class MDFLookAheadNode(MDFCustomNode):
    nodetype_args = ["value_unused", "owner_node", "periods", "filter_node", "offset"]

    # don't mark this node as dirty when dependent nodes are dirtied
    # because of changes to the current date.
    dirty_flags_propagate_mask = ~DIRTY_FLAGS.DATETIME

    def on_set_date(self, ctx, date):
        """called just before 'now' is changed"""
        # return True if date is going backwards to indicate we should be marked as dirty
        return ctx.get_date() > date


def _lookaheadnode(value_unused, owner_node, periods, filter_node=None, offset=pa.datetools.BDay()):
    """
    Node type that creates an :py:class:`MDFNode` that returns
    a pandas Series of values of the underlying node for a sequence
    of dates in the future.

    Unlike most other node types this shouldn't be used as a decorator, but instead
    should only be used via the method syntax for node types, (see :ref:`nodetype_method_syntax`)
    e.g.::

        future_values = some_node.lookahead(periods=10)

    This would get the next 10 values of ``some_node`` after the current date. Once
    evaluated it won't be re-evaluated as time moves forwards; it's always the first
    set of future observations. It is intended to be used sparingly for seeding
    moving average calculations or other calculations that need some initial value
    based on the first few samples of another node.

    The dates start with the current context date (i.e. :py:func:`now`) and is
    incremented by the optional argument `offset` which defaults to weekdays
    (see :py:class:`pandas.datetools.BDay`).

    :param int periods: the total number of observations to collect, excluding any that are ignored due
                        to any filter being used.

    :param offset: date offset object (e.g. datetime timedelta or pandas date offset) to use to
                   increment the date for each sample point.

    :param filter: optional node that if specified should evaluate to True if an observation is to
                   be included, or False otherwise.

    """
    assert owner_node.base_node is not None, \
        "lookahead nodes must be called via the lookahead or lookaheadnode methods on another node"

    ctx = cython.declare(MDFContext)
    shifted_ctx = cython.declare(MDFContext)

    # create a shifted context from the current context shifted by date
    ctx = _get_current_context()
    date = ctx.get_date()
    shifted_ctx = ctx.shift({now : date})

    # collect results from the shifted context
    count = cython.declare(int, 0)
    values = cython.declare(list, [])
    dates = cython.declare(list, [])

    try:
        while count < periods:
            shifted_ctx.set_date(date)
            date += offset

            if filter_node is not None:
                if not shifted_ctx.get_value(filter_node):
                    continue

            value = shifted_ctx.get_value(owner_node.base_node)
            values.append(value)
            dates.append(shifted_ctx.get_date())
            count += 1
    finally:
        # removed any cached values from the context since they won't be needed again
        # and would otherwise just be taking up memory.
        shifted_ctx.clear()

    if count > 0 and isinstance(values[0], pa.Series):
        return pa.DataFrame(values, index=dates)

    return pa.Series(values, index=dates)


# decorators don't work on cythoned classes
lookaheadnode = nodetype(_lookaheadnode, cls=MDFLookAheadNode, method="lookahead")


