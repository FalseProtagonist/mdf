from mdf import MDFContext, evalnode
from datetime import datetime
import pandas as pd
import unittest
import logging

# this is necessary to stop namespace from looking
# too far up the stack as it looks for the first frame
# not in the mdf package

__package__ = None

_logger = logging.getLogger(__name__)


@evalnode
def B():
    accum = 0
    while True:
        yield accum
        accum += 1


@evalnode
def lookahead_until():
    return B() > 2



class NodeTest(unittest.TestCase):
    def setUp(self):
        self.daterange = pd.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
        self.ctx = MDFContext()

    def test_lookahead_node(self):
        B_queue = B.queuenode()
        B_lookahead = B.lookaheadnode(periods=len(self.daterange))

        self.ctx.set_date(self.daterange[0])
        actual = self.ctx[B_lookahead]

        self._run(B_queue)
        expected = self.ctx[B_queue]

        self.assertEquals(actual.values.tolist(), list(expected))
        self.assertEquals(actual.index.tolist(), list(self.daterange))

    def test_lookahead_node_conditional(self):
        B_queue = B.queuenode()
        B_lookahead = B.lookaheadnode(until=lookahead_until)

        self.ctx.set_date(self.daterange[0])
        actual = self.ctx[B_lookahead]

        self._run_for_daterange(self.daterange[:3], B_queue)
        expected = self.ctx[B_queue]

        self.assertEquals(actual.values.tolist(), list(expected))

    def test_lookahead_node_conditional_nonstrict(self):
        B_queue = B.queuenode()
        B_lookahead = B.lookaheadnode(until=lookahead_until, strict_until=False)

        self.ctx.set_date(self.daterange[0])
        actual = self.ctx[B_lookahead]

        self._run_for_daterange(self.daterange[:4], B_queue)
        expected = self.ctx[B_queue]

        self.assertEquals(actual.values.tolist(), list(expected))



    def _run_for_daterange(self, date_range, *nodes):
        for t in date_range:
            self.ctx.set_date(t)
            for node in nodes:
                self.ctx[node]

    def _run(self, *nodes):
        self._run_for_daterange(self.daterange, *nodes)
