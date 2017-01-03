from mdf import (
    MDFContext,
    evalnode,
    queuenode,
    nansumnode,
    cumprodnode,
    delaynode,
    ffillnode,
    vargroup
)

from datetime import datetime
import pandas as pd
import numpy as np
import unittest
import logging
import operator

# this is necessary to stop namespace from looking
# too far up the stack as it looks for the first frame
# not in the mdf package

__package__ = None

_logger = logging.getLogger(__name__)

params = vargroup(C=None, D=None)

@queuenode
def queue_output():
    return A() + B()

@queuenode
def queue_yield():
    while True:
        yield A() + B()

@evalnode
def queue_filter():
    yield False
    while True:
        yield True

@queuenode(filter=queue_filter)
def queue_filter_test():
    yield "THIS SHOULD NOT BE IN THE QUEUE"
    while True:
        yield 0

@nansumnode()
def nansum_output():
    return A() + sometimes_nan_B()

@cumprodnode()
def cumprod_output():
    return A() + B()

@evalnode
def sometimes_nan_B():
    b = B()
    return np.nan if b % 2 else b

@evalnode
def A():
    return params.C() * params.D()

@evalnode
def B():
    accum = 0

    while True:
        yield accum
        accum += 1

@evalnode
def A_plus_B():
    return A() + B()


@queuenode
def ffill_queue():
    return ffill_test()

@ffillnode
def ffill_test():
    i = 0
    while True:
        yield np.nan if i % 2 else float(i)
        i += 1

@ffillnode
def ffill_array_test():
    return ffill_array_test_not_filled()

@evalnode
def ffill_array_test_not_filled():
    i = 0
    array = np.ndarray((5,), dtype=float)
    array.fill(10.0)
    yield array
    
    array.fill(np.nan)
    while True:
        yield array

class NodeTest(unittest.TestCase):

    def setUp(self):
        self.daterange = pd.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
        self.ctx = MDFContext()
        self.ctx[params.C] = 10
        self.ctx[params.D] = 20

    def test_queuenode(self):
        self._run(queue_output)
        queue = self.ctx[queue_output]
        self.assertEqual(len(queue), len(self.daterange))

    def test_queueyield(self):
        self._run(queue_yield)
        queue = self.ctx[queue_yield]
        self.assertEqual(len(queue), len(self.daterange))

    def test_queue_filter(self):
        self._run(queue_filter_test)
        queue = self.ctx[queue_filter_test]
        self.assertEqual(list(queue), [0] * (len(self.daterange) - 1))
        
    @staticmethod
    def diff_dfs(lhs_df, rhs_df, tolerance):
        diffs = np.abs(lhs_df - rhs_df)
        mask = (diffs > tolerance).values
        mask &= ~(np.isnan(lhs_df) and np.isnan(rhs_df)).values
        mask |= np.isnan(lhs_df).values & ~np.isnan(rhs_df).values
        mask |= np.isnan(rhs_df).values & ~np.isnan(lhs_df).values
        return mask.any()

    def test_nansumnode(self):
        self._run(nansum_output)
        nansum = self.ctx[nansum_output]
        self.assertEqual(nansum, 812)

    def test_cumprodnode(self):
        self._run(cumprod_output)
        cumprod = self.ctx[cumprod_output]
        self.assertEqual(cumprod, 14201189062704000)

    def test_ffillnode(self):
        self._run(ffill_queue)
        value =  self.ctx[ffill_queue]
        self.assertEqual(tuple(value), (0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0))

    def test_ffill_array(self):
        self._run(ffill_array_test)
        value =  self.ctx[ffill_array_test]
        unfilled_value = self.ctx[ffill_array_test_not_filled]
        self.assertTrue(np.isnan(unfilled_value).all())
        self.assertEquals(value.tolist(), [10., 10., 10., 10., 10.])

    def test_apply_node(self):
        actual_node = A.applynode(func=operator.add, args=(B,)).queuenode()
        expected_node = A_plus_B.queuenode()

        self._run(actual_node, expected_node)
        actual = self.ctx[actual_node]
        expected = self.ctx[expected_node]

        self.assertEquals(actual, expected)
        
    def _test(self, node, expected_values):
        values = node.queuenode()
        self._run(values)
        actual = self.ctx[values]
        self.assertEquals(list(actual), expected_values)
        
    def _run_for_daterange(self, date_range, *nodes):
        for t in date_range:
            self.ctx.set_date(t)
            for node in nodes:
                self.ctx[node] 
                
    def _run(self, *nodes):
        self._run_for_daterange(self.daterange, *nodes)


