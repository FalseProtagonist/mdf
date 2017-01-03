from mdf import (
    MDFContext,
    evalnode,
    varnode,
    nodetype,
    run)
import datetime as dt
import unittest
import logging


# this is necessary to stop namespace from looking
# too far up the stack as it looks for the first frame
# not in the mdf package

__package__ = None

_logger = logging.getLogger(__name__)


# basic nodetype that simply multiplies one node by another (or by a constant)
@nodetype(method="test_nodetype", node_method="test_nodetype_node")
def test_nodetype(value, multiplier):
    return value * multiplier


# 'multiplier_node' is passed as a node, rather than a value
@nodetype(method="test_nodetype_chain", node_method="test_nodetype_chain_node")
def test_nodetype_chain(value, multiplier_node, multiplier):
    return value * multiplier_node.test_nodetype(multiplier)


# 'value_node' is passed as a node, rather than a value
@nodetype(method="test_nodetype_chain_2", node_method="test_nodetype_chain_2_node")
def test_nodetype_chain_2(value_node, multiplier):
    return value_node.test_nodetype_chain(value_node, multiplier)


# return 2 * 2 * 5
@test_nodetype_chain_2(multiplier=5)
def test_node():
    return 2


class NodeExtensionsTest(unittest.TestCase):

    def setUp(self):
        self.ctx = MDFContext()

    def test_nodetype_value_node(self):
        # test node = 2 * 2 * 5
        expected = 20
        actual = self.ctx[test_node]

        self.assertEqual(actual, expected)

    def test_simple_nodetype_kwarg_value(self):
        a = varnode()
        self.ctx[a] = 100

        # b = a * 5
        b = a.test_nodetype_node(multiplier=5)

        expected = 500
        actual = self.ctx[b]

        self.assertEqual(actual, expected)

    def test_simple_nodetype_arg_value(self):
        a = varnode()
        self.ctx[a] = 100

        # b = a * 5
        b = a.test_nodetype_node(5)

        expected = 500
        actual = self.ctx[b]

        self.assertEqual(actual, expected)

    def test_simple_nodetype_kwarg_node(self):
        a = varnode()
        b = varnode()
        self.ctx[a] = 5
        self.ctx[b] = 10

        # c = a * b = 50
        c = a.test_nodetype_node(multiplier=b)

        expected = 50
        actual = self.ctx[c]

        self.assertEqual(actual, expected)

    def test_simple_nodetype_arg_node(self):
        a = varnode()
        b = varnode()
        self.ctx[a] = 5
        self.ctx[b] = 10

        # c = a * b = 50
        c = a.test_nodetype_node(b)

        expected = 50
        actual = self.ctx[c]

        self.assertEqual(actual, expected)

    def test_nodetype_chain(self):
        a = varnode()
        b = varnode()
        c = varnode()
        self.ctx[a] = 2
        self.ctx[b] = 3

        # c = a * (b * 4)
        c = a.test_nodetype_chain_node(b, 4)

        expected = 24
        actual = self.ctx[c]

        self.assertEqual(actual, expected)

    def test_nodetype_chain_asserts_if_not_node(self):
        a = varnode()
        self.ctx[a] = 2

        # c = a * (2 * 4)  # multiplier_node must be a node
        self.assertRaises(AssertionError, a.test_nodetype_chain_node, 2, 3)

    def test_nodetype_value_node_as_method(self):
        a = varnode()
        self.ctx[a] = 2

        # c = (a * a) * 10
        c = a.test_nodetype_chain_2_node(10)

        expected = 40
        actual = self.ctx[c]

        self.assertEqual(actual, expected)

    def test_nodetype_with_generator(self):
        from collections import deque
        import numpy as np
        import pandas as pd
        from mdf import nodetype, datanode

        @nodetype(method='rolling_func', node_method='rolling_func_node')
        def rolling_func_nodetype(value, func=np.max, window=10, min_periods=10):
            q = deque(maxlen=window)

            while True:
                if value == value:
                    q.append(value)

                if len(q) >= min_periods:
                    value = yield func(q)
                else:
                    value = yield np.nan

        @nodetype(method='breakout', node_method='breakout_node')
        def breakout_nodetype(value_node):
            value_low_0_node = value_node.rolling_func_node()

            while True:
                yield value_low_0_node()

        s = pd.Series(index=pd.date_range(start=pd.Timestamp('20160101'), periods=100), data=np.random.rand(100))

        s_datanode = datanode(data=s)
        bo_node = s_datanode.breakout_node()

        def func(date, ctx):
            ctx[bo_node]

        run(s.index, [func], ctx=self.ctx)

