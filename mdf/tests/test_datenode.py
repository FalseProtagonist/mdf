from mdf import (
    MDFContext,
    evalnode,
    run,
    now
)

from datetime import datetime
import pandas as pa
import numpy as np
import unittest
import logging

# this is necessary to stop namespace from looking
# too far up the stack as it looks for the first frame
# not in the mdf package

__package__ = None

_logger = logging.getLogger(__name__)


daterange = pa.Index([
    datetime(2016, 10, 10, 10, 0, 0),
    datetime(2016, 10, 10, 14, 0, 0),
    datetime(2016, 10, 10, 18, 0, 0),
    datetime(2016, 10, 10, 20, 0, 0),
    datetime(2016, 10, 11, 10, 0, 0),
    datetime(2016, 10, 11, 14, 0, 0),
    datetime(2016, 10, 11, 18, 0, 0),
    datetime(2016, 10, 11, 20, 0, 0),
    datetime(2016, 10, 12, 10, 0, 0),
    datetime(2016, 10, 12, 14, 0, 0),
    datetime(2016, 10, 12, 18, 0, 0),
    datetime(2016, 10, 12, 20, 0, 0),
    datetime(2016, 10, 13, 10, 0, 0),
    datetime(2016, 10, 13, 14, 0, 0),
    datetime(2016, 10, 13, 18, 0, 0),
    datetime(2016, 10, 13, 20, 0, 0),
    datetime(2016, 10, 14, 10, 0, 0),
    datetime(2016, 10, 14, 14, 0, 0),
    datetime(2016, 10, 14, 18, 0, 0),
    datetime(2016, 10, 14, 20, 0, 0),
])


now_num_calls = 0

@evalnode
def now_num_calls_node():
    global now_num_calls
    now()
    now_num_calls += 1
    return now()


now_date_num_calls = 0

@evalnode
def now_date_num_calls_node():
    global now_date_num_calls
    now_date_num_calls += 1
    return now.date()


class DateNodeTests(unittest.TestCase):

    def setUp(self):
        self.daterange = daterange
        self.ctx = MDFContext()

    def test_now_node(self):
        self._run(now.queuenode())
        value = self.ctx[now.queuenode()]
        expected = list(daterange)
        self.assertEqual(list(value), expected)

    def test_now_date_node(self):
        self._run(now.date.queuenode())
        value = self.ctx[now.date.queuenode()]
        expected = list(daterange.date)
        self.assertEqual(list(value), expected)

    def test_now_delayed(self):
        test_node = now.delaynode(periods=1, initial_value=datetime(1990, 1, 1))
        self._run(test_node.queuenode())

        value = self.ctx[test_node.queuenode()]
        expected = [datetime(1990, 1, 1)] + list(daterange)[:-1]
        self.assertEqual(list(value), expected)

        value = self.ctx._get_node_all_values(test_node)
        self.assertEqual(list(value), expected)

    def test_now_date_delayed(self):
        test_node = now.date.delaynode(periods=1, initial_value=datetime(1990, 1, 1))
        self._run(test_node.queuenode())

        value = self.ctx[test_node.queuenode()]
        expected = [datetime(1990, 1, 1)] + list(daterange.date)[:-1]
        self.assertEqual(list(value), expected)

        value = self.ctx._get_node_all_values(test_node)
        self.assertEqual(list(value), expected)

    def test_now_node_all_data(self):
        self._run()
        value = self.ctx._get_node_all_values(now)
        expected = list(daterange)
        self.assertEqual(list(value), expected)

    def test_now_date_node_all_data(self):
        self._run()
        value = self.ctx._get_node_all_values(now.date)
        expected = list(daterange.date)
        self.assertEqual(list(value), expected)

    def test_now_dependent_node_called_every_timestep(self):
        global now_num_calls
        now_num_calls = 0
        self._run(now_num_calls_node.queuenode())
        expected = len(daterange)
        self.assertEqual(now_num_calls, expected)

        expected = list(daterange)
        actual = list(self.ctx[now_num_calls_node.queuenode()])
        self.assertEqual(actual, expected)

    def test_date_dependent_node_called_every_date(self):
        global now_date_num_calls
        now_date_num_calls = 0
        self._run(now_date_num_calls_node.queuenode())
        expected = len(np.unique(daterange.date))
        self.assertEqual(now_date_num_calls, expected)

        expected = list(daterange.date)
        actual = list(self.ctx[now_date_num_calls_node.queuenode()])
        self.assertEqual(actual, expected)

    def _run_for_daterange(self, date_range, *nodes):
        results = []
        def callback(date, ctx):
            results.append([ctx[node] for node in nodes])
        run(date_range, [callback], ctx=self.ctx)
        return results

    def _run(self, *nodes):
        return self._run_for_daterange(self.daterange, *nodes)


