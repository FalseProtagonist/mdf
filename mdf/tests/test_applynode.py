from mdf import (
    MDFContext,
    varnode,
    datanode,
    run
)

from datetime import datetime
import pandas as pa
import unittest
import logging

# this is necessary to stop namespace from looking
# too far up the stack as it looks for the first frame
# not in the mdf package

__package__ = None

_logger = logging.getLogger(__name__)


daterange = pa.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
price_series = pa.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], index=daterange, dtype=float)
price_series_node = datanode(data=price_series)

const_varnode = varnode(default=20.0)


def apply_func(a, b):
    return a + b


class ApplyNodeTests(unittest.TestCase):

    def setUp(self):
        self.daterange = pa.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
        self.ctx = MDFContext()

    def test_applynode(self):
        test_node = price_series_node.applynode(apply_func, args=(10.0,))
        self._run(test_node.queuenode())

        expected = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        actual = list(self.ctx[test_node.queuenode()])

        self.assertEqual(expected, actual)

    def test_applynode_with_varnode(self):
        test_node = price_series_node.applynode(apply_func, args=(const_varnode,))
        self._run(test_node.queuenode())

        expected = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]
        actual = list(self.ctx[test_node.queuenode()])

        self.assertEqual(expected, actual)

    def test_applynode_with_datanode(self):
        test_node = price_series_node.applynode(apply_func, args=(price_series_node,))
        self._run(test_node.queuenode())

        expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        actual = list(self.ctx[test_node.queuenode()])

        self.assertEqual(expected, actual)

    def test_applynode_all_data(self):
        test_node = price_series_node.applynode(apply_func, args=(10.0,))

        self._run()
        actual = list(self.ctx._get_all_values(test_node))
        expected = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]

        self.assertEqual(expected, actual)

    def test_applynode_with_varnode_all_data(self):
        test_node = price_series_node.applynode(apply_func, args=(const_varnode,))

        self._run()
        actual = list(self.ctx._get_all_values(test_node))
        expected = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]

        self.assertEqual(expected, actual)

    def test_applynode_with_datanode_all_data(self):
        test_node = price_series_node.applynode(apply_func, args=(price_series_node,))

        self._run()
        actual = list(self.ctx._get_all_values(test_node))
        expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]

        self.assertEqual(expected, actual)

    def _run_for_daterange(self, date_range, *nodes):
        results = []
        def callback(date, ctx):
            results.append([ctx[node] for node in nodes])
        run(date_range, [callback], ctx=self.ctx)
        return results

    def _run(self, *nodes):
        return self._run_for_daterange(self.daterange, *nodes)

