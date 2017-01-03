from mdf import (
    MDFContext,
    evalnode,
    queuenode,
    ffillnode,
    datanode,
    run
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


@evalnode
def initial_value():
    return 0


@ffillnode
def ffilled_node():
    i = 1
    while True:
        yield np.nan if i % 2 == 0 else float(i)
        i += 1


@ffillnode(initial_value=-1.0)
def ffilled_node_initial_value():
    i = 0
    while True:
        yield np.nan if i % 2 == 0 else float(i)
        i += 1


daterange = pa.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
price_series = pa.Series([1.0, np.nan, 2.0, np.nan, 3.0, np.nan, 4.0], index=daterange, dtype=float)
price_series_node = datanode(data=price_series)

price_series_iv = pa.Series([np.nan, 1.0, np.nan, 2.0, np.nan, 3.0, np.nan], index=daterange, dtype=float)
price_series_iv_node = datanode(data=price_series_iv)


class FfillNodeTests(unittest.TestCase):

    def setUp(self):
        self.daterange = pa.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
        self.ctx = MDFContext()

    def test_ffillnode(self):
        self._run(ffilled_node.queuenode(), ffilled_node_initial_value.queuenode())
        value = self.ctx[ffilled_node.queuenode()]
        value_iv = self.ctx[ffilled_node_initial_value.queuenode()]
        expected = [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0]
        expected_iv = [-1.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0]
        self.assertEqual(list(value), expected)
        self.assertEqual(list(value_iv), expected_iv)

    def test_chained_ffill_datanode(self):
        test_node = price_series_node.ffillnode().queuenode()
        self._run(test_node)

        expected = price_series.fillna(method="ffill").tolist()
        actual = list(self.ctx[test_node])

        self.assertEqual(expected, actual)

    def test_chained_ffill_initial_value_datanode(self):
        test_node = price_series_iv_node.ffillnode(initial_value=-1.0).queuenode()
        self._run(test_node)

        expected = price_series_iv.fillna(method="ffill").fillna(value=-1.0).tolist()
        actual = list(self.ctx[test_node])

        self.assertEqual(expected, actual)

    def _run_for_daterange(self, date_range, *nodes):
        results = []
        def callback(date, ctx):
            results.append([ctx[node] for node in nodes])
        run(date_range, [callback], ctx=self.ctx)
        return results

    def _run(self, *nodes):
        return self._run_for_daterange(self.daterange, *nodes)


