from mdf import (
    MDFContext,
    evalnode,
    queuenode,
    delaynode,
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


@evalnode
def initial_value():
    return 0


@delaynode(periods=1, initial_value=initial_value)
def delayed_node():
    i = 1
    while True:
        yield i
        i += 1


@delaynode(periods=1, initial_value=initial_value, lazy=True)
def delayed_node_lazy():
    return delay_test()[-1]


@queuenode
def delay_test():
    return 1 + delayed_node()


@queuenode
def delay_test_lazy():
    return 1 + delayed_node_lazy()


daterange = pa.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
price_series = pa.Series([1.0, 2.0, 2.5, 5.0, 7.5, 3.75, 0.0], index=daterange, dtype=float)
price_series_node = datanode(data=price_series)



class DelayNodeTests(unittest.TestCase):

    def setUp(self):
        self.daterange = pa.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
        self.ctx = MDFContext()

    def test_delaynode(self):
        self._run(delay_test, delay_test_lazy)
        value = self.ctx[delay_test]
        value_lazy = self.ctx[delay_test_lazy]
        self.assertEqual(list(value), list(range(1, len(self.daterange) + 1)))
        self.assertEqual(list(value_lazy), list(range(1, len(self.daterange) + 1)))

    def test_delayed_datanode(self):
        test_node = price_series_node.delaynode(periods=1, initial_value=0.0).queuenode()
        self._run(test_node)

        expected = price_series.shift(1).fillna(value=0.0).tolist()
        actual = list(self.ctx[test_node])

        self.assertEqual(expected, actual)

    def test_chained_delayed_datanode(self):
        test_node = price_series_node \
                        .delaynode(periods=1, initial_value=0.0) \
                        .delaynode(periods=1, initial_value=0.0) \
                        .queuenode()
        self._run(test_node)

        expected = price_series.shift(2).fillna(value=0.0).tolist()
        actual = list(self.ctx[test_node])

        self.assertEqual(expected, actual)

    def _test(self, node, expected_values):
        values = node.queuenode()
        self._run(values)
        actual = self.ctx[values]
        self.assertEquals(list(actual), expected_values)

    def _run_for_daterange(self, date_range, *nodes):
        results = []
        def callback(date, ctx):
            results.append([ctx[node] for node in nodes])
        run(date_range, [callback], ctx=self.ctx)
        return results

    def _run(self, *nodes):
        return self._run_for_daterange(self.daterange, *nodes)


