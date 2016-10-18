from mdf import (
    MDFContext,
    evalnode,
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

daterange = pa.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
price_series = pa.Series([1.0, 2.0, 2.5, 5.0, 7.5, 3.75, 0.0], index=daterange, dtype=float)
price_series_node = datanode("price_series_node", price_series)

expected_geom_returns = [0.0, 1.0, 0.25, 1.0, 0.5, -0.5, -1.0]
expected_arth_returns = [0.0, 1.0, 0.5, 2.5, 2.5, -3.75, -3.75]

filter_series = pa.Series([True, True, False, True, True, True, True], index=daterange, dtype=float)
filter_node = datanode("filter_node", filter_series)

expected_filtered_geom_returns = [0.0, 1.0, 1.0, 1.5, 0.5, -0.5, -1.0]
expected_filtered_arth_returns = [0.0, 1.0, 1.0, 3.0, 2.5, -3.75, -3.75]


@evalnode
def price_generator():
    for px in price_series:
        yield px


class ReturnsNodeTest(unittest.TestCase):
    def setUp(self):
        self.daterange = daterange
        self.ctx = MDFContext()

    def test_returns(self):
        self._test(price_series_node.returnsnode(), expected_geom_returns)

    def test_filtered_returns(self):
        self._test(price_series_node.returnsnode(filter=filter_node), expected_filtered_geom_returns)

    def test_arithmetic_returns(self):
        self._test(price_series_node.returnsnode(use_diff=True), expected_arth_returns)

    def test_filtered_arithmetic_returns(self):
        self._test(price_series_node.returnsnode(use_diff=True, filter=filter_node), expected_filtered_arth_returns)

    def test_returns_gen(self):
        self._test(price_generator.returnsnode(), expected_geom_returns)

    def test_filtered_returns_gen(self):
        self._test(price_generator.returnsnode(filter=filter_node), expected_filtered_geom_returns)

    def test_arithmetic_returns_gen(self):
        self._test(price_generator.returnsnode(use_diff=True), expected_arth_returns)

    def test_filtered_arithmetic_returns_gen(self):
        self._test(price_generator.returnsnode(use_diff=True, filter=filter_node), expected_filtered_arth_returns)

    def test_get_all_data(self):
        self._run()

        actual = self.ctx._get_all_values(price_series_node.returnsnode())
        expected = pa.Series(expected_geom_returns, index=self.daterange)
        self.assertTrue((actual == expected).all().all())

        actual = self.ctx._get_all_values(price_series_node.returnsnode(use_diff=True))
        expected = pa.Series(expected_arth_returns, index=self.daterange)
        self.assertTrue((actual == expected).all().all())

        actual = self.ctx._get_all_values(price_generator.returnsnode())
        self.assertIsNone(actual)

    def test_append_data(self):
        self._run()

        node = price_series_node.returnsnode()

        actual = self.ctx._get_all_values(node)
        expected = pa.Series(expected_geom_returns, index=self.daterange)
        self.assertTrue((actual == expected).all().all())

        # construct some extra data to append to the price series
        extra_dates = pa.bdate_range(self.daterange[-1] + pa.Timedelta("1D"),
                                     self.daterange[-1] + pa.Timedelta("7D"))
        extra_data = pa.Series(np.random.random(len(extra_dates)), index=extra_dates)

        price_series_node.append(extra_data, self.ctx)

        all_prices = pa.concat([price_series, extra_data])
        all_returns = ((all_prices / all_prices.shift(1)) - 1.0).fillna(value=0.0)

        # advance the context through the extra dates
        extras = self._run_for_daterange(extra_dates, [node], reset=False)

        actual = pa.Series([x[0] for x in extras], index=extra_dates)
        expected = all_returns.reindex(extra_dates)
        self.assertTrue((actual.round(9) == expected.round(9)).all().all())

        # check getting all data works too
        actual = self.ctx._get_all_values(node)
        expected = all_returns
        self.assertTrue((actual.round(9) == expected.round(9)).all().all())

    def _test(self, node, expected_values):
        values = node.queuenode()
        self._run(values)
        actual = self.ctx[values]
        self.assertEquals(list(actual), expected_values)

    def _run_for_daterange(self, date_range, nodes, reset=True):
        results = []
        def callback(date, ctx):
            results.append([ctx[node] for node in nodes])
        run(date_range, [callback], ctx=self.ctx, reset=reset)
        return results

    def _run(self, *nodes):
        return self._run_for_daterange(self.daterange, nodes)
