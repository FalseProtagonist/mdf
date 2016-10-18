from mdf import (
    MDFContext,
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
price_series = pa.Series([1.0, np.nan, 2.0, np.nan, 3.0, np.nan, 4.0], index=daterange, dtype=float)
price_series_node = datanode(data=price_series)

price_series_iv = pa.Series([np.nan, 1.0, np.nan, 2.0, np.nan, 3.0, np.nan], index=daterange, dtype=float)
price_series_iv_node = datanode(data=price_series_iv)

price_df = pa.DataFrame({"A": price_series, "B": price_series_iv}, index=daterange, dtype=float)
price_df_node = datanode(data=price_df)


class NanSumNodeTests(unittest.TestCase):

    def setUp(self):
        self.daterange = pa.bdate_range(datetime(1970, 1, 1), datetime(1970, 1, 10))
        self.ctx = MDFContext()

    def test_nansum(self):
        self._run(price_series_node.nansumnode().ffillnode(initial_value=0.0).queuenode(),
                  price_series_iv_node.nansumnode().ffillnode(initial_value=0.0).queuenode(),
                  price_df_node.nansumnode().ffillnode(initial_value=0.0).queuenode())

        value = self.ctx[price_series_node.nansumnode().ffillnode(initial_value=0.0).queuenode()]
        expected = price_series.cumsum(skipna=True).fillna(method="ffill")
        self.assertEqual(list(value), list(expected))

        value = self.ctx[price_series_iv_node.nansumnode().ffillnode(initial_value=0.0).queuenode()]
        expected = price_series_iv.cumsum(skipna=True).fillna(method="ffill").fillna(value=0.0)
        self.assertEqual(list(value), list(expected))

        value = self.ctx[price_df_node.nansumnode().ffillnode(initial_value=0.0).queuenode()]
        value_df = pa.DataFrame(list(value), index=self.daterange)
        expected = price_df.cumsum(skipna=True).fillna(method="ffill").fillna(value=0.0)
        self.assertTrue((value_df == expected).all().all())

    def test_nansum_all_data(self):
        self._run()

        value = self.ctx._get_all_values(price_series_node.nansumnode().ffillnode(initial_value=0.0))
        expected = price_series.cumsum(skipna=True).fillna(method="ffill")
        self.assertEqual(list(value), list(expected))

        value = self.ctx._get_all_values(price_series_iv_node.nansumnode().ffillnode(initial_value=0.0))
        expected = price_series_iv.cumsum(skipna=True).fillna(method="ffill").fillna(value=0.0)
        self.assertEqual(list(value), list(expected))

        value = self.ctx._get_all_values(price_df_node.nansumnode().ffillnode(initial_value=0.0))
        expected = price_df.cumsum(skipna=True).fillna(method="ffill").fillna(value=0.0)
        self.assertTrue((value == expected).all().all())

    def test_append_data(self):
        self._run()

        node = price_series_node.nansumnode().ffillnode(initial_value=0.0)

        actual = self.ctx._get_all_values(node)
        expected = price_series.cumsum(skipna=True).fillna(method="ffill")
        self.assertTrue((actual == expected).all().all())

        # construct some extra data to append to the price series
        extra_dates = pa.bdate_range(self.daterange[-1] + pa.Timedelta("1D"),
                                     self.daterange[-1] + pa.Timedelta("7D"))
        extra_data = pa.Series(np.random.random(len(extra_dates)), index=extra_dates)

        price_series_node.append(extra_data, self.ctx)

        # advance the context through the extra dates
        extras = self._run_for_daterange(extra_dates, [node], reset=False)

        actual = pa.Series([x[0] for x in extras], index=extra_dates)
        expected = extra_data.cumsum(skipna=True).fillna(method="ffill") + expected.ix[-1]
        self.assertTrue((actual.round(9) == expected.round(9)).all().all())

        # check getting all data works too
        actual = self.ctx._get_all_values(node)
        expected = pa.concat([price_series, extra_data]).cumsum(skipna=True).fillna(method="ffill")
        self.assertTrue((actual.round(9) == expected.round(9)).all().all())

    def _run_for_daterange(self, date_range, nodes, reset=True):
        results = []
        def callback(date, ctx):
            results.append([ctx[node] for node in nodes])
        run(date_range, [callback], ctx=self.ctx, reset=reset)
        return results

    def _run(self, *nodes):
        return self._run_for_daterange(self.daterange, nodes)


if __name__ == "__main__":
    # speed test for improving nansum
    import mdf
    import time

    logging.basicConfig()
    mdf.enable_profiling()

    index = pa.date_range(datetime(2001, 1, 1), periods=10000, freq="10min")
    values = pa.DataFrame({chr(x): np.random.random(len(index)) for x in range(10)}, index=index)
    values[values > 0.95] = np.nan

    @mdf.evalnode
    def breaks_chaining():
        return 1.0

    # nansum_node can be evaluated as a single vector operation but iterative_node needs to be run each timestep
    data = datanode("data", data=values)
    nansum_node = data.nansumnode()
    iterative_node = (data * breaks_chaining).label("data_times_one").nansumnode()

    start_time = time.time()
    ctx = run(index, [lambda date, ctx: (ctx[nansum_node], ctx[iterative_node])])
    end_time = time.time()

    ctx.ppstats()

    print("Actual total time: %0.2fs" % (end_time - start_time))
