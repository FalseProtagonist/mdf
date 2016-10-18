from mdf import (
    DataFrameBuilder,
    MDFContext,
    datanode,
    run
)
import pandas as pa
import numpy as np
import datetime as dt
import unittest
import logging


daterange = pa.bdate_range(dt.datetime(1970, 1, 1), dt.datetime(1970, 1, 10))

df_a = pa.DataFrame([{"A": i, "B": -i} for i in range(len(daterange))], index=daterange, dtype=float)
df_b = pa.DataFrame([{"C": -i, "D": i} for i in range(len(daterange))], index=daterange, dtype=float)
series_a = pa.Series([x * 2 for x in range(len(daterange))], index=daterange)

df_node_a = datanode("df_a", df_a)
df_node_b = datanode("df_b", df_b)
series_node_a = datanode("series_a", series_a)


class DataFrameBuilderTest(unittest.TestCase):

    def setUp(self):
        self.daterange = daterange
        self.ctx = MDFContext()

    def test_dataframe_builder_from_series(self):
        """test building a dataframe from a node that returns a series"""
        builder = DataFrameBuilder(df_node_a)
        self._run(builder)
        self.assertTrue((builder.dataframe == df_a).all().all())

        builder = DataFrameBuilder([df_node_a, df_node_b])
        expected = df_a.join(df_b)
        self._run(builder)
        self.assertTrue((builder.dataframe == expected).all().all())

    def test_dataframe_builder_from_scalar(self):
        """test building a dataframe from a node that returns a scalar value"""
        builder = DataFrameBuilder(series_node_a)
        expected = pa.DataFrame({series_node_a.short_name: series_a})
        self._run(builder)
        self.assertTrue((builder.dataframe == expected).all().all())

    def _run(self, *builders):
        run(self.daterange, builders, ctx=self.ctx)


if __name__ == "__main__":
    # speed test for improving dataframe builder
    import mdf
    import time

    logging.basicConfig()
    mdf.enable_profiling()

    index = pa.date_range(dt.datetime(2001, 1, 1), periods=100000, freq="10min")
    values = pa.DataFrame({chr(x): np.random.random(len(index)) for x in range(10)}, index=index)

    node = datanode("data", data=values)
    builder = DataFrameBuilder(node)

    start_time = time.time()
    ctx = run(index, [builder])
    end_time = time.time()

    ctx.ppstats()

    print("Actual total time: %0.2fs" % (end_time - start_time))
