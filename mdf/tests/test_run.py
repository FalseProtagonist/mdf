from mdf import (
    MDFContext,
    DataFrameBuilder,
    evalnode,
    run
)
import datetime as dt
import pandas as pa
import unittest
import logging


@evalnode
def A():
    i = 0
    while True:
        yield i
        i += 1


# this is necessary to stop namespace from looking
# too far up the stack as it looks for the first frame
# not in the mdf package
__package__ = None
_logger = logging.getLogger(__name__)

class RunnerTest(unittest.TestCase):

    def setUp(self):
        self.daterange = pa.bdate_range(dt.datetime(2016, 1, 1), dt.datetime(2016, 1, 10), freq="1D")
        self.daterange2 = pa.bdate_range(dt.datetime(2016, 1, 11), dt.datetime(2016, 1, 20), freq="1D")
        self.ctx = MDFContext()

    def test_run_reset(self):
        df_builder = DataFrameBuilder(nodes=[A])
        run(self.daterange, [df_builder], ctx=self.ctx, reset=True)

        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        actual = list(df_builder.dataframe["A"])
        self.assertEqual(expected, actual)

        # running again should reset the state of A
        df_builder.clear()
        run(self.daterange2, [df_builder], ctx=self.ctx, reset=True)

        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        actual = list(df_builder.dataframe["A"])
        self.assertEqual(expected, actual)

    def test_run_continue(self):
        df_builder = DataFrameBuilder(nodes=[A])
        run(self.daterange, [df_builder], ctx=self.ctx, reset=True)

        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        actual = list(df_builder.dataframe["A"])
        self.assertEqual(expected, actual)

        # running again should continue without resetting A
        df_builder.clear()
        run(self.daterange2, [df_builder], ctx=self.ctx, reset=False)

        expected = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        actual = list(df_builder.dataframe["A"])
        self.assertEqual(expected, actual)
