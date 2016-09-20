from mdf import MDFContext, datanode, DIRTY_FLAGS, now, filternode, varnode, rowiternode
import datetime as dt
import pandas as pd
import numpy as np
import unittest
import pickle

ffill_datanode_data = varnode()
ffill_datanode_filter = varnode()
ffill_datanode_filternode = filternode('ffill_datanode_filternode', data=ffill_datanode_filter)
ffill_datanode = datanode('ffill_datanode', data=ffill_datanode_data, filter=ffill_datanode_filternode, ffill=True)


class NodeTest(unittest.TestCase):

    def setUp(self):
        self.daterange = pd.bdate_range(dt.datetime(1970, 1, 1), dt.datetime(1970, 1, 10))
        self.daterange2 = pd.bdate_range(dt.datetime(1970, 1, 11), dt.datetime(1970, 1, 20))
        self.daterange3 = pd.bdate_range(dt.datetime(1970, 1, 21), dt.datetime(1970, 1, 30))
        self.ctx = MDFContext()

    def _run_for_daterange(self, date_range, *nodes):
        for t in date_range:
            self.ctx.set_date(t)
            for node in nodes:
                self.ctx[node]

    def _run(self, *nodes):
        self._run_for_daterange(self.daterange, *nodes)

    def test_datanode_ffill(self):
        data = pd.Series(range(len(self.daterange)), self.daterange, dtype=float)
        data = data[[bool(i % 2) for i in range(len(data.index))]]

        expected = data.reindex(self.daterange, method="ffill")
        expected[np.isnan(expected)] = np.inf

        node = datanode("test_datanode_ffill", data, ffill=True, missing_value=np.inf)
        qnode = node.queuenode()

        self._run(qnode)
        value = self.ctx[qnode]

        self.assertEquals(list(value), expected.values.tolist())

    def test_datanode_append_series(self):
        data = pd.Series(range(len(self.daterange)), self.daterange, dtype=float)

        node = datanode("test_datanode_append_series", data)
        qnode = node.queuenode()

        self._run(qnode)
        value = self.ctx[qnode]

        self.assertEquals(list(value), data.values.tolist())

        # append some data to the data node
        data2 = pd.Series(range(len(self.daterange2)), self.daterange2, dtype=float)
        data3 = pd.Series(range(len(self.daterange3)), self.daterange3, dtype=float)
        node.append(data2, ctx=self.ctx)
        node.append(data3, ctx=self.ctx)

        # after appending the data the queuenode should have been marked as having some future data added
        # (which doesn't cause the node to need to be re-evaluated)
        self.assertTrue(node.is_dirty(self.ctx) == DIRTY_FLAGS.FUTURE_DATA)
        self.assertTrue(qnode.is_dirty(self.ctx) == DIRTY_FLAGS.FUTURE_DATA)
        self.assertEquals(list(self.ctx[qnode]), data.values.tolist())

        self._run_for_daterange(self.daterange2, qnode)
        self._run_for_daterange(self.daterange3, qnode)
        value = self.ctx[qnode]

        # after continuing with the extra ranges the queuenode should contain all 3 sets of datas
        self.assertEquals(list(value), data.values.tolist() + data2.values.tolist() + data3.values.tolist())

    def test_datanode_append_dataframe(self):
        data = pd.DataFrame({"A": range(len(self.daterange))}, index=self.daterange, dtype=float)

        node = datanode("test_datanode_append_dataframe", data)
        qnode = node.queuenode()

        self._run(qnode)
        value = self.ctx[qnode]

        self.assertEquals([x["A"] for x in value], data["A"].values.tolist())

        # append some data to the data node
        data2 = pd.DataFrame({"A": range(len(self.daterange2))}, index=self.daterange2, dtype=float)
        data3 = pd.DataFrame({"A": range(len(self.daterange3))}, index=self.daterange3, dtype=float)
        node.append(data2, ctx=self.ctx)
        node.append(data3, ctx=self.ctx)

        # after appending the data the queuenode should have been marked as having some future data added
        # (which doesn't cause the node to need to be re-evaluated)
        self.assertTrue(node.is_dirty(self.ctx) == DIRTY_FLAGS.FUTURE_DATA)
        self.assertTrue(qnode.is_dirty(self.ctx) == DIRTY_FLAGS.FUTURE_DATA)
        self.assertEquals([x["A"] for x in value], data["A"].values.tolist())

        self._run_for_daterange(self.daterange2, qnode)
        self._run_for_daterange(self.daterange3, qnode)
        value = self.ctx[qnode]

        # after continuing with the extra ranges the queuenode should contain all 3 sets of datas
        self.assertEquals([x["A"] for x in value], data["A"].values.tolist() +
                                                      data2["A"].values.tolist() +
                                                      data3["A"].values.tolist())

    def test_datanode_append_wrong_type_raises(self):
        data = pd.Series(range(len(self.daterange)), self.daterange, dtype=float)
        node = datanode("test_datanode_append_wrong_type_raises", data)
        self.assertRaises(Exception, node.append, None, ctx=self.ctx)

    def test_datanode_append_in_past_raises(self):
        data = pd.Series(range(len(self.daterange)), self.daterange, dtype=float)

        node = datanode("test_datanode_append_in_past_raises", data)
        qnode = node.queuenode()

        self._run(qnode)
        value = self.ctx[qnode]

        self.assertEquals(list(value), data.values.tolist())

        # trying to append data where the index is <= now should raise an exception
        data2 = pd.Series(range(len(self.daterange)), self.daterange, dtype=float)
        self.assertRaises(Exception, node.append, data2, ctx=self.ctx)

    def test_datanode_in_class_append(self):
        class X(object):
            def __init__(self, initial_data):
                self.ctx = MDFContext()
                X.data = datanode('data', data=initial_data, index_node=now)

            def __call__(self, update_data):
                X.data.append(update_data, self.ctx)

        initial_data = pd.DataFrame(index=self.daterange, columns=['a'],
                                    data=np.random.rand(len(self.daterange), 1))
        x = X(initial_data)

        # check that initial data is there
        for t in self.daterange:
            x.ctx.set_date(t)
            self.assertEquals(initial_data.ix[t, 'a'], x.ctx[X.data]['a'])

        # do an update
        update_data = pd.DataFrame(index=self.daterange2, columns=['a'],
                                   data=np.random.rand(len(self.daterange2), 1))

        x(update_data)

        # check that update data is there
        for t in self.daterange2:
            x.ctx.set_date(t)
            self.assertEquals(update_data.ix[t, 'a'], x.ctx[X.data]['a'])

    def test_datanode_append_ffill(self):
        # ffill fills forward missing values, not values that are nan
        start1 = pd.Timestamp('20160917 000000')
        end1 = pd.Timestamp('20160917 000900')
        start2 = pd.Timestamp('20160917 001000')
        end2 = pd.Timestamp('20160917 001900')
        ix1 = pd.date_range(start=start1, end=end1, freq='1Min')
        ix2 = pd.date_range(start=start2, end=end2, freq='1Min')
        dr = pd.date_range(start=start1, end=end2, freq='1Min')

        # 0 nan 2 nan 4 nan 6 nan 8 nan
        data1 = pd.Series(index=[x for (i, x) in enumerate(ix1) if i % 2 == 0],
                          data=[i for (i, x) in enumerate(ix1) if i % 2 == 0])

        # Start the second data set with a missing value to check
        # it fills forward from the previous data.
        # nan 1 nan 3 nan 5 nan 7 nan 9
        data2 = pd.Series(index=[x for (i, x) in enumerate(ix2) if i % 2 != 0],
                          data=[i for (i, x) in enumerate(ix2) if i % 2 != 0])

        ctx = MDFContext(ix1[0])
        node = datanode('test_datanode_append_ffill_datanode', data=data1, ffill=True)
        node.append(data2, ctx)

        expected = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 8, 1, 1, 3, 3, 5, 5, 7, 7, 9]
        actual = []

        for d in dr:
            ctx.set_date(d)
            actual.append(ctx[node])

        self.assertEquals(actual, expected)

    def test_datanode_append_ffill_and_filter(self):
        # construct the initial data
        start = pd.Timestamp('20160802 045000')
        end = pd.Timestamp('20160802 053000')
        dr1 = pd.date_range(start=start, end=end, freq='10Min')

        initial_data = pd.Series(index=dr1, data=[1.31967]*len(dr1))
        end2 = pd.Timestamp('20160802 060000')

        # The datanode filter spans pd.date_range(start=start, end=end2, freq='10Min'), i.e. the whole
        # date range of interest, but only the first and last points are valid (i.e. filter is True)
        ctx = MDFContext()
        ctx[ffill_datanode_filter] = pd.Series(index=[start, end2])
        ctx[ffill_datanode_data] = initial_data

        # run the clock and collect the output
        actual = []
        for dt in dr1:
            ctx.set_date(dt)
            actual.append(ctx[ffill_datanode])

        self.assertEquals(actual, ([1.31967] * len(dr1)))

        # do an update for the single point pd.Timestamp('20160802 055000')
        # NB. there is a gap: we didn't get an update for pd.Timestamp('20160802 054000')
        update_date = pd.Series(index=[pd.Timestamp('20160802 055000')], data=[1.056562])
        ffill_datanode.append(update_date, ctx)

        # continue running the clock until end2
        # 1. the value doesn't change until 060000 because filter is false
        # 2. the value at 060000 is forward filled from 055000
        dr2 = pd.date_range(start=pd.Timestamp('20160802 054000'), end=end2, freq='10Min')
        actual = []
        for dt in dr2:
            ctx.set_date(dt)
            actual.append(ctx[ffill_datanode])

        self.assertEquals(actual, ([1.31967] * (len(dr2) - 1)) + [1.056562])

    def test_datanode_pickle(self):
        # ffill fills forward missing values, not values that are nan
        start1 = pd.Timestamp('20160917 000000')
        end1 = pd.Timestamp('20160917 000900')
        start2 = pd.Timestamp('20160917 001000')
        end2 = pd.Timestamp('20160917 001900')
        ix1 = pd.date_range(start=start1, end=end1, freq='1Min')
        ix2 = pd.date_range(start=start2, end=end2, freq='1Min')
        dr1 = pd.date_range(start=start1, end=end1, freq='1Min')
        dr2 = pd.date_range(start=start2, end=end2, freq='1Min')

        # 0 nan 2 nan 4 nan 6 nan 8 nan
        data1 = pd.Series(index=[x for (i, x) in enumerate(ix1) if i % 2 == 0],
                          data=[i for (i, x) in enumerate(ix1) if i % 2 == 0])

        # Start the second data set with a missing value to check
        # it fills forward from the previous data.
        # nan 1 nan 3 nan 5 nan 7 nan 9
        data2 = pd.Series(index=[x for (i, x) in enumerate(ix2) if i % 2 != 0],
                          data=[i for (i, x) in enumerate(ix2) if i % 2 != 0])

        ctx = MDFContext()
        ctx[ffill_datanode_filter] = pd.Series(0, index=pd.date_range(start=start1, end=end2, freq='1Min'))
        ctx[ffill_datanode_data] = data1

        expected1 = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]
        actual1 = []

        for d in dr1:
            ctx.set_date(d)
            actual1.append(ctx[ffill_datanode])

        self.assertEquals(actual1, expected1)

        # add the second chunk of data and serialize
        ffill_datanode.append(data2, ctx)
        x = pickle.dumps(ctx)
        new_ctx = pickle.loads(x)

        # check the new context continues where we left off
        expected2 = [8, 1, 1, 3, 3, 5, 5, 7, 7, 9]
        actual2 = []

        for d in dr2:
            ctx.set_date(d)
            actual2.append(ctx[ffill_datanode])

        self.assertEquals(actual2, expected2)

        # and the pickled context should too
        actual_pickled = []
        for d in dr2:
            new_ctx.set_date(d)
            actual_pickled.append(new_ctx[ffill_datanode])

        self.assertEquals(actual_pickled, expected2)
