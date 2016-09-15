from mdf import MDFContext, datanode, DIRTY_FLAGS, now, filternode
import datetime as dt
import pandas as pd
import numpy as np
import unittest


class NodeTest(unittest.TestCase):

    def setUp(self):
        self.daterange = pd.bdate_range(dt.datetime(1970, 1, 1), dt.datetime(1970, 1, 10))
        self.daterange2 = pd.bdate_range(dt.datetime(1970, 1, 11), dt.datetime(1970, 1, 20))
        self.daterange3 = pd.bdate_range(dt.datetime(1970, 1, 21), dt.datetime(1970, 1, 30))
        self.ctx = MDFContext()

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

    def _run_for_daterange(self, date_range, *nodes):
        for t in date_range:
            self.ctx.set_date(t)
            for node in nodes:
                self.ctx[node]

    def _run(self, *nodes):
        self._run_for_daterange(self.daterange, *nodes)

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
        # construct the initial data
        start = pd.Timestamp('20160802 045000')
        end = pd.Timestamp('20160802 053000')
        dr1 = pd.date_range(start=start, end=end, freq='10Min')

        initial_data = pd.Series(index=dr1, data=[1.31967]*len(dr1))
        end2 = pd.Timestamp('20160802 060000')

        # NB. the datanode filter spans pd.date_range(start=start, end=end2, freq='10Min'), i.e. the whole
        # date range of interest, but only the first and last points are valid (i.e. filter is True)
        data_filter = filternode('test_datanode_append_ffill_filternode', data=pd.Series(index=[start, end2]))
        node = datanode('test_datanode_append_ffill_node', data=initial_data, filter=data_filter)

        ctx = MDFContext()

        # run the clock and collect the output
        actual = []
        for dt in dr1:
            ctx.set_date(dt)
            actual.append(ctx[node])

        # do an update for the single point pd.Timestamp('20160802 055000')
        # NB. there is a gap: we didn't get an update for pd.Timestamp('20160802 054000')
        update_date = pd.Series(index=[pd.Timestamp('20160802 055000')], data=[1.056562])
        node.append(update_date, ctx)

        # continue running the clock until end2
        # NB. 1. missing point 054000 should be ffill - it is
        #     2. updated point 055000 should be ignored (filter is False at this point) and the previous ffill - it is
        #     3. point 060000 is valid (filter is True at this point) and should be ffill from 054000? - but it's np.nan
        dr2 = pd.date_range(start=pd.Timestamp('20160802 054000'), end=end2, freq='10Min')
        for dt in dr2:
            ctx.set_date(dt)
            actual.append(ctx[node])

        self.assertEquals(actual, [1.31967]*8)


