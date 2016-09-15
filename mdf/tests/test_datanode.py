from mdf import MDFContext, datanode, DIRTY_FLAGS, varnode, now
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

            # set this to some node type so that it can be overridden in the initializer
            data = varnode

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

