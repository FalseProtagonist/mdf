"""
A masked node returns a fixed value (e.g. nan) when a mask is True
or the underlying value otherwise.

e.g.

@evalnode
def masked_node():
    return value_node.mask(is_weekend, mask_value=np.nan)

@evalnode
def value_node():
    return x

Returns 'x' on weekdays and np.nan on weekends.
"""
from mdf import MDFContext, varnode, run
import mdf
from datetime import datetime
import unittest
import logging
import pandas as pa
import numpy as np

# this is necessary to stop namespace from looking
# too far up the stack as it looks for the first frame
# not in the mdf package
__package__ = None

_logger = logging.getLogger(__name__)

index = pa.Index([datetime(2016, 9, 1),
                  datetime(2016, 9, 2),
                  datetime(2016, 9, 3),
                  datetime(2016, 9, 4),  # Saturday
                  datetime(2016, 9, 5),  # Sunday
                  datetime(2016, 9, 6),
                  datetime(2016, 9, 7),
                  datetime(2016, 9, 8),
                  datetime(2016, 9, 9),
                  datetime(2016, 9, 10),
                  datetime(2016, 9, 11),  # Saturday
                  datetime(2016, 9, 12),  # Sunday
                  datetime(2016, 9, 13)])

data = pa.Series(list(range(len(index))), index=index)
datanode = mdf.datanode("data", data)

df = pa.DataFrame({"A": list(range(len(index)))}, index=index, dtype=float)
dfnode = mdf.datanode("df", df)

mask = pa.Series([x.isoweekday() in (1, 7) for x in index], index=index)
masknode = mdf.datanode("mask", mask)

minus_one = varnode(default=-1.0)


class NodeFilterTest(unittest.TestCase):

    def setUp(self):
        self.ctx = MDFContext(index[0])

    def test_masked_node(self):
        expected_value = [i for i in range(len(index))]
        actual_value = self._run(datanode)
        self.assertEqual(expected_value, actual_value)

        expected_mask = [x.isoweekday() in (1, 7) for x in index]
        actual_mask = self._run(masknode)
        self.assertEqual(expected_mask, actual_mask)

        expected = [-1 if m else i for (i, m) in zip(data, mask)]
        actual = self._run(datanode.masknode(masknode, mask_value=minus_one))
        self.assertEqual(expected, actual)

        # test getting all values returns the same
        actual = self.ctx._get_all_values(datanode.masknode(masknode, mask_value=minus_one))
        self.assertEqual(expected, list(actual))

    def test_masked_df(self):
        expected_value = pa.DataFrame({"A": [i for i in range(len(index))]}, index=index)
        actual_value = pa.DataFrame(self._run(dfnode), index=index)
        self.assertTrue((expected_value == actual_value).all().bool())

        expected_mask = [x.isoweekday() in (1, 7) for x in index]
        actual_mask = self._run(masknode)
        self.assertEqual(expected_mask, actual_mask)

        expected = pa.DataFrame([{"A": -1 if m else i} for (i, m) in zip(data, mask)], index=index)
        actual = pa.DataFrame(self._run(dfnode.masknode(masknode, mask_value=-1)), index=index)
        self.assertTrue((expected == actual).all().bool())

        # test getting all values returns the same
        actual = self.ctx._get_all_values(dfnode.masknode(masknode, mask_value=-1))
        self.assertTrue((expected == actual).all().bool())

    def test_masked_df_mask_is_series(self):
        columns = ['a', 'b', 'c']
        random_data = np.random.rand(len(index), len(columns))
        test_data_df = pa.DataFrame(index=index, columns=columns, data=random_data)
        test_dn = mdf.datanode("test_data", test_data_df)

        mask_df = pa.DataFrame(index=index, columns=columns, data=False)
        mask_df.ix[datetime(2016, 9, 7), 'a'] = True
        mask_df.ix[datetime(2016, 9, 8), 'b'] = True

        mask_dn = mdf.datanode("mask", mask_df)

        expected = test_data_df.copy()
        expected.ix[datetime(2016, 9, 7), 'a'] = -1.0
        expected.ix[datetime(2016, 9, 8), 'b'] = -1.0

        actual = pa.DataFrame(self._run(test_dn.masknode(mask_dn, mask_value=-1.0)), index=index)
        self.assertTrue((expected == actual).all().all())

    def test_masked_df_mask_is_bool(self):
        columns = ['a', 'b', 'c']
        random_data = np.random.rand(len(index), len(columns))
        test_data_df = pa.DataFrame(index=index, columns=columns, data=random_data)
        test_dn = mdf.datanode("test_data", test_data_df)

        mask_s = pa.Series(index=index, data=False)
        mask_s[datetime(2016, 9, 7)] = True
        mask_s[datetime(2016, 9, 9)] = True

        mask_dn = mdf.datanode("mask", mask_s)

        expected = test_data_df.copy()
        expected.ix[datetime(2016, 9, 7), :] = -1.0
        expected.ix[datetime(2016, 9, 9), :] = -1.0

        actual = pa.DataFrame(self._run(test_dn.masknode(mask_dn, mask_value=-1.0)), index=index)
        self.assertTrue((expected == actual).all().all())

    def _run(self, node):
        results = []
        def callback(date, ctx):
            results.append(ctx[node])
        run(index, [callback], ctx=self.ctx)
        return results

