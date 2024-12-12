from unittest import TestCase

from stock_utils.stock_utils import get_data, create_train_data


class Test(TestCase):
    def test_get_data(self):
        res = get_data("300001.SZ", "20230101", "20231230", 10)
        print(res)
        self.assertEqual(True, False)

    def test_create_train_data(self):
        res = create_train_data("300001.SZ", "20230101", "20231230", 10)
        print(res)
        self.assertEqual(True, False)