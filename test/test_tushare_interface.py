from datetime import datetime
from unittest import TestCase

from stock_utils.tushare_interface import TushareInterface


class TestTushareInterface(TestCase):
    def test_update_csv_data(self):
        tu = TushareInterface()
        tu.update_csv_data(['300001.SZ'], 10)
        new_data_frames = tu.get_data_between_dates("20241211", "20241212", '300001.SZ')
        print(new_data_frames)

    def test_get_stock_price(self):
        tu = TushareInterface()
        tu.get_stock_price("300001.SZ", datetime(2024, 12, 13))
