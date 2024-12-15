import os
from datetime import datetime

import numpy as np
import pandas
import pandas as pd
from scipy.stats import linregress

from daily_line import DailyLine
from data_interface_base import DataInterfaceBase
from reverse_running import ReverseRunning
from tushare_interface import TushareInterface


class LocalCsvInterface(DataInterfaceBase):
    def __init__(self):
        self.realtime_daily_lines = {}
        self.daily_line_dict = {}

    def get_all_realtime_data(self, stock_list):
        tushare = TushareInterface()
        self.realtime_daily_lines.clear()
        self.realtime_daily_lines = tushare.get_all_stock_realtime_lines(stock_list)

    def get_daily_lines(self, code, end_date, back_days):

        # start = time.time()
        # df_basic_filtered = self.data_between_from_csv(code, end_date, back_days)
        df_basic_filtered = self.data_between_from_memory(code, end_date, back_days)
        if df_basic_filtered is None:
            return None
        # print(df_basic_filtered)
        daily_lines = []
        for index, row in df_basic_filtered.iterrows():
            # 提取每一行的数据
            trade_date = str(row['trade_date'])
            open_price = row['open_qfq']
            close_price = row['close_qfq']
            high_price = row['high_qfq']
            low_price = row['low_qfq']
            volume = row['vol']
            turnover_rate_f = row['turnover_rate_f']
            code = row['ts_code']
            average_price = row['weight_avg']
            pre_close = 0
            volume_ratio = 0
            if 'pre_close_qfq' in row:
                pre_close = row['pre_close_qfq']
            if 'volume_ratio' in row:
                volume_ratio = row['volume_ratio']
            max_pct_change = self.change_pct_of_day(high_price, pre_close, low_price)
            up_shadow_pct = self.up_shadow_pct_of_day(high_price, pre_close, close_price, open_price)
            # 创建一个新的DailyLine对象并添加到列表中
            daily_line = DailyLine(trade_date, open_price, close_price, high_price, low_price, volume,
                                   turnover_rate_f,
                                   code, average_price, max_pct_change, up_shadow_pct, volume_ratio)
            daily_lines.append(daily_line)

        today = self.get_today_date()
        if not isinstance(end_date, str):
            end_date = end_date.strftime("%Y%m%d")
        if self.is_a_stock_trading_day(today) and today == end_date and self.is_between_9_30_and_19_00():
            daily_realtime = self.realtime_daily_lines[code]
            daily_lines.append(daily_realtime)
        # end = time.time()
        # print('cost:', end - start)
        return daily_lines

    def get_name(self, code):
        basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
        df_basic = pd.read_csv(basic_csv_path)
        if not df_basic.empty:
            return df_basic.iloc[0]['name']
        return "名字不存在"

    def get_average_price(self, code, start_date, end_date):
        basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
        df_basic = pd.read_csv(basic_csv_path, parse_dates=['trade_date'])
        df_basic_filtered = df_basic.loc[(df_basic['trade_date'] >= start_date) & (df_basic['trade_date'] <= end_date)]
        weight_avg_list = df_basic_filtered['weight_avg'].tolist()
        return weight_avg_list

    def get_close_price_of_day(self, code, date):
        basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
        dtype_dict = {'trade_date': str}
        df = pd.read_csv(basic_csv_path, dtype=dtype_dict)
        row = df[df['trade_date'] == date]
        if not row.empty:
            # 获取close_qfq的值
            close_qfq_value = row['close_qfq'].iloc[0]
            print(f"The close_qfq value for {date} is: {close_qfq_value}")
        else:
            print(f"No data found for the date {date}.")
            close_qfq_value = None
        return close_qfq_value

    def get_before_days_up_times(self, code, end_date, back_days):
        df_basic_filtered = self.data_between_from_memory(code, end_date, back_days)
        up_times = 0
        for index, row in df_basic_filtered.iterrows():
            # 提取每一行的数据
            limit = str(row['limit'])
            if limit == 'U':
                up_times += 1
        return up_times

    def get_first_limit_up_days(self, code, end_date, back_days):
        df_basic_filtered = self.data_between_from_memory(code, end_date, back_days)
        df_basic_filtered = df_basic_filtered.sort_values(by='trade_date', ascending=False)
        # print(df_basic_filtered)
        up_times = 0
        days = 0
        for index, row in df_basic_filtered.iterrows():
            # 提取每一行的数据
            days += 1
            limit = str(row['limit'])
            if limit == 'U':
                return days
        return days

    def get_second_limit_up_days(self, code, end_date, back_days):
        df_basic_filtered = self.data_between_from_memory(code, end_date, back_days)
        df_basic_filtered = df_basic_filtered.sort_values(by='trade_date', ascending=False)
        # print(df_basic_filtered)
        up_times = 0
        up_count = 0
        days = 0
        for index, row in df_basic_filtered.iterrows():
            # 提取每一行的数据
            days += 1
            limit = str(row['limit'])
            if limit == 'U':
                up_count += 1
                if (up_count == 2):
                    return days
        return days

    def past_years_max_up_pct(self, code, end_date, back_days):
        df_basic_filtered = self.data_between_from_memory(code, end_date, back_days)
        # print(df_basic_filtered)
        # print(df_basic_filtered)
        prices = df_basic_filtered['close_qfq'].to_list()

        # print(prices)
        max_increase = 0
        start_index = 0
        end_index = 0

        current_start_index = 0
        current_max_increase = 0

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                current_max_increase = (prices[i] - prices[current_start_index]) / prices[current_start_index] * 100

                if current_max_increase > max_increase:
                    max_increase = current_max_increase
                    start_index = current_start_index
                    end_index = i

            else:
                current_start_index = i
        # print(df_basic_filtered.iloc[start_index]['trade_date'])
        # print(df_basic_filtered.iloc[end_index]['trade_date'])
        return max_increase

    def data_between_from_csv(self, code, end_date, back_days):
        basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
        if not os.path.exists(basic_csv_path):
            return None
        df_basic = pd.read_csv(basic_csv_path, parse_dates=['trade_date'])
        # print(df_basic)
        selected_row = df_basic[df_basic['trade_date'] == end_date]
        if not selected_row.empty:
            index = selected_row.index[0]
            result = df_basic.iloc[index - back_days + 1:index + 1]
            # print(result)
            return result

    def get_before_days_down_times(self, code, end_date, back_days):
        df_basic_filtered = self.data_between_from_csv(code, end_date, back_days)
        down_times = 0
        for index, row in df_basic_filtered.iterrows():
            # 提取每一行的数据
            limit = str(row['limit'])
            if limit == 'D':
                down_times += 1
        return down_times

    def find_sideways_trading(self, code, max_vol, date):
        diff = 0
        # basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
        # df_basic = pd.read_csv(basic_csv_path, parse_dates=['trade_date'])
        df_basic = self.daily_line_dict[code]
        df_basic_filtered = df_basic.loc[(df_basic['trade_date'] < date)]
        df = df_basic_filtered.sort_values(by='trade_date', ascending=False)
        # date = date.strftime('%Y%m%d')
        for index, row in df.iterrows():
            if row['vol'] > max_vol:
                found_date = row['trade_date']
                days = pd.bdate_range(found_date, date)
                diff = len(days)
                return diff
        if diff == 0:
            diff = 300
        return diff

    def get_history_mean_price(self, code, end_date, how_many_days):
        df_basic_filtered = self.data_between_from_csv(code, end_date, how_many_days)
        mean = df_basic_filtered['close_qfq'].mean()
        return round(mean, 3)

    def get_history_close_price(self, code, end_date, how_many_days):
        df = self.data_between_from_csv(code, end_date, how_many_days)
        # print(df)
        ts_code_list = df['close_qfq'].to_list()
        return ts_code_list

    def is_limit_down(self, code, date):
        # basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
        #
        # dtype_dict = {'trade_date': str}
        # df = pd.read_csv(basic_csv_path, dtype=dtype_dict)
        df = self.daily_line_dict[code]
        row = df[df['trade_date'] == date]
        if not row.empty:
            # 获取close_qfq的值
            if str(row['limit'].values[0]) == 'D':
                return True
            else:
                return False
        else:
            return None

    def get_local_neg_count(self, date):
        # todo 盘中通过接口获取
        basic_csv_path = f'history_neg_count.csv'  # 基础数据的CSV文件路径
        dtype_dict = {'Date': str}
        df = pd.read_csv(basic_csv_path, dtype=dtype_dict)
        row = df[df['Date'] == date]
        if not row.empty:
            # 获取close_qfq的值
            return row['NegCount'].values[0]

        else:
            return 0

    def load_csv_data(self, stock_list):
        index = 0
        for code in stock_list:
            index = index + 1
            basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
            if not os.path.exists(basic_csv_path):
                continue
            df_basic = pd.read_csv(basic_csv_path, parse_dates=['trade_date'])
            self.daily_line_dict[code] = df_basic

            progress = (index) / len(stock_list) * 100
            if index % 30 == 0:
                print(f"Load progress: {progress:.2f}%")

    def data_between_from_memory(self, code, end_date, back_days):
        if code in self.daily_line_dict:
            df_basic = self.daily_line_dict[code]
            trade_dates = pd.bdate_range(end=end_date, periods=back_days)
            start_date = trade_dates[0].strftime('%Y%m%d')
            # print(start_date)
            df_basic_filtered = df_basic.loc[
                (df_basic['trade_date'] >= start_date) & (df_basic['trade_date'] <= end_date)]
            # print(df_basic_filtered)
            return df_basic_filtered
        return None

    def data_between_from_memory2(self, code, start_date, end_date):
        if code in self.daily_line_dict:
            df_basic = self.daily_line_dict[code]
            start_date =start_date
            # print(start_date)
            df_basic_filtered = df_basic.loc[
                (df_basic['trade_date'] >= start_date) & (df_basic['trade_date'] <= end_date)]
            # print(df_basic_filtered)
            return df_basic_filtered
        return None

    def get_circ_mv(self, code, date):
        df = self.daily_line_dict[code]
        date = self.find_pre_data_publish_date(date, 10)
        row = df[df['trade_date'] == date]
        if not row.empty:
            # circ_mv
            return row.iloc[0]['circ_mv']
        # return 100

    def get_circ_mv_2(self, code, date):
        basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
        dtype_dict = {'trade_date': str}
        df = pd.read_csv(basic_csv_path, dtype=dtype_dict)
        # df = self.daily_line_dict[code]
        row = df[df['trade_date'] == date]
        if not row.empty:
            # 获取close_qfq的值
            return round(row.iloc[0]['circ_mv'] / 1e4, 2)
        return None

    def get_buy_price(self, code, date):
        print(code)
        if len(self.daily_line_dict) > 0:
            df = self.daily_line_dict[code]
        else:
            basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
            if not os.path.exists(basic_csv_path):
                return None
            dtype_dict = {'trade_date': str}
            df = pd.read_csv(basic_csv_path, dtype=dtype_dict)
        # 将日期列转换为日期时间类型
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 筛选指定日期之前3天的数据
        filtered_df = df[df['trade_date'] < date]

        recent_two_days = filtered_df.tail(2)
        tow_min_low_price = round(recent_two_days['low_qfq'].min(), 2)
        today = df[df['trade_date'] == date]
        if not today.empty:
            today_open_price = today.iloc[0]['open_qfq']
            if today_open_price < tow_min_low_price:
                return round(today_open_price, 2)
            else:
                return tow_min_low_price
        return tow_min_low_price

    def data_before_days(self, code, end_date, back_days):
        if len(self.daily_line_dict) > 0:
            df_basic = self.daily_line_dict[code]
        else:
            basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
            df_basic = pd.read_csv(basic_csv_path, parse_dates=['trade_date'])
        trade_dates = pd.bdate_range(end=end_date, periods=back_days + 1)
        start_date = trade_dates[0].strftime('%Y%m%d')
        df_basic_filtered = df_basic.loc[(df_basic['trade_date'] >= start_date) & (df_basic['trade_date'] < end_date)]
        return df_basic_filtered

    def data_of_days_include_end_date(self, code, end_date, back_days):
        basic_csv_path = f'data/{code}_daily_data.csv'  # 基础数据的CSV文件路径
        df_basic = pd.read_csv(basic_csv_path, parse_dates=['trade_date'])
        trade_dates = pd.bdate_range(end=end_date, periods=back_days)
        start_date = trade_dates[0].strftime('%Y%m%d')
        df_basic_filtered = df_basic.loc[(df_basic['trade_date'] >= start_date) & (df_basic['trade_date'] <= end_date)]
        return df_basic_filtered

    def average_turnover_rate_of(self, code, end_date, days):
        df = self.data_before_days(code, end_date, days)
        if len(df) == 0:
            return 0.0
        sum_turnover_rate = 0
        for index, row in df.iterrows():
            sum_turnover_rate += row['turnover_rate_f']
        average = sum_turnover_rate / len(df)
        return average

    def get_next_three_days_positive_return(self, date, stock_list):
        result = []
        reverse_running = ReverseRunning(self)
        for stock in stock_list:
            # print(stock)
            buy_price = self.get_buy_price(stock, date)
            if buy_price is None:
                continue
            revenue = reverse_running.rate_of_return(stock, date, buy_price, 3)
            # print(revenue)
            if revenue > 0.05:
                result.append(stock)
        return result

    def get_next_three_days_positive_return_more_than_5_percent(self, stock, date):
        reverse_running = ReverseRunning(self)
        buy_price = self.get_buy_price(stock, date)
        if buy_price is None:
            revenue = 0.0
        revenue = reverse_running.rate_of_return(stock, date, buy_price, 3)
        print(revenue)
        return revenue

    def moving_average(self, prices, window=10):
        return np.convolve(prices, np.ones(window) / window, 'valid')

    def slope_of_ma10(self, code, end_date, days):
        df = self.data_of_days_include_end_date(code, end_date, days)
        if len(df) == 0:
            print(code)
            return 0
        closes = df['close_qfq'].to_list()
        # print(closes)
        ma10 = self.moving_average(closes, 10)
        # print(ma10)
        days = np.arange(len(ma10))  # 创建一个天数数组，与MA10对齐
        slope, intercept, r_value, p_value, std_err = linregress(days, ma10)
        # print(slope)
        slope, intercept = np.polyfit(days, ma10, 1)
        # print(slope)

        # 给定数据
        data = np.array(ma10)

        # 创建与数据对应的x轴（索引）
        x = np.arange(len(data))

        # 使用numpy的polyfit进行线性拟合，返回斜率和截距
        slope, intercept = np.polyfit(x, data, 1)
        return round(slope, 2)

        # 计算拟合的y值
        fitted_line = slope * x + intercept

        # 打印斜率和截距
        print(f"拟合的直线方程: y = {slope:.4f}x + {intercept:.4f}")

        # 绘制原始数据和拟合直线
        plt.scatter(x, data, color='blue', label='原始数据')
        plt.plot(x, fitted_line, color='red', label='拟合直线')
        plt.xlabel('数据点索引')
        plt.ylabel('数据值')
        plt.title('数据的线性拟合')
        plt.legend()
        plt.show()

    def slope_of_ma5(self, code, end_date, days):
        df = self.data_of_days_include_end_date(code, end_date, days)
        if len(df) == 0:
            print(code)
            return 0
        closes = df['close_qfq'].to_list()
        # print(closes)
        ma5 = self.moving_average(closes, 5)
        # print(ma5)
        days = np.arange(len(ma5))  # 创建一个天数数组，与MA10对齐
        slope, intercept, r_value, p_value, std_err = linregress(days, ma5)
        # print(slope)
        slope, intercept = np.polyfit(days, ma5, 1)
        # print(slope)
        return round(slope, 2)

        # 给定数据
        data = np.array(ma5)

        # 创建与数据对应的x轴（索引）
        x = np.arange(len(data))

        # 使用numpy的polyfit进行线性拟合，返回斜率和截距
        slope, intercept = np.polyfit(x, data, 1)

        # 计算拟合的y值
        fitted_line = slope * x + intercept

        # 打印斜率和截距
        print(f"拟合的直线方程: y = {slope:.4f}x + {intercept:.4f}")

        # 绘制原始数据和拟合直线
        plt.scatter(x, data, color='blue', label='原始数据')
        plt.plot(x, fitted_line, color='red', label='拟合直线')
        plt.xlabel('数据点索引')
        plt.ylabel('数据值')
        plt.title('数据的线性拟合')
        plt.legend()
        plt.show()

    def thirty_positive_count(self, code, date, days):
        df = self.data_between_from_memory(code, date, days)
        # date = date.strftime('%Y%m%d')
        positive_count = 0
        for index, row in df.iterrows():
            if row['close_qfq'] > row['open_qfq']:
                positive_count += 1

        return positive_count
