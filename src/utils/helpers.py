import pandas as pd

from src.data_storage.data_saver import DataSaver


def get_ts_code(stock_code:str)->str:
        stock_code_arr = stock_code.split('.')
        if len(stock_code_arr) != 2:
            return ''
        
        if stock_code_arr[0] == 'SZ' or stock_code_arr[0] == 'SH' or stock_code_arr[0] == 'BJ':
            stock_code = stock_code_arr[1]
        else:
            stock_code = stock_code_arr[0]
        
        return stock_code

def get_new_trade_date(data_saver:DataSaver, table_name:str, ts_code:str, start_date:str)->str:
    # 先从数据表里面查询最新的时间，然后从最新的时间开始获取数据
    latest_trade_date:str = data_saver.read_latest_trade_date(table_name, ts_code)
    if len(latest_trade_date) > 0:
        start_date = latest_trade_date
        trade_date = pd.to_datetime(start_date) + pd.DateOffset(days=1)
        trade_date = trade_date.strftime('%Y%m%d')
        return trade_date
    else:
        return start_date
