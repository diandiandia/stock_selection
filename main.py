

from src.data_acquisition.akshare_data import AkshareDataFetcher
from src.data_acquisition.tushare_data import TushareDataFetcher


def akshare_test():
    fetcher = AkshareDataFetcher()
    # df = fetcher.get_all_stock_codes()
    # fetcher.batch_fetch_historical_data(df, '20250101', '20250810')
    fetcher.get_lhb_data(start_date='20250811', end_date='20250812')
    fetcher.get_latest_trade_date('stock_daily', '600006.SH')


def tushare_test():
    fetcher = TushareDataFetcher()
    df = fetcher.get_all_stock_codes()
    # fetcher.batch_fetch_historical_data(df, '20250101', '20250810')
    fetcher.get_lhb_data(start_date='20250811', end_date='20250812')
    fetcher.get_latest_trade_date('stock_daily', '600006.SH')


if __name__ == '__main__':
    tushare_test()
    akshare_test()
    # akshare_fetcher = AkshareDataFetcher()
    # tushare_fetcher = TushareDataFetcher()
    # df = tushare_fetcher.get_all_stock_codes()
    # df = df.head(2)
    # ak_historical_data = akshare_fetcher.batch_fetch_historical_data(df, '20250811', '20250812')
    # tushare_historical_data = tushare_fetcher.batch_fetch_historical_data(df, '20250811', '20250812')
    # print('ak_historical_data')
    # print(ak_historical_data)
    # print('tushare_historical_data')
    # print(tushare_historical_data)






