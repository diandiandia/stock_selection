

from src.data_acquisition.akshare_data import AkshareDataFetcher


if __name__ == '__main__':
    fetcher = AkshareDataFetcher('')
    df = fetcher.get_all_stock_codes()
    fetcher.batch_fetch_historical_data(df, '20250101', '20250810')
    fetcher.get_lhb_data(start_date='20250101', end_date='20250810')
    fetcher.get_latest_trade_date('stock_daily', '600006.SH')


