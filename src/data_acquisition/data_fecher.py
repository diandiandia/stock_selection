import pandas as pd
from abc import abstractmethod
from src.data_storage.data_csv_saver import CsvSaver
from src.data_storage.data_sqlite_saver import SqliteSaver
from src.utils.log_helper import LogHelper


class DataFetcher:
    def __init__(self):
        self.logger = LogHelper().get_logger(__name__)
        self.data_saver = SqliteSaver()
        self.login()

    @abstractmethod
    def login(self, token=''):
        pass

    @abstractmethod
    def get_stock_codes_by_symbol(self, symbol: str, save: bool = True) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_all_stock_codes(self, save: bool = True) -> pd.DataFrame:
        pass

    @abstractmethod
    def batch_fetch_historical_data(self, df_stock_codes: pd.DataFrame, start_date: str, end_date: str, save: bool = True) -> list[pd.DataFrame]:
        pass

    @abstractmethod
    def get_history_stock_data(self, stock_code: str, start_date: str, end_date: str, save: bool = True) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_lhb_data(self, start_date, end_date, save: bool = True):
        pass

    @abstractmethod
    def get_stock_news(self, stock_code, start_date, end_date):
        pass

    @abstractmethod
    def batch_fetch_stock_news(self, df_stock_codes: pd.DataFrame, start_date: str, end_date: str) -> list[pd.DataFrame]:
        pass

    @abstractmethod
    def get_latest_trade_date(self, stock_code: str):
        pass

    @abstractmethod
    def get_all_historical_data_from_db(self, stock_code: str, start_date: str, end_date: str):

        pass

    def __del__(self):
        pass
