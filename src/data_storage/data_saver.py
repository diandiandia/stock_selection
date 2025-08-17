from abc import abstractmethod
from src.utils.log_helper import LogHelper


class DataSaver:

    def __init__(self, file_path:str, file_name:str):
        self.file_path = file_path
        self.file_name = file_name
        self.logger = LogHelper().get_logger(__name__)
        self.init_saver()

    @abstractmethod
    def init_saver(self):
        pass

    @abstractmethod
    def save(self, df):
        pass

    @abstractmethod
    def save_batch(self, df_list):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def read_all_data(self, table_name:str, ts_code:str = None, start_date:str = None, end_date:str = None):
        pass



    def __del__(self):
        self.close()