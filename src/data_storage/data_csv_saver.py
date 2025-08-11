from data_saver import DataSaver



class CsvSaver(DataSaver):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.file_path = file_path

    def init_saver(self):
        pass

    def save(self, df):
        df.to_csv(self.file_path)

    def save_batch(self, df_list):
        pass

    def close(self):
        pass    
