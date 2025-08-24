from src.data_storage.data_saver import DataSaver
import pandas as pd
import os
from src.utils.helpers import get_ts_code


class CsvSaver(DataSaver):
    def __init__(self, file_path='./data', file_name='stock_data.csv'):
        super().__init__(file_path, file_name)
        self.save_path = os.path.join(file_path, file_name)
        # 创建保存路径（如果不存在）
        os.makedirs(self.save_path, exist_ok=True)

    def init_saver(self):
        self.logger.info(f'初始化CSV保存器，文件路径: {self.save_path}')
        # CSV不需要预创建表结构，在首次保存时自动生成

    def _get_csv_file_path(self, table_name: str) -> str:
        """获取指定表对应的CSV文件路径"""
        return os.path.join(self.save_path, f'{table_name}.csv')

    def save(self, df: pd.DataFrame, table_name: str):
        csv_path = self._get_csv_file_path(table_name)
        # 判断文件是否存在，不存在则写入表头
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, encoding='utf-8')
        else:
            # 追加模式，不写入表头
            df.to_csv(csv_path, mode='a', header=False,
                      index=False, encoding='utf-8')
        self.logger.info(
            f'保存{table_name}数据到CSV，数据形状: {df.shape}，文件路径: {csv_path}')

    def save_batch(self, df_list: list):
        """批量保存多个DataFrame到对应的表"""
        for df in df_list:
            # 从DataFrame中提取表名（假设DataFrame有'table_name'属性）
            table_name = getattr(df, 'table_name', None)
            if table_name is None:
                raise ValueError("DataFrame必须设置'table_name'属性")
            self.save(df, table_name)
        self.logger.info(f'批量保存完成，共处理{len(df_list)}个DataFrame')

    def read(self, table_name: str, ts_code: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        csv_path = self._get_csv_file_path(table_name)
        if not os.path.exists(csv_path):
            self.logger.warning(f'CSV文件不存在，返回空DataFrame: {csv_path}')
            return pd.DataFrame()

        # 读取CSV文件
        df = pd.read_csv(csv_path, encoding='utf-8')
        self.logger.info(f'从CSV读取{table_name}数据，原始数据形状: {df.shape}')

        # 应用过滤条件
        if ts_code is not None:
            df = df[df['ts_code'] == ts_code]
        if start_date is not None and 'trade_date' in df.columns:
            df = df[df['trade_date'] >= start_date]
        if end_date is not None and 'trade_date' in df.columns:
            df = df[df['trade_date'] <= end_date]

        self.logger.info(f'应用过滤条件后的数据形状: {df.shape}')
        return df

    def read_latest_trade_date(self, table_name: str, ts_code: str = None) -> str:
        df = self.read(table_name, ts_code=ts_code)
        if df.empty or 'trade_date' not in df.columns:
            return ''

        # 找到最新的交易日期
        latest_date = df['trade_date'].max()
        self.logger.info(f'获取{table_name}表{ts_code}的最新交易日期: {latest_date}')
        return str(latest_date) if latest_date is not None else ''

    def query(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        '''
        df['ts_code']数据是股票代码['600000','600001']，
        从table_name查询出来的数据ts_code数据格式为['600000.SH','600001.SZ, '600002.BJ']
        将df['ts_code']替换为['600000.SH','600001.SH']格式
        '''
        csv_path = self._get_csv_file_path(table_name)
        if not os.path.exists(csv_path):
            self.logger.warning(f'CSV文件不存在，无法进行查询: {csv_path}')
            return df

        # 读取CSV文件中的所有ts_code
        existing_df = pd.read_csv(csv_path, encoding='utf-8')
        if 'ts_code' not in existing_df.columns:
            self.logger.warning(f'CSV文件中不存在ts_code列: {csv_path}')
            return df

        # 提取已存在的股票代码前缀映射
        code_prefix_map = {}
        for code in existing_df['ts_code'].unique():
            # 假设代码格式为"600000.SH"，提取前缀"600000"和后缀".SH"
            if '.' in code:
                prefix = code.split('.')[0]
                suffix = code.split('.')[1]
                code_prefix_map[prefix] = suffix

        # 应用代码转换
        def get_full_ts_code(ts_code: str):
            if ts_code in code_prefix_map:
                return f'{ts_code}.{code_prefix_map[ts_code]}'
            # 如果找不到映射，返回原始代码
            return ts_code

        df['ts_code'] = df['ts_code'].apply(get_full_ts_code)
        return df

    def read_all_data(self, table_name: str, ts_code: str = None) -> pd.DataFrame:
        csv_path = self._get_csv_file_path(table_name)
        if not os.path.exists(csv_path):
            self.logger.warning(f'CSV文件不存在，返回空DataFrame: {csv_path}')
            return pd.DataFrame()
        df = pd.read_csv(csv_path, encoding='utf-8')
        self.logger.info(f'从CSV读取{table_name}数据，原始数据形状: {df.shape}')
        if ts_code is not None:
            df = df[df['ts_code'] == ts_code]
        return df

    def close(self):
        # CSV不需要关闭连接，仅记录日志
        self.logger.info("CSV保存器已关闭")
