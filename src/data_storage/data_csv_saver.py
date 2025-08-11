from src.data_storage.data_saver import DataSaver
import pandas as pd
import os
from src.utils.helpers import get_ts_code


class CsvSaver(DataSaver):
    def __init__(self, file_path, file_name: str):
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
            df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
        self.logger.info(f'保存{table_name}数据到CSV，数据形状: {df.shape}，文件路径: {csv_path}')

    def save_batch(self, df_list: list, table_names: list):
        """批量保存多个DataFrame到对应的表"""
        if len(df_list) != len(table_names):
            raise ValueError("DataFrame列表和表名列表长度必须一致")

        for df, table_name in zip(df_list, table_names):
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

    def read_latest_trade_date(self, table_name: str, ts_code: str) -> str:
        df = self.read(table_name, ts_code=ts_code)
        if df.empty or 'trade_date' not in df.columns:
            return ''

        # 找到最新的交易日期
        latest_date = df['trade_date'].max()
        self.logger.info(f'获取{table_name}表{ts_code}的最新交易日期: {latest_date}')
        return str(latest_date) if latest_date is not None else ''

    def close(self):
        # CSV不需要关闭连接，仅记录日志
        self.logger.info("CSV保存器已关闭")