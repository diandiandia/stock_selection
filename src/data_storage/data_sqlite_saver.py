from src.data_storage.data_saver import DataSaver
import sqlite3
import pandas as pd
import os
from src.utils.helpers import get_ts_code


class SqliteSaver(DataSaver):
    def __init__(self, file_path, file_name:str):
        super().__init__(file_path, file_name)
        

    def init_saver(self):
        self.save_path = self.file_path + os.sep + self.file_name
        self.logger.info(f'init saver, file path: {self.save_path}')
        self.conn = sqlite3.connect(self.save_path)
        self.cursor = self.conn.cursor()
        self.create_tables()


    def create_tables(self):
        '''
        columns = {
                '代码': 'ts_code',
                '名称': 'name',
            }
        '''
        sql_command = '''
            CREATE TABLE IF NOT EXISTS stocks_info (
                ts_code TEXT PRIMARY KEY,
                name TEXT,
                industry TEXT
            )
        '''
        self.cursor.execute(sql_command)
        self.conn.commit()
        '''
            columns = {
                '日期': 'trade_date',
                '股票代码': 'ts_code',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change',
                '涨跌额': 'change_amount',
                '换手率': 'turnover',
            }
        
        '''
        sql_command = '''
            CREATE TABLE IF NOT EXISTS stock_daily (
                trade_date TEXT,
                ts_code TEXT,
                open REAL,
                close REAL,
                high REAL,
                low REAL,
                volume INTEGER,
                amount REAL,
                amplitude REAL,
                change REAL,
                change_amount REAL,
                turnover REAL,
                PRIMARY KEY (ts_code, trade_date)
            )
        '''
        self.cursor.execute(sql_command)
        self.conn.commit()

        '''
        df = df[['ts_code', 'name', 'trade_date', 'close', 'change', 'net_buy_amount', 'buy_amount', 
                'sell_amount', 'amount', 'net_buy_amount_ratio', 'turnover', 
                'reason']]
        '''

        sql_command = '''
            CREATE TABLE IF NOT EXISTS lhb_data (
                ts_code TEXT,
                name TEXT,
                trade_date TEXT,
                close REAL,
                change REAL,
                net_buy_amount REAL,
                buy_amount REAL,
                sell_amount REAL,
                amount REAL,
                net_buy_amount_ratio REAL,
                turnover REAL,
                reason TEXT,
                PRIMARY KEY (ts_code, name, trade_date)

            )
        '''
        self.cursor.execute(sql_command)
        self.conn.commit()

        '''
        column = {
            '关键词': 'ts_code',
            '新闻标题': 'title',
            '新闻内容': 'content',
            '发布时间': 'publish_time',
            '文章来源': 'source',
            '新闻链接': 'link',
        }
        '''
        sql_command = '''
            CREATE TABLE IF NOT EXISTS stock_news (
                ts_code TEXT,
                title TEXT,
                content TEXT,
                publish_time TEXT,
                source TEXT,
                link TEXT,
                PRIMARY KEY (ts_code, publish_time)
            )
        '''
        self.cursor.execute(sql_command)
        self.conn.commit()
        


    def save(self, df:pd.DataFrame, table_name:str):
        df.to_sql(table_name, self.conn, if_exists='append', index=False)
        self.conn.commit()
        self.logger.info(f'save {table_name} data to sqlite, data shape: {df.shape}')


    def save_batch(self, df_list:list):
        for df in df_list:
            self.save(df)
        self.conn.commit()
        self.logger.info(f'save batch data to sqlite, data shape: {len(df_list)}')
    
    def read(self, table_name:str, ts_code:str, start_date:str, end_date:str):
        sql_command = f'select * from {table_name}'
        if ts_code is not None:
            sql_command += f' where ts_code = "{ts_code}"'
        if start_date is not None and end_date is not None:
            sql_command += f' and trade_date >= {start_date} and trade_date <= {end_date}'
        self.logger.info(f'read {table_name} data from sqlite, sql command: {sql_command}')

        df = pd.read_sql(sql_command, self.conn)
        return df
    
    def query(self, df:pd.DataFrame, table_name:str)->pd.DataFrame:
        '''
        df['ts_code']数据是股票代码['600000','600001']，
        从table_name查询出来的数据ts_code数据格式为['600000.SH','600001.SZ, '600002.BJ']
        将df['ts_code']替换为['600000.SH','600001.SH']格式
        '''
        def get_ts_code(ts_code:str):
            sql_commend = f'select * from {table_name} where ts_code like "%{ts_code}%" limit 1'
            df = pd.read_sql(sql_commend, self.conn)
            if df is None or df.empty:
                return ts_code
            else:
                return df['ts_code'].values[0]
        df['ts_code'] = df['ts_code'].apply(get_ts_code)
        return df

    
    def read_latest_trade_date(self, table_name:str, ts_code:str)->str:
        sql_command = f'select max(trade_date) as latest_trade_date from {table_name}'
        if len(ts_code) > 0:
            sql_command += f' where ts_code = "{ts_code}"'
        self.logger.info(f'read {table_name} data from sqlite, sql command: {sql_command}')
        df = pd.read_sql(sql_command, self.conn)
        if df is None or df.empty:
            return ''
        else:
            latest_trade_date = df['latest_trade_date'].values[0]
            if latest_trade_date is None:
                return ''
            else:
                return latest_trade_date

    def close(self):
        self.conn.close()