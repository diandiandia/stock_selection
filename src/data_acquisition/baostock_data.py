import baostock as bs
import pandas as pd
import datetime
from src.data_acquisition.data_fecher import DataFetcher
from src.utils.helpers import get_new_trade_date, get_ts_code


class BaostockDataFetcher(DataFetcher):
    def __init__(self, url):
        super().__init__(url)
        # 初始化Baostock连接
        lg = bs.login()
        if lg.error_code != '0':
            self.logger.error(f'Baostock登录失败: {lg.error_msg}')
        else:
            self.logger.info('Baostock登录成功')

    def login(self, url, token):
        # Baostock无需额外token登录，已在__init__中处理
        pass

    def get_all_stock_codes(self, save:bool=True) -> pd.DataFrame:
        """获取所有A股股票代码列表"""
        # 查询所有A股股票
        rs = bs.query_stock_basic(code_name="A股")
        if rs.error_code != '0':
            self.logger.error(f'获取股票列表失败: {rs.error_msg}')
            return pd.DataFrame()

        df = rs.get_data()
        if df.empty:
            return pd.DataFrame()

        # 转换为标准ts_code格式 (证券代码.市场标识)
        df['ts_code'] = df.apply(lambda x: f'{x.code}.SH' if x.exchange == 'sh' else f'{x.code}.SZ', axis=1)
        
        # 重命名列并过滤
        columns = {
            'code_name': 'name',
            'industry': 'industry'
        }
        df = df.rename(columns=columns)
        df = df[['ts_code', 'name', 'industry']]

        # 过滤ST股票和创业板
        df = df[~df['name'].str.contains(r'ST|\*ST', na=False)]
        df = df[~df['ts_code'].str.startswith('3', na=False)]

        return df

    def get_history_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史行情数据"""
        ts_code = stock_code
        stock_code = get_ts_code(stock_code)
        start_date = get_new_trade_date(self.data_saver, 'stock_daily', ts_code, start_date)

        # 转换日期格式为YYYY-MM-DD (Baostock要求的格式)
        bs_start_date = datetime.datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
        bs_end_date = datetime.datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')

        # 查询日线数据
        rs = bs.query_history_k_data_plus(
            code=stock_code,
            fields='date,open,high,low,close,volume,amount,turnover, pctChg',
            start_date=bs_start_date,
            end_date=bs_end_date,
            frequency='d',
            adjustflag='2'  # 2=前复权
        )

        if rs.error_code != '0':
            self.logger.error(f'获取{stock_code}历史数据失败: {rs.error_msg}')
            return pd.DataFrame()

        df = rs.get_data()
        if df.empty:
            return pd.DataFrame()

        # 重命名列并转换数据类型
        columns = {
            'date': 'trade_date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount',
            'turnover': 'turnover',
            'pctChg': 'change'
        }
        df = df.rename(columns=columns)

        # 转换数据类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'change']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # 计算振幅和涨跌额
        df['amplitude'] = (df['high'] - df['low']) / df['open'] * 100
        df['change_amount'] = df['close'] - df['open']

        # 添加ts_code和格式化日期
        df['ts_code'] = ts_code
        df['trade_date'] = df['trade_date'].str.replace('-', '')

        # 选择需要的列
        df = df[['trade_date', 'ts_code', 'open', 'close', 'high', 'low', 'volume', 'amount', 
                 'amplitude', 'change', 'change_amount', 'turnover']]

        self.data_saver.save(df, 'stock_daily')
        return df

    def batch_fetch_historical_data(self, df_stock_codes: pd.DataFrame, start_date: str, end_date: str, save:bool=True) -> list[pd.DataFrame]:
        """批量获取股票历史数据"""
        df_all_list = []
        for stock_code in df_stock_codes['ts_code']:
            df = self.get_history_stock_data(stock_code, start_date, end_date, save)
            if not df.empty:
                df_all_list.append(df)
            else:
                self.logger.warning(f"股票代码 {stock_code} 数据为空，跳过")
        return df_all_list

    def get_lhb_data(self, start_date, end_date):
        """Baostock暂不支持龙虎榜数据接口"""
        self.logger.warning("Baostock数据源不支持龙虎榜数据获取")
        return pd.DataFrame()

    def get_stock_news(self, stock_code, start_date, end_date) -> pd.DataFrame:
        """Baostock暂不支持新闻数据接口"""
        self.logger.warning("Baostock数据源不支持新闻数据获取")
        return pd.DataFrame()

    def batch_fetch_stock_news(self, df_stock_codes: pd.DataFrame, start_date: str, end_date: str) -> list[pd.DataFrame]:
        """Baostock暂不支持新闻数据接口"""
        self.logger.warning("Baostock数据源不支持新闻数据获取")
        return []

    def get_latest_trade_date(self, table_name: str, ts_code: str):
        return self.data_saver.read_latest_trade_date(table_name, ts_code)

    def __del__(self):
        # 退出Baostock连接
        bs.logout()