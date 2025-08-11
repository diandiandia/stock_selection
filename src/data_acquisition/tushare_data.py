import tushare as ts
import pandas as pd
from src.data_acquisition.data_fecher import DataFetcher
from src.utils.helpers import get_new_trade_date, get_ts_code


class TushareDataFetcher(DataFetcher):
    def __init__(self, url, token):
        super().__init__(url)
        # 初始化Tushare接口
        ts.set_token(token)
        self.pro = ts.pro_api()

    def login(self, url, token):
        # Tushare通过token认证，此处无需额外登录
        pass

    def get_all_stock_codes(self) -> pd.DataFrame:
        """获取所有股票代码列表"""
        # 获取A股列表
        df = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        
        if df is not None:
            # 过滤ST股票
            df = df[~df['name'].str.contains(r'ST|\*ST', na=False)]
            # 过滤创业板
            df = df[~df['ts_code'].str.startswith('3', na=False)]
            return df
        else:
            self.logger.error('获取股票代码失败')
            return pd.DataFrame()

    def get_history_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票历史行情数据"""
        ts_code = stock_code
        # 获取最新交易日之后的数据
        start_date = get_new_trade_date(self.data_saver, 'stock_daily', ts_code, start_date)
        
        # 调用Tushare接口获取日线数据
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if df is not None and not df.empty:
            columns = {
                'trade_date': 'trade_date',
                'ts_code': 'ts_code',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'vol': 'volume',  # Tushare的成交量字段是vol
                'amount': 'amount',
                'pct_chg': 'change',  # 涨跌幅字段
            }
            
            # 重命名列并选择需要的列
            df = df.rename(columns=columns)
            df = df[['trade_date', 'ts_code', 'open', 'close', 'high', 'low', 'volume', 'amount', 'change']]
            
            # 计算振幅和换手率（Tushare需要单独获取）
            df = self._calculate_additional_fields(df, ts_code)
            
            self.data_saver.save(df, 'stock_daily')
            return df
        else:
            return pd.DataFrame()

    def _calculate_additional_fields(self, df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
        """计算振幅、换手率等额外字段"""
        # 获取股票基本信息
        stock_basic = self.pro.stock_basic(ts_code=ts_code, fields='total_share,float_share')
        if stock_basic.empty:
            return df

        # 计算换手率 (成交量/流通股本)
        df['turnover'] = df['volume'] / stock_basic.iloc[0]['float_share'] * 100
        
        # 计算振幅 ((最高价-最低价)/开盘价)*100
        df['amplitude'] = (df['high'] - df['low']) / df['open'] * 100
        
        # 计算涨跌额
        df['change_amount'] = df['close'] - df['open']
        
        return df

    def get_lhb_data(self, start_date, end_date):
        """获取龙虎榜数据"""
        # 获取最新交易日之后的数据
        start_date = get_new_trade_date(self.data_saver, 'lhb_data', '', start_date)
        
        # 调用Tushare龙虎榜接口
        df = self.pro.top_list(ts_code='', start_date=start_date, end_date=end_date)
        
        if df is not None and not df.empty:
            columns = {
                'trade_date': 'trade_date',
                'ts_code': 'ts_code',
                'name': 'name',
                'close': 'close',
                'pct_chg': 'change',
                'net_amount': 'net_buy_amount',
                'buy_amount': 'buy_amount',
                'sell_amount': 'sell_amount',
                'amount': 'amount',
                'turnover_rate': 'turnover',
                'reason': 'reason',
            }
            
            df = df.rename(columns=columns)
            # 补充其他需要的字段
            df['market_value'] = df['amount'] / df['turnover'] * 100
            df['total_amount'] = df['amount']
            df['net_buy_amount_ratio'] = df['net_buy_amount'] / df['amount'] * 100
            df['amount_ratio'] = 100  # Tushare没有直接提供，这里用100代替
            
            # 添加缺失的字段并填充默认值
            for col in ['rank', 'interpretation', 'change_1d', 'change_2d', 'change_5d', 'change_10d']:
                df[col] = None
            
            self.data_saver.save(df, 'lhb_data')
            return df
        else:
            return pd.DataFrame()

    def batch_fetch_historical_data(self, df_stock_codes: pd.DataFrame, start_date: str, end_date: str) -> list[pd.DataFrame]:
        """批量获取股票历史数据"""
        df_all_list = []
        for stock_code in df_stock_codes['ts_code']:
            df = self.get_history_stock_data(stock_code, start_date, end_date)
            if not df.empty:
                df_all_list.append(df)
            else:
                self.logger.warning(f"股票代码 {stock_code} 数据为空，跳过")
        return df_all_list

    def get_latest_trade_date(self, table_name: str, ts_code: str):
        return self.data_saver.read_latest_trade_date(table_name, ts_code)