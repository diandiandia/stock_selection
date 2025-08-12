import tushare as ts
import pandas as pd
from src.data_acquisition.data_fecher import DataFetcher
from src.utils.helpers import get_new_trade_date, get_ts_code


class TushareDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__()
        

    def login(self, token='c477c6691a86fa6f410f520f8f2e59f195ba9cb93b76384047de3d8d'):
        # Tushare通过token认证，此处无需额外登录
        ts.set_token(token)
        self.pro = ts.pro_api()

    def get_all_stock_codes(self, save:bool=True) -> pd.DataFrame:
        """获取所有股票代码列表"""
        # 获取A股列表
        df = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        
        if df is not None:
            # 过滤ST股票
            df = df[~df['name'].str.contains(r'ST|\*ST', na=False)]
            # 过滤创业板
            df = df[~df['ts_code'].str.startswith('3', na=False)]
            if df is not None and save:
                self.data_saver.save(df, 'stocks_info')
            return df
        else:
            self.logger.error('获取股票代码失败')
            return pd.DataFrame()

    def get_history_stock_data(self, stock_code: str, start_date: str, end_date: str, save:bool=False) -> pd.DataFrame:

        """获取股票历史行情数据"""
        ts_code = stock_code
        # 获取最新交易日之后的数据
        start_date = get_new_trade_date(self.data_saver, 'stock_daily', ts_code, start_date)
        
        # 调用Tushare接口获取日线数据
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        '''
        ts_code	str	股票代码
        trade_date	str	交易日期
        open	float	开盘价
        high	float	最高价
        low	float	最低价
        close	float	收盘价
        pre_close	float	昨收价【除权价，前复权】
        change	float	涨跌额
        pct_chg	float	涨跌幅 【基于除权后的昨收计算的涨跌幅：（今收-除权昨收）/除权昨收 】
        vol	float	成交量 （手）
        amount	float	成交额 （千元）
        
        '''
        
        if df is not None and not df.empty:
            # 获取目标列
            df = df[['trade_date', 'ts_code', 'open', 'close', 'high', 'low', 'vol', 'amount', 'pct_chg']]
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
            
            # 计算振幅
            df['amplitude'] = (df['high'] - df['low']) / df['open'] * 100

            # 计算涨跌额
            df['change_amount'] = (df['close'] - df['open'])

            # 计算换手率
            df_turnover = self._get_turnover(df, start_date, end_date)
            df = pd.merge(df, df_turnover, on=['trade_date', 'ts_code'], how='left')
            df = df.rename(columns={'turnover_rate':'turnover'})

            # 转换日期格式为字符串格式
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y%m%d')


            # 修改数据标识方法
            df['volume'] = df['volume'].astype(float)
            df['amount'] = df['amount'].astype(float)
            df['change'] = df['change'].astype(float)
            df['change_amount'] = df['change_amount'].astype(float)
            df['turnover'] = df['turnover'].astype(float)
            df['amplitude'] = df['amplitude'].astype(float)

            
            if save:
                self.data_saver.save(df, 'stock_daily')
            return df
        else:
            return pd.DataFrame()
        

    def _get_turnover(self, df:pd.DataFrame, start_date, end_date) -> pd.DataFrame:
        """计算换手率"""
        df = self.pro.daily_basic(ts_code=df['ts_code'].iloc[0], start_date=start_date, end_date=end_date)
        df = df[['trade_date', 'ts_code', 'turnover_rate']]
        return df



    def get_lhb_data(self, start_date, end_date, save:bool=False):

        """获取龙虎榜数据"""
        # 获取最新交易日之后的数据
        start_date = get_new_trade_date(self.data_saver, 'lhb_data', '', start_date)
        
        # 调用Tushare龙虎榜接口
        start_date = pd.to_datetime(start_date, format='%Y%m%d')
        end_date = pd.to_datetime(end_date, format='%Y%m%d')


        df_list = []

        for trade_date in pd.date_range(start_date, end_date):
            trade_date = trade_date.strftime('%Y%m%d')

            df = self.pro.top_list(trade_date=trade_date)

            '''
            trade_date	str	Y	交易日期
            ts_code	str	Y	TS代码
            name	str	Y	名称
            close	float	Y	收盘价
            pct_change	float	Y	涨跌幅
            turnover_rate	float	Y	换手率
            amount	float	Y	总成交额
            l_sell	float	Y	龙虎榜卖出额
            l_buy	float	Y	龙虎榜买入额
            l_amount	float	Y	龙虎榜成交额
            net_amount	float	Y	龙虎榜净买入额
            net_rate	float	Y	龙虎榜净买额占比
            amount_rate	float	Y	龙虎榜成交额占比
            float_values	float	Y	当日流通市值
            reason	str	Y	上榜理由
            '''

            if df is not None and not df.empty: 
                columns = {
                    'trade_date': 'trade_date',
                    'ts_code': 'ts_code',
                    'name': 'name',
                    'close': 'close',
                    'pct_change': 'change',
                    'turnover_rate': 'turnover',
                    'amount': 'total_amount',
                    'l_sell': 'sell_amount',
                    'l_buy': 'buy_amount',
                    'l_amount': 'amount',
                    'net_amount': 'net_buy_amount',
                    'net_rate': 'net_buy_amount_ratio',
                    'amount_rate': 'amount_ratio',
                    'float_values': 'market_value',
                    'reason': 'reason',
                }
            
                df = df.rename(columns=columns)

                # 转换日期格式
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y%m%d')

                df = df[['ts_code', 'name', 'trade_date', 'close', 'change', 'net_buy_amount', 'buy_amount', 
                         'sell_amount', 'amount', 'net_buy_amount_ratio', 'turnover', 
                         'reason']]
            
                if save:
                    self.data_saver.save(df, 'lhb_data')
                df_list.append(df)

            else:
                self.logger.warning(f"日期 {trade_date} 数据为空，跳过")
                continue
        df_all = pd.concat(df_list, axis=0)
        self.logger.info(f"获取龙虎榜数据完成，日期范围：{start_date} 至 {end_date}")
        for index, row in df_all.iterrows():
            self.logger.info(f"获取龙虎榜数据完成， 股票代码：{row['ts_code']}，股票名称：{row['name']}，上榜时间：{row['trade_date']}，"
                             f"收盘价：{row['close']}，涨跌幅：{row['change']}，净买入额：{row['net_buy_amount']}，"
                             f"买入额：{row['buy_amount']}，卖出额：{row['sell_amount']}，龙虎榜成交额：{row['amount']}，"
                             f"净买入额占比：{row['net_buy_amount_ratio']}，换手率：{row['turnover']}，"
                             f"上榜理由：{row['reason']}")


        return df_all

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