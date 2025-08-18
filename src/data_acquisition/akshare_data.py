import random
import time
import akshare as ak
import pandas as pd

from src.data_acquisition.data_fecher import DataFetcher
from src.utils.helpers import get_new_trade_date, get_ts_code



class AkshareDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__()


    def login(self, token=''):
        pass


    def get_stock_codes_by_symbol(self, symbol:str='000906', save:bool=False) ->pd.DataFrame:
        """
        获取指定交易市场的股票代码列表
        """
        '''
        名称	类型	描述
        日期	object	-
        指数代码	object	-
        指数名称	object	-
        指数英文名称	object	-
        成分券代码	object	-
        成分券名称	object	-
        成分券英文名称	object	-
        交易所	object	-
        交易所英文名称	object	-
        '''
        df = ak.index_stock_cons_csindex(symbol=symbol)
        columns = {
            '成分券代码': 'ts_code',
            '成分券名称': 'name',
            '成分券英文名称': 'stock_exchange',
        }
        df = df.rename(columns=columns)
        df['ts_code'] = df['ts_code'].apply(ak.stock_a_code_to_symbol)
        # 将ts_code的值从sz000688转化为000688.SZ
        df['ts_code'] = df['ts_code'].str[2:] + '.' + df['ts_code'].str[0:2].str.upper()

        df = df[['ts_code', 'name']]
        df['industry'] = ''
        

        if save:
            self.data_saver.save(df, 'stocks_info')

        return df.sort_values('ts_code', ascending=False)


    def get_all_stock_codes(self, save:bool=True) ->pd.DataFrame:

        '''
        获取所有的股票代码
        '''
        df_sh = ak.stock_info_sh_name_code(symbol="主板A股")
        df_sz = ak.stock_info_sz_name_code(symbol="A股列表")
        df_bj = ak.stock_info_bj_name_code()

        sh_columns = {
            '证券代码': 'ts_code',
            '证券简称': 'name',
        }
        sz_columns = {
            'A股代码': 'ts_code',
            'A股简称': 'name',
            '所属行业': 'industry',
        }
        bj_columns = {
            '证券代码': 'ts_code',
            '证券简称': 'name',
            '所属行业': 'industry',
        }

        if df_sh is not None:
            df_sh = df_sh.rename(columns=sh_columns)
            df_sh = df_sh[['ts_code', 'name']]
            df_sh['industry'] = ''
            df_sh['ts_code'] = df_sh['ts_code'].apply(lambda x:x+'.SH')
        if df_sz is not None:
            df_sz = df_sz.rename(columns=sz_columns)
            df_sz = df_sz[['ts_code', 'name', 'industry']]
            df_sz['ts_code'] = df_sz['ts_code'].apply(lambda x:x+'.SZ')
        if df_bj is not None:
            df_bj = df_bj.rename(columns=bj_columns)
            df_bj = df_bj[['ts_code', 'name', 'industry']]
            df_bj['ts_code'] = df_bj['ts_code'].apply(lambda x:x+'.BJ')

        df = pd.concat([df_sh, df_sz, df_bj], axis=0)
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
        
    def get_history_stock_data(self, stock_code:str, start_date:str, end_date:str, save:bool=True) ->pd.DataFrame:

        '''
        名称	类型	描述
        日期	object	交易日
        股票代码	object	不带市场标识的股票代码
        开盘	float64	开盘价
        收盘	float64	收盘价
        最高	float64	最高价
        最低	float64	最低价
        成交量	int64	注意单位: 手
        成交额	float64	注意单位: 元
        振幅	float64	注意单位: %
        涨跌幅	float64	注意单位: %
        涨跌额	float64	注意单位: 元
        换手率	float64	注意单位: %
        
        '''
        ts_code = stock_code
        stock_code = get_ts_code(stock_code)
        start_date = get_new_trade_date(self.data_saver, 'stock_daily', ts_code,start_date)

        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df is not None and not df.empty:
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
            df = df.rename(columns=columns)
            # 修改ts_code的值
            df['ts_code'] = df['ts_code'].apply(lambda x:ts_code)
            # 转换日期格式为字符串格式
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y%m%d')

            # 修改数据标识形式
            df['open'] = df['open'].astype(float).round(2)
            df['close'] = df['close'].astype(float).round(2)
            df['high'] = df['high'].astype(float).round(2)
            df['low'] = df['low'].astype(float).round(2)
            df['volume'] = df['volume'].astype(float).round(2)
            df['amount'] = df['amount'].astype(float).round(2)
            df['change'] = df['change'].astype(float).round(2)
            df['change_amount'] = df['change_amount'].astype(float).round(2)
            df['turnover'] = df['turnover'].astype(float).round(2)
            df['amplitude'] = df['amplitude'].astype(float).round(2)



            if save:
                self.data_saver.save(df, 'stock_daily')
            return df.sort_values('trade_date', ascending=False)

        else:
            return pd.DataFrame()
       
    def batch_fetch_historical_data(self, df_stock_codes:pd.DataFrame, start_date:str, end_date:str) ->list[pd.DataFrame]:
        '''
        批量获取股票历史数据
        '''
        df_all_list = []
        for stock_code in df_stock_codes['ts_code']:
            time.sleep(random.random())  # 随机休眠0-2秒，防止被封IP
            df = self.get_history_stock_data(stock_code, start_date, end_date)
            if not df.empty:
                df_all_list.append(df)
            else:
                self.logger.warning(f"股票代码 {stock_code} 数据为空，跳过")
        return df_all_list


    def get_lhb_data(self, start_date, end_date, save:bool=False):

        '''
        名称	类型	描述
        序号	int64	-
        代码	object	-
        名称	object	-
        上榜日	object	-
        解读	object	-
        收盘价	float64	-
        涨跌幅	float64	注意单位: %
        龙虎榜净买额	float64	注意单位: 元
        龙虎榜买入额	float64	注意单位: 元
        龙虎榜卖出额	float64	注意单位: 元
        龙虎榜成交额	float64	注意单位: 元
        市场总成交额	int64	注意单位: 元
        净买额占总成交比	float64	注意单位: %
        成交额占总成交比	float64	注意单位: %
        换手率	float64	注意单位: %
        流通市值	float64	注意单位: 元
        上榜原因	object	-
        上榜后1日	float64	注意单位: %
        上榜后2日	float64	注意单位: %
        上榜后5日	float64	注意单位: %
        上榜后10日	float64	注意单位: %
        '''
        # 先从数据表里面查询最新的时间，然后从最新的时间开始获取数据
        start_date = get_new_trade_date(self.data_saver, 'lhb_data', '', start_date)
        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            self.logger.error('开始日期不能大于结束日期')
            return pd.DataFrame()


        df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
        column = {
            '序号': 'rank',
            '代码': 'ts_code',
            '名称': 'name',
            '上榜日': 'trade_date',
            '解读': 'interpretation',
            '收盘价': 'close',
            '涨跌幅': 'change',
            '龙虎榜净买额': 'net_buy_amount',
            '龙虎榜买入额': 'buy_amount',
            '龙虎榜卖出额': 'sell_amount',
            '龙虎榜成交额': 'amount',
            '市场总成交额': 'total_amount',
            '净买额占总成交比': 'net_buy_amount_ratio',
            '成交额占总成交比': 'amount_ratio',
            '换手率': 'turnover',
            '流通市值': 'market_value',
            '上榜原因': 'reason',
            '上榜后1日': 'change_1d',
            '上榜后2日': 'change_2d',
            '上榜后5日': 'change_5d',
            '上榜后10日': 'change_10d',
        }
        if df is not None and not df.empty:
            df = df.rename(columns=column)
            
            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y%m%d')
            df = df[['ts_code', 'name', 'trade_date', 'close', 'change', 'net_buy_amount', 'buy_amount', 
                         'sell_amount', 'amount', 'net_buy_amount_ratio', 'turnover', 
                         'reason']]
            
            # 根据ts_code判断术语哪个交易所
            df['ts_code'] = df['ts_code'].apply(ak.stock_a_code_to_symbol)
            # 将ts_code的值从sz000688转化为000688.SZ
            df['ts_code'] = df['ts_code'].str[2:] + '.' + df['ts_code'].str[0:2].str.upper()
            
            if save:
                self.data_saver.save(df, 'lhb_data')
            
            self.logger.info(f"获取龙虎榜数据完成，日期范围：{start_date} 至 {end_date}")
            for _, row in df.iterrows():
                self.logger.info(f"获取龙虎榜数据完成， 股票代码：{row['ts_code']}，股票名称：{row['name']}，上榜时间：{row['trade_date']}，"
                                f"收盘价：{row['close']}，涨跌幅：{row['change']}，净买入额：{row['net_buy_amount']}，"
                                f"买入额：{row['buy_amount']}，卖出额：{row['sell_amount']}，龙虎榜成交额：{row['amount']}， "
                                f"净买入额占比：{row['net_buy_amount_ratio']}，换手率：{row['turnover']}，"
                                f"上榜理由：{row['reason']}")
            return df
        else:
            return pd.DataFrame()


    def get_stock_news(self, stock_code, start_date, end_date)->pd.DataFrame:
        '''
        关键词	object	-
        新闻标题	object	-
        新闻内容	object	-
        发布时间	object	-
        文章来源	object	-
        新闻链接	object	-
        '''

        ts_code = stock_code
        stock_code = get_ts_code(stock_code)
        
        column = {
            '关键词': 'ts_code',
            '新闻标题': 'title',
            '新闻内容': 'content',
            '发布时间': 'publish_time',
            '文章来源': 'source',
            '新闻链接': 'link',
        }

        self.logger.info(f'获取股票 {stock_code} 从 {start_date} 到 {end_date} 新闻')
        df = ak.stock_news_em(symbol=stock_code)
        if df is not None and not df.empty:
            df = df.rename(columns=column)
            # 增加一列ts_code
            df['ts_code'] = df['ts_code'].apply(lambda x:ts_code)
            self.data_saver.save(df, 'stock_news')
            return df
        else:
            return pd.DataFrame()
        

    def batch_fetch_stock_news(self, df_stock_codes:pd.DataFrame, start_date:str, end_date:str)->list[pd.DataFrame]:
        '''
        批量获取股票新闻
        '''
        df_all_list = []
        for stock_code in df_stock_codes['ts_code']:
            df = self.get_stock_news(stock_code, start_date, end_date)
            if not df.empty:
                df_all_list.append(df)
            else:
                self.logger.warning(f"股票代码 {stock_code} 数据为空，跳过")
        return df_all_list
    
    def get_latest_trade_date(self, table_name:str, ts_code:str):
        self.data_saver.read_latest_trade_date(table_name, ts_code)


    def get_all_historical_data_from_db(self, table_name:str, ts_code:str= None, start_date:str= None, end_date:str= None):
        return self.data_saver.read_all_data(table_name, ts_code, start_date, end_date)
