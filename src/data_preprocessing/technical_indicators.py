import talib
import pandas as pd

class TechnicalIndicators:
    def __init__(self, df):
        self.df = df

    def calculate_technical_indicators(self)->pd.DataFrame:
        '''
        计算股票的技术指标
            df columns = {
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

        # 定义计算技术指标的函数
        def calculate_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
            """计算趋势指标。"""
            df["ma5"] = talib.SMA(df["close"], timeperiod=5)
            df["ma10"] = talib.SMA(df["close"], timeperiod=10)
            df["ema8"] = talib.EMA(df["close"], timeperiod=8)
            df["ema21"] = talib.EMA(df["close"], timeperiod=21)
            df["ma_cross_diff"] = df["ma5"] - df["ma10"]
            df["ema_cross_diff"] = df["ema8"] - df["ema21"]
            return df

        def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
            """计算动能指标。"""
            df["rsi"] = talib.RSI(df["close"], timeperiod=14)
            df["slowk"], df["slowd"] = talib.STOCH(df["high"], df["low"], df["close"])
            df["kdj_diff"] = df["slowk"] - df["slowd"]
            df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(df["close"])
            return df

        def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
            """计算成交量指标。"""
            df["volume_ratio"] = df["volume"] / df["volume"].shift(1)
            df["vol_prev1"] = df["volume"].shift(1)
            df["net_mf_amount"] = (df["close"] - df["open"]) * df["volume"]
            return df

        def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
            """计算波动性指标。"""
            df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
            df["intraday_range"] = (df["high"] - df["low"]) / df["close"].shift(1)
            return df

        def calculate_lag_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
            """计算滞后价格指标。"""
            df["close_prev1"] = df["close"].shift(1)
            df["open_prev1"] = df["open"].shift(1)
            df["gap"] = (df["open"] - df["close_prev1"]) / df["close_prev1"]
            return df

        def calculate_market_environment_indicators(df: pd.DataFrame) -> pd.DataFrame:
            """计算市场环境指标。"""
            df["turnover_rate"] = df["volume"] / df["circ_mv"] * 100
            return df
    

        # 计算技术指标
        self.df = calculate_trend_indicators(self.df)
        self.df = calculate_momentum_indicators(self.df)
        self.df = calculate_volume_indicators(self.df)
        self.df = calculate_volatility_indicators(self.df)
        self.df = calculate_lag_price_indicators(self.df)
        self.df = calculate_market_environment_indicators(self.df)


    def calculate_sentiment_indicators(self):
        """
        计算情感指标
        """
        pass








