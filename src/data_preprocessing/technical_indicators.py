import pandas as pd
import numpy as np
import talib
import logging

class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        """初始化技术指标计算器"""
        self.df = df.copy()
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """计算所有技术指标"""
        if not self.validate_data(self.df):
            raise ValueError("Input data validation failed")
        
        logging.info("Starting technical indicator calculations...")
        
        # 按股票代码分组计算
        for code in self.df['ts_code'].unique():
            mask = self.df['ts_code'] == code
            stock_data = self.df.loc[mask].copy()
            
            # 确保数据按时间排序
            stock_data = stock_data.sort_values('trade_date')
            close_prices = stock_data['close'].values
            
            try:
                # 移动平均线
                self.df.loc[mask, 'MA5'] = talib.SMA(close_prices, timeperiod=5)
                self.df.loc[mask, 'MA10'] = talib.SMA(close_prices, timeperiod=10)
                self.df.loc[mask, 'MA20'] = talib.SMA(close_prices, timeperiod=20)
                
                # RSI指标
                self.df.loc[mask, 'RSI6'] = talib.RSI(close_prices, timeperiod=6)
                self.df.loc[mask, 'RSI12'] = talib.RSI(close_prices, timeperiod=12)
                
                # MACD指标
                macd, signal, hist = talib.MACD(close_prices)
                self.df.loc[mask, 'MACD'] = macd
                self.df.loc[mask, 'MACD_signal'] = signal
                self.df.loc[mask, 'MACD_hist'] = hist
                
                # 布林带
                upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)
                self.df.loc[mask, 'BB_upper'] = upper
                self.df.loc[mask, 'BB_middle'] = middle
                self.df.loc[mask, 'BB_lower'] = lower
                
                # 成交量指标
                volume = stock_data['volume'].values
                self.df.loc[mask, 'VOL_MA5'] = talib.SMA(volume, timeperiod=5)
                self.df.loc[mask, 'VOL_MA10'] = talib.SMA(volume, timeperiod=10)
                
                # 动量指标
                self.df.loc[mask, 'MOM'] = talib.MOM(close_prices, timeperiod=10)
                
                # 计算未来收益率（作为标签）
                self.df.loc[mask, 'future_return'] = stock_data['close'].shift(-1) / stock_data['close'] - 1
                
            except Exception as e:
                logging.error(f"Error calculating indicators for stock {code}: {str(e)}")
                continue
        
        # 移除无效数据
        self.df = self.df.replace([np.inf, -np.inf], np.nan)  # 替换无穷值为NaN
        self.df = self.df.dropna()  # 删除所有包含NaN的行
        
        logging.info(f"Completed technical indicator calculations. Shape: {self.df.shape}")
        logging.info(f"Columns available: {self.df.columns.tolist()}")
        
        return self.df
        

    def prepare_features(self) -> tuple:
        """准备模型训练所需的特征"""
        feature_columns = ['MA5', 'MA10', 'MA20', 'RSI6', 'RSI12', 
                         'MACD', 'MACD_signal', 'MACD_hist',
                         'BB_upper', 'BB_middle', 'BB_lower',
                         'VOL_MA5', 'VOL_MA10', 'MOM']
        
        X = self.df[feature_columns].values
        y = self.df['future_return'].values
        
        return X, y

    def prepare_single_stock_features(self, row: pd.Series) -> np.ndarray:
        """准备单个股票的特征"""
        feature_columns = ['MA5', 'MA10', 'MA20', 'RSI6', 'RSI12', 
                         'MACD', 'MACD_signal', 'MACD_hist',
                         'BB_upper', 'BB_middle', 'BB_lower',
                         'VOL_MA5', 'VOL_MA10', 'MOM']
        
        try:
            features = row[feature_columns].values
            if np.any(pd.isna(features)):
                return None
            return features
        except Exception as e:
            print(f"Error preparing features for row: {e}")
            return None
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证输入数据的完整性"""
        required_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return False
            
        return True