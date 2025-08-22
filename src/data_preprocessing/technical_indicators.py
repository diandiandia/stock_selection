from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import talib
from src.utils.log_helper import LogHelper
from concurrent.futures import ThreadPoolExecutor
import gc

class TechnicalIndicators:
    """股票技术指标计算器，用于A股T+1交易推荐"""
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.logger = LogHelper.get_logger(__name__)
        self._df: Optional[pd.DataFrame] = None
        if df is not None:
            self._df = df.copy().sort_values('trade_date').reset_index(drop=True)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in self._df.columns for col in numeric_cols):
                self._df[numeric_cols] = self._df[numeric_cols].astype(np.float32)
                self._df = self._df.replace([np.inf, -np.inf], np.nan)
            else:
                raise ValueError(f"Missing required numeric columns: {numeric_cols}")

    def _empty_indicators(self, keys: List[str], length: int = 0) -> Dict[str, np.ndarray]:
        return {k: np.full(length, np.nan, dtype=np.float32) for k in keys}

    def _to_float64(self, arr: np.ndarray) -> np.ndarray:
        """Convert array to float64 for TA-Lib compatibility"""
        if arr.dtype != np.float64:
            self.logger.debug(f"Converting array from {arr.dtype} to float64 for TA-Lib")
            return arr.astype(np.float64)
        return arr

    def calculate_moving_averages(self, close: np.ndarray, periods: List[int] = [5, 10, 20], 
                                 use_ema: bool = False) -> Dict[str, np.ndarray]:
        if len(close) == 0:
            return self._empty_indicators([f'{"EMA" if use_ema else "MA"}{p}' for p in periods])
        
        max_period = max(periods)
        if len(close) < max_period:
            self.logger.warning(f"Data too short ({len(close)} < {max_period}) for moving averages")
            return self._empty_indicators([f'{"EMA" if use_ema else "MA"}{p}' for p in periods], len(close))
        
        close = self._to_float64(close)
        result = {}
        for period in periods:
            if use_ema:
                result[f'EMA{period}'] = talib.EMA(close, timeperiod=period).astype(np.float32)
            else:
                result[f'MA{period}'] = talib.SMA(close, timeperiod=period).astype(np.float32)
        return result

    def calculate_rsi(self, close: np.ndarray, period: int = 14) -> Dict[str, np.ndarray]:
        if len(close) == 0:
            return self._empty_indicators(['RSI'])
        
        if len(close) < period:
            self.logger.warning(f"Data too short ({len(close)} < {period}) for RSI")
            return self._empty_indicators(['RSI'], len(close))
        
        close = self._to_float64(close)
        return {'RSI': talib.RSI(close, timeperiod=period).astype(np.float32)}

    def calculate_macd(self, close: np.ndarray, fastperiod: int = 12, 
                      slowperiod: int = 26, signalperiod: int = 9) -> Dict[str, np.ndarray]:
        if len(close) == 0:
            return self._empty_indicators(['MACD', 'MACD_SIGNAL', 'MACD_HIST'])
        
        max_period = max(fastperiod, slowperiod, signalperiod)
        if len(close) < max_period:
            self.logger.warning(f"Data too short ({len(close)} < {max_period}) for MACD")
            return self._empty_indicators(['MACD', 'MACD_SIGNAL', 'MACD_HIST'], len(close))
        
        close = self._to_float64(close)
        macd, signal, hist = talib.MACD(close, fastperiod=fastperiod, 
                                      slowperiod=slowperiod, signalperiod=signalperiod)
        return {
            'MACD': macd.astype(np.float32),
            'MACD_SIGNAL': signal.astype(np.float32),
            'MACD_HIST': hist.astype(np.float32)
        }

    def calculate_bollinger_bands(self, close: np.ndarray, period: int = 20, 
                                nbdevup: int = 2, nbdevdn: int = 2) -> Dict[str, np.ndarray]:
        if len(close) == 0:
            return self._empty_indicators(['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'])
        
        if len(close) < period:
            self.logger.warning(f"Data too short ({len(close)} < {period}) for Bollinger Bands")
            return self._empty_indicators(['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'], len(close))
        
        close = self._to_float64(close)
        upper, middle, lower = talib.BBANDS(close, timeperiod=period, 
                                          nbdevup=nbdevup, nbdevdn=nbdevdn)
        return {
            'BB_UPPER': upper.astype(np.float32),
            'BB_MIDDLE': middle.astype(np.float32),
            'BB_LOWER': lower.astype(np.float32)
        }

    def calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int = 14) -> Dict[str, np.ndarray]:
        if len(high) == 0 or len(low) == 0 or len(close) == 0:
            return self._empty_indicators(['ADX', 'PLUS_DI', 'MINUS_DI'])
        
        if len(high) < period:
            self.logger.warning(f"Data too short ({len(high)} < {period}) for ADX")
            return self._empty_indicators(['ADX', 'PLUS_DI', 'MINUS_DI'], len(high))
        
        high = self._to_float64(high)
        low = self._to_float64(low)
        close = self._to_float64(close)
        return {
            'ADX': talib.ADX(high, low, close, timeperiod=period).astype(np.float32),
            'PLUS_DI': talib.PLUS_DI(high, low, close, timeperiod=period).astype(np.float32),
            'MINUS_DI': talib.MINUS_DI(high, low, close, timeperiod=period).astype(np.float32)
        }

    def calculate_momentum(self, close: np.ndarray, period: int = 10) -> Dict[str, np.ndarray]:
        if len(close) == 0:
            return self._empty_indicators(['MOM'])
        
        if len(close) < period:
            self.logger.warning(f"Data too short ({len(close)} < {period}) for Momentum")
            return self._empty_indicators(['MOM'], len(close))
        
        close = self._to_float64(close)
        return {'MOM': talib.MOM(close, timeperiod=period).astype(np.float32)}

    def calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int = 14) -> Dict[str, np.ndarray]:
        if len(high) == 0 or len(low) == 0 or len(close) == 0:
            return self._empty_indicators(['CCI'])
        
        if len(high) < period:
            self.logger.warning(f"Data too short ({len(high)} < {period}) for CCI")
            return self._empty_indicators(['CCI'], len(high))
        
        high = self._to_float64(high)
        low = self._to_float64(low)
        close = self._to_float64(close)
        return {'CCI': talib.CCI(high, low, close, timeperiod=period).astype(np.float32)}

    def calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        if len(close) == 0 or len(volume) == 0:
            return self._empty_indicators(['OBV'])
        
        close = self._to_float64(close)
        volume = self._to_float64(volume)
        return {'OBV': talib.OBV(close, volume).astype(np.float32)}

    def calculate_cmf(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     volume: np.ndarray, period: int = 20) -> Dict[str, np.ndarray]:
        if len(high) == 0 or len(low) == 0 or len(close) == 0 or len(volume) == 0:
            return self._empty_indicators(['CMF'])
        
        if len(high) < period:
            self.logger.warning(f"Data too short ({len(high)} < {period}) for CMF")
            return self._empty_indicators(['CMF'], len(high))
        
        high = self._to_float64(high)
        low = self._to_float64(low)
        close = self._to_float64(close)
        volume = self._to_float64(volume)
        
        # Convert to pandas Series for rolling operations
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        volume_series = pd.Series(volume)
        
        money_flow_multiplier = ((close_series - low_series) - (high_series - close_series)) / (high_series - low_series + 1e-10)
        money_flow_volume = money_flow_multiplier * volume_series
        cmf = money_flow_volume.rolling(window=period).sum() / volume_series.rolling(window=period).sum()
        return {'CMF': cmf.to_numpy().astype(np.float32)}

    def calculate_candlestick_patterns(self, open: np.ndarray, high: np.ndarray, 
                                     low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        if len(open) == 0 or len(high) == 0 or len(low) == 0 or len(close) == 0:
            return self._empty_indicators(['CDL_DOJI'])
        
        open = self._to_float64(open)
        high = self._to_float64(high)
        low = self._to_float64(low)
        close = self._to_float64(close)
        
        return {
            'CDL_DOJI': talib.CDLDOJI(open, high, low, close).astype(np.float32)
        }

    def calculate_volume_indicators(self, volume: np.ndarray, period: int = 5) -> Dict[str, np.ndarray]:
        if len(volume) == 0:
            return self._empty_indicators(['VOL_MA5'])
        
        if len(volume) < period:
            self.logger.warning(f"Data too short ({len(volume)} < {period}) for volume indicators")
            return self._empty_indicators(['VOL_MA5'], len(volume))
        
        volume = self._to_float64(volume)
        return {'VOL_MA5': talib.SMA(volume, timeperiod=period).astype(np.float32)}

    def calculate_all_indicators(self, fillna_method: str = 'hybrid', min_data_points: int = 20) -> pd.DataFrame:
        """计算所有技术指标，过滤数据不足的股票"""
        if self._df is None:
            self.logger.error("DataFrame is None")
            return pd.DataFrame()
        
        required_columns = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in self._df.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return pd.DataFrame()
        
        indicators = {
            'MA': self.calculate_moving_averages,
            'RSI': self.calculate_rsi,
            'MACD': self.calculate_macd,
            'BBANDS': self.calculate_bollinger_bands,
            'ADX': self.calculate_adx,
            'MOM': self.calculate_momentum,
            'CCI': self.calculate_cci,
            'OBV': self.calculate_obv,
            'CMF': self.calculate_cmf,
            'CANDLESTICK': self.calculate_candlestick_patterns,
            'VOLUME': self.calculate_volume_indicators
        }
        
        def compute_group(group):
            # 跳过数据点不足的股票
            if len(group) < min_data_points:
                self.logger.warning(f"Skipping stock {group['ts_code'].iloc[0]}: insufficient data points ({len(group)} < {min_data_points})")
                return pd.DataFrame()
            
            result = group[required_columns].copy()
            
            # 计算基本指标
            result['amplitude'] = (group['high'] - group['low']) / group['close'].shift(1)
            result['change'] = group['close'].pct_change()
            result['change_amount'] = group['close'] - group['close'].shift(1)
            result['turnover'] = group['volume'] / group['close']
            
            # 计算技术指标
            for name, func in indicators.items():
                if name == 'MA':
                    ind = func(group['close'].values, periods=[5, 10, 20], use_ema=False)
                elif name == 'RSI':
                    ind = func(group['close'].values)
                elif name == 'MACD':
                    ind = func(group['close'].values)
                elif name == 'BBANDS':
                    ind = func(group['close'].values)
                elif name == 'ADX':
                    ind = func(group['high'].values, group['low'].values, group['close'].values)
                elif name == 'MOM':
                    ind = func(group['close'].values)
                elif name == 'CCI':
                    ind = func(group['high'].values, group['low'].values, group['close'].values)
                elif name == 'OBV':
                    ind = func(group['close'].values, group['volume'].values)
                elif name == 'CMF':
                    ind = func(group['high'].values, group['low'].values, group['close'].values, group['volume'].values)
                elif name == 'CANDLESTICK':
                    ind = func(group['open'].values, group['high'].values, group['low'].values, group['close'].values)
                elif name == 'VOLUME':
                    ind = func(group['volume'].values)
                else:
                    continue
                
                for key, value in ind.items():
                    result[key] = value
            
            # 处理NaN值
            indicator_cols = [col for col in result.columns if col not in required_columns]
            if fillna_method == 'hybrid':
                result[indicator_cols] = result[indicator_cols].ffill().bfill().fillna(result[indicator_cols].median()).astype(np.float32)
            elif fillna_method == 'ffill':
                result[indicator_cols] = result[indicator_cols].ffill().fillna(0).astype(np.float32)
            elif fillna_method == 'bfill':
                result[indicator_cols] = result[indicator_cols].bfill().fillna(0).astype(np.float32)
            elif fillna_method == 'mean':
                result[indicator_cols] = result[indicator_cols].fillna(result[indicator_cols].mean()).astype(np.float32)
            elif fillna_method is None:
                pass
            else:
                raise ValueError(f"Unsupported fillna_method: {fillna_method}")
            
            # Log missing features
            expected_features = set(indicators.keys())
            actual_features = set([col for col in result.columns if col not in required_columns])
            missing_features = expected_features - actual_features
            if missing_features:
                self.logger.warning(f"Missing features for stock {group['ts_code'].iloc[0]}: {missing_features}")
            
            return result
        
        if 'ts_code' in self._df.columns:
            groups = [group for _, group in self._df.groupby('ts_code')]
            with ThreadPoolExecutor(max_workers=2) as executor:  # Further reduced to conserve memory
                results = list(executor.map(compute_group, groups))
            # 过滤空DataFrame
            results = [r for r in results if not r.empty]
            if not results:
                self.logger.error("No valid stock data after filtering")
                return pd.DataFrame()
            result_df = pd.concat(results)
        else:
            result_df = compute_group(self._df)
        
        # Final NaN check
        indicator_cols = [col for col in result_df.columns if col not in required_columns]
        nan_counts = result_df[indicator_cols].isna().sum()
        if nan_counts.sum() > 0:
            self.logger.warning(f"Remaining NaNs in indicators: {nan_counts[nan_counts > 0].to_dict()}")
        
        gc.collect()
        return result_df