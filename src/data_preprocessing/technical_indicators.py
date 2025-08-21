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
            self.logger.warning(f"Data too short ({len(close)} < {max_period}), returning NaNs.")
            return self._empty_indicators([f'{"EMA" if use_ema else "MA"}{p}' for p in periods], len(close))
        
        close = self._to_float64(close)
        if use_ema:
            return {f'EMA{p}': talib.EMA(close, timeperiod=p).astype(np.float32) for p in periods}
        return {f'MA{p}': talib.SMA(close, timeperiod=p).astype(np.float32) for p in periods}

    def calculate_momentum_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                     rsi_periods: List[int] = [6, 12], mom_period: int = 10,
                                     kdj_fastk: int = 3, kdj_slowk: int = 3, kdj_slowd: int = 3,
                                     cci_period: int = 14, adx_period: int = 14,
                                     macd_fast: int = 6, macd_slow: int = 13, macd_signal: int = 5) -> Dict[str, np.ndarray]:
        if len(close) == 0:
            keys = [f'RSI{p}' for p in rsi_periods] + ['MOM', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'CCI', 'ADX', 'PLUS_DI', 'MINUS_DI', 'MACD', 'MACD_Signal', 'macd_signal_diff']
            return self._empty_indicators(keys)
        
        min_length = max(max(rsi_periods, default=0), mom_period, kdj_fastk, cci_period, adx_period, macd_slow)
        if len(close) < min_length:
            self.logger.warning(f"Data too short ({len(close)} < {min_length}), returning NaNs.")
            return self._empty_indicators(keys, len(close))
        
        high = self._to_float64(high)
        low = self._to_float64(low)
        close = self._to_float64(close)
        
        rsi_dict = {f'RSI{p}': talib.RSI(close, timeperiod=p).astype(np.float32) for p in rsi_periods}
        mom = talib.MOM(close, timeperiod=mom_period).astype(np.float32)
        kdj_k, kdj_d = talib.STOCH(high, low, close, fastk_period=kdj_fastk, slowk_period=kdj_slowk, slowd_period=kdj_slowd)
        kdj_k = kdj_k.astype(np.float32)
        kdj_d = kdj_d.astype(np.float32)
        kdj_j = (3 * kdj_k - 2 * kdj_d).astype(np.float32)
        cci = talib.CCI(high, low, close, timeperiod=cci_period).astype(np.float32)
        adx = talib.ADX(high, low, close, timeperiod=adx_period).astype(np.float32)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=adx_period).astype(np.float32)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=adx_period).astype(np.float32)
        macd, signal, hist = talib.MACD(close, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
        macd = macd.astype(np.float32)
        signal = signal.astype(np.float32)
        hist = hist.astype(np.float32)
        
        return {
            **rsi_dict, 'MOM': mom, 'KDJ_K': kdj_k, 'KDJ_D': kdj_d, 'KDJ_J': kdj_j,
            'CCI': cci, 'ADX': adx, 'PLUS_DI': plus_di, 'MINUS_DI': minus_di,
            'MACD': macd, 'MACD_Signal': signal, 'macd_signal_diff': hist
        }

    def calculate_volatility_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                       bb_period: int = 20, bb_nbdev_up: float = 2.5, bb_nbdev_dn: float = 2.5, 
                                       bb_matype: int = 0, atr_period: int = 14) -> Dict[str, np.ndarray]:
        if len(close) == 0:
            return self._empty_indicators(['BB_upper', 'BB_middle', 'BB_lower', 'ATR'])
        
        min_length = max(bb_period, atr_period)
        if len(close) < min_length:
            self.logger.warning(f"Data too short ({len(close)} < {min_length}), returning NaNs.")
            return self._empty_indicators(['BB_upper', 'BB_middle', 'BB_lower', 'ATR'], len(close))
        
        high = self._to_float64(high)
        low = self._to_float64(low)
        close = self._to_float64(close)
        
        upper, middle, lower = talib.BBANDS(close, timeperiod=bb_period, nbdevup=bb_nbdev_up, nbdevdn=bb_nbdev_dn, matype=bb_matype)
        return {
            'BB_upper': upper.astype(np.float32),
            'BB_middle': middle.astype(np.float32),
            'BB_lower': lower.astype(np.float32),
            'ATR': talib.ATR(high, low, close, timeperiod=atr_period).astype(np.float32)
        }

    def calculate_volume_indicators(self, volume: np.ndarray, close: np.ndarray, high: np.ndarray, low: np.ndarray,
                                   vol_ma_periods: List[int] = [5, 10], cmf_period: int = 20,
                                   mfi_period: int = 14) -> Dict[str, np.ndarray]:
        if len(volume) == 0:
            keys = [f'VOL_MA{p}' for p in vol_ma_periods] + ['OBV', 'CMF', 'MFI']
            return self._empty_indicators(keys)
        
        min_length = max(max(vol_ma_periods, default=0), cmf_period, mfi_period)
        if len(volume) < min_length:
            self.logger.warning(f"Data too short ({len(volume)} < {min_length}), returning NaNs.")
            return self._empty_indicators(keys, len(volume))
        
        volume = self._to_float64(volume)
        close = self._to_float64(close)
        high = self._to_float64(high)
        low = self._to_float64(low)
        
        vol_ma_dict = {f'VOL_MA{p}': talib.SMA(volume, timeperiod=p).astype(np.float32) for p in vol_ma_periods}
        obv = talib.OBV(close, volume).astype(np.float32)
        mfi = talib.MFI(high, low, close, volume, timeperiod=mfi_period).astype(np.float32)
        
        # Manual CMF calculation
        money_flow_multiplier = ((close - low) - (high - close)) / np.where(high != low, high - low, np.nan)
        money_flow_volume = money_flow_multiplier * volume
        cmf = np.zeros_like(close, dtype=np.float32)
        for i in range(len(close)):
            if i >= cmf_period - 1:
                cmf[i] = np.sum(money_flow_volume[i-cmf_period+1:i+1]) / np.sum(volume[i-cmf_period+1:i+1])
            else:
                cmf[i] = np.nan
        
        return {
            **vol_ma_dict,
            'OBV': obv,
            'CMF': cmf.astype(np.float32),
            'MFI': mfi
        }

    def calculate_sar(self, high: np.ndarray, low: np.ndarray, acceleration: float = 0.015, maximum: float = 0.15) -> np.ndarray:
        if len(high) == 0:
            return np.full(0, np.nan, dtype=np.float32)
        if len(high) < 2:
            self.logger.warning(f"Data too short ({len(high)} < 2), returning NaNs for SAR.")
            return np.full(len(high), np.nan, dtype=np.float32)
        
        high = self._to_float64(high)
        low = self._to_float64(low)
        return talib.SAR(high, low, acceleration=acceleration, maximum=maximum).astype(np.float32)

    def calculate_vwap(self, open_price: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        if len(close) == 0:
            return np.full(0, np.nan, dtype=np.float32)
        typical_price = (high + low + close + open_price) / 4
        vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
        return vwap.astype(np.float32)

    def calculate_stoch_rsi(self, close: np.ndarray, timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        if len(close) == 0:
            return np.full(0, np.nan, dtype=np.float32), np.full(0, np.nan, dtype=np.float32)
        if len(close) < timeperiod:
            self.logger.warning(f"Data too short ({len(close)} < {timeperiod}), returning NaNs for StochRSI.")
            return np.full(len(close), np.nan, dtype=np.float32), np.full(len(close), np.nan, dtype=np.float32)
        
        close = self._to_float64(close)
        rsi = talib.RSI(close, timeperiod=timeperiod)
        stoch_rsi_k, stoch_rsi_d = talib.STOCHF(rsi, rsi, rsi, fastk_period=fastk_period, fastd_period=fastd_period)
        return stoch_rsi_k.astype(np.float32), stoch_rsi_d.astype(np.float32)

    def calculate_candlestick_patterns(self, open_price: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        if len(close) == 0:
            return self._empty_indicators(['CDL_DOJI', 'CDL_HAMMER', 'CDL_MORNINGSTAR'])
        
        open_price = self._to_float64(open_price)
        high = self._to_float64(high)
        low = self._to_float64(low)
        close = self._to_float64(close)
        
        return {
            'CDL_DOJI': talib.CDLDOJI(open_price, high, low, close).astype(np.float32),
            'CDL_HAMMER': talib.CDLHAMMER(open_price, high, low, close).astype(np.float32),
            'CDL_MORNINGSTAR': talib.CDLMORNINGSTAR(open_price, high, low, close).astype(np.float32)
        }

    def calculate_derivative_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        close = df['close'].values.astype(np.float32)
        volume = df['volume'].values.astype(np.float32)
        indicators = {}
        
        # Bollinger Bands衍生特征
        bb_cols = ['BB_upper', 'BB_lower', 'BB_middle']
        if all(col in df.columns for col in bb_cols):
            upper, lower, middle = df[bb_cols].values.T.astype(np.float32)
            indicators['bb_position'] = ((close - lower) / (upper - lower)).astype(np.float32)
            indicators['bb_width'] = ((upper - lower) / middle).astype(np.float32)

        # KDJ衍生特征
        kdj_cols = ['KDJ_K', 'KDJ_D', 'KDJ_J']
        if all(col in df.columns for col in kdj_cols):
            k, d, j = df[kdj_cols].values.T.astype(np.float32)
            indicators['kdj_kd_diff'] = (k - d).astype(np.float32)
            indicators['kdj_j_extreme'] = ((j > 100) | (j < 0)).astype(np.float32)

        # 成交量均线比率
        vol_ma_cols = [col for col in df.columns if 'VOL_MA' in col]
        for vol_ma_col in vol_ma_cols:
            ma_values = df[vol_ma_col].values.astype(np.float32)
            indicators[f'{vol_ma_col}_ratio'] = (volume / ma_values).astype(np.float32)

        # 均线排列特征
        ma_cols = [col for col in df.columns if 'MA' in col and 'VOL' not in col]
        if len(ma_cols) >= 3:
            ma_values = df[ma_cols].values.astype(np.float32)
            indicators['ma_bullish'] = (ma_values[:, 0] > ma_values[:, 1]).astype(np.float32)
            indicators['ma_alignment'] = np.std(ma_values, axis=1).astype(np.float32)

        # 价格相对均线位置
        for ma_col in ma_cols:
            ma_values = df[ma_col].values.astype(np.float32)
            indicators[f'price_vs_{ma_col}'] = (close / ma_values - 1).astype(np.float32)

        # ATR比率
        if 'ATR' in df.columns:
            indicators['atr_ratio'] = (df['ATR'].values / close).astype(np.float32)

        return indicators

    def calculate_all_indicators(self, fillna_method: Optional[str] = 'ffill',
                                indicators_to_compute: List[str] = ['all'],
                                ma_periods: List[int] = [5, 10, 20], use_ema: bool = False,
                                rsi_periods: List[int] = [6, 12], mom_period: int = 10,
                                kdj_fastk: int = 5, kdj_slowk: int = 3, kdj_slowd: int = 3, cci_period: int = 14,
                                bb_period: int = 20, bb_nbdev_up: float = 2.5, bb_nbdev_dn: float = 2.5, bb_matype: int = 0,
                                atr_period: int = 14, vol_ma_periods: List[int] = [5, 10],
                                macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                                adx_period: int = 14, cmf_period: int = 20,
                                sar_acceleration: float = 0.015, sar_maximum: float = 0.15) -> pd.DataFrame:
        if self._df is None or self._df.empty:
            self.logger.warning("Empty DataFrame. Returning empty result.")
            return pd.DataFrame()
        
        required_columns = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in self._df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        def compute_group(group: pd.DataFrame) -> pd.DataFrame:
            ti = TechnicalIndicators(group)
            result = group.copy()
            close = group['close'].values
            high = group['high'].values
            low = group['low'].values
            volume = group['volume'].values
            open_ = group['open'].values
            
            indicators = {}
            if 'all' in indicators_to_compute or 'ma' in indicators_to_compute:
                indicators.update(ti.calculate_moving_averages(close, ma_periods, use_ema))
                gc.collect()
            if 'all' in indicators_to_compute or 'momentum' in indicators_to_compute:
                indicators.update(ti.calculate_momentum_indicators(high, low, close, rsi_periods, mom_period,
                                                                  kdj_fastk, kdj_slowk, kdj_slowd, cci_period, adx_period,
                                                                  macd_fast, macd_slow, macd_signal))
                gc.collect()
            if 'all' in indicators_to_compute or 'volatility' in indicators_to_compute:
                indicators.update(ti.calculate_volatility_indicators(high, low, close, bb_period, bb_nbdev_up, bb_nbdev_dn, bb_matype, atr_period))
                gc.collect()
            if 'all' in indicators_to_compute or 'volume' in indicators_to_compute:
                indicators.update(ti.calculate_volume_indicators(volume, close, high, low, vol_ma_periods, cmf_period))
                gc.collect()
            if 'all' in indicators_to_compute or 'candlestick' in indicators_to_compute:
                indicators.update(ti.calculate_candlestick_patterns(open_, high, low, close))
                gc.collect()
            if 'all' in indicators_to_compute or 'sar' in indicators_to_compute:
                indicators['SAR'] = ti.calculate_sar(high, low, sar_acceleration, sar_maximum)
                gc.collect()
            indicators['VWAP'] = ti.calculate_vwap(open_, high, low, close, volume)
            stoch_rsi_k, stoch_rsi_d = ti.calculate_stoch_rsi(close)
            indicators['StochRSI_K'] = stoch_rsi_k
            indicators['StochRSI_D'] = stoch_rsi_d

            # 添加衍生特征
            indicators.update(ti.calculate_derivative_features(result))
            gc.collect()

            for name, values in indicators.items():
                result[name] = values
            
            # Hybrid NaN filling: forward-fill then median
            indicator_cols = [col for col in result.columns if col not in required_columns]
            if fillna_method == 'hybrid':
                result[indicator_cols] = result[indicator_cols].ffill()
                result[indicator_cols] = result[indicator_cols].fillna(result[indicator_cols].median()).astype(np.float32)
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
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(compute_group, groups))
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