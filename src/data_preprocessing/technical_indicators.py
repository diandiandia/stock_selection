import pandas as pd
import talib
import numpy as np
from typing import List, Optional
import torch

class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        """
        基于专业分析优化的技术指标计算器
        专注于短期股票预测模型（隔夜策略）
        """
        self.df = df.copy()
        self._validate_and_prepare()
        
    def _validate_and_prepare(self):
        """数据验证和预处理"""
        required_cols = ['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'turnover']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")
            
        # 确保数据类型正确
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        
        # 按股票代码和日期排序
        self.df = self.df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
    def calculate_core_features(self) -> pd.DataFrame:
        """
        计算12个核心技术指标（必须指标）
        专为短期隔夜策略优化
        """
        result_dfs = []
        
        for ts_code, group in self.df.groupby('ts_code'):
            group = group.copy()
            
            # === 趋势指标（短期周期优化）===
            group['ma5'] = talib.SMA(group['close'].values, timeperiod=5)
            group['ma10'] = talib.SMA(group['close'].values, timeperiod=10)
            group['ema8'] = talib.EMA(group['close'].values, timeperiod=8)
            group['ma_cross_diff'] = group['ma5'] - group['ma10']
            
            # === 动量指标（短期参数）===
            group['rsi'] = talib.RSI(group['close'].values, timeperiod=14)
            
            # KDJ优化：使用标准短期参数
            slowk, slowd = talib.STOCH(
                group['high'].values, group['low'].values, group['close'].values,
                fastk_period=9, slowk_period=3, slowd_period=3
            )
            group['kdj_diff'] = slowk - slowd
            
            # MACD标准参数
            macd, macd_signal, macd_hist = talib.MACD(
                group['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            group['macd'] = macd
            group['macd_signal'] = macd_signal
            group['macd_hist'] = macd_hist
            
            # === 成交量指标 ===
            group['volume_ratio'] = group['volume'] / group['volume'].shift(1)
            group['net_mf_amount'] = (group['close'] - group['open']) * group['volume']
            
            # === 波动性指标（短期周期）===
            group['atr'] = talib.ATR(
                group['high'].values, group['low'].values, group['close'].values,
                timeperiod=10  # 从14日优化为10日，更适合短期
            )
            group['intraday_range'] = (group['high'] - group['low']) / group['close'].shift(1)
            
            # === 价格行为 ===
            group['gap'] = (group['open'] - group['close'].shift(1)) / group['close'].shift(1)
            
            # === 流动性指标（修复circ_mv问题）===
            if 'circ_mv' in group.columns and group['circ_mv'].notna().any():
                group['turnover_rate'] = group['volume'] / group['circ_mv'] * 100
            else:
                # 使用现有换手率或计算简化版本
                group['turnover_rate'] = group['turnover']
            
            result_dfs.append(group)
        
        self.df = pd.concat(result_dfs, ignore_index=True)
        
        # 清理NaN值
        core_cols = [
            'ma5', 'ma10', 'ema8', 'ma_cross_diff', 'rsi', 'kdj_diff',
            'macd', 'macd_signal', 'macd_hist', 'volume_ratio',
            'net_mf_amount', 'atr', 'intraday_range', 'gap', 'turnover_rate'
        ]
        self.df = self.df.dropna(subset=[col for col in core_cols if col in self.df.columns])
        
        return self.df
    
    def calculate_enhanced_features(self) -> pd.DataFrame:
        """
        计算增强指标（建议添加的布林带、OBV、CCI）
        """
        result_dfs = []
        
        for ts_code, group in self.df.groupby('ts_code'):
            group = group.copy()
            
            # === 布林带（20日标准参数）===
            upper, middle, lower = talib.BBANDS(
                group['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            group['bb_upper'] = upper
            group['bb_middle'] = middle
            group['bb_lower'] = lower
            group['bb_position'] = (group['close'] - lower) / (upper - lower)
            group['bb_width'] = (upper - lower) / middle
            
            # === OBV能量潮 ===
            group['obv'] = talib.OBV(group['close'].values, group['volume'].values)
            group['obv_ratio'] = group['obv'] / group['obv'].shift(1)
            
            # === CCI商品通道指标（短期周期）===
            group['cci'] = talib.CCI(
                group['high'].values, group['low'].values, group['close'].values,
                timeperiod=14
            )
            
            # === 价格位置指标 ===
            group['price_ma_ratio'] = group['close'] / group['ma10']
            group['price_ema_ratio'] = group['close'] / group['ema8']
            
            result_dfs.append(group)
        
        self.df = pd.concat(result_dfs, ignore_index=True)
        return self.df
    
    def calculate_momentum_features(self) -> pd.DataFrame:
        """
        计算动量相关特征
        """
        result_dfs = []
        
        for ts_code, group in self.df.groupby('ts_code'):
            group = group.copy()
            
            # === 动量变化率 ===
            group['momentum_5'] = talib.MOM(group['close'].values, timeperiod=5)
            group['roc_5'] = talib.ROC(group['close'].values, timeperiod=5)
            
            # === 威廉指标（短期超买超卖）===
            group['willr'] = talib.WILLR(
                group['high'].values, group['low'].values, group['close'].values,
                timeperiod=10
            )
            
            result_dfs.append(group)
        
        self.df = pd.concat(result_dfs, ignore_index=True)
        return self.df
    
    def prepare_ml_dataset(self, target_threshold: float = 0.01, 
                          include_enhanced: bool = False) -> pd.DataFrame:
        """
        准备机器学习数据集
        
        Args:
            target_threshold: 目标阈值（默认1%涨幅）
            include_enhanced: 是否包含增强特征
        """
        # 计算核心特征
        self.calculate_core_features()
        
        if include_enhanced:
            self.calculate_enhanced_features()
            self.calculate_momentum_features()
        
        # 准备特征列
        feature_cols = [
            'ma5', 'ma10', 'ema8', 'ma_cross_diff', 'rsi', 'kdj_diff',
            'macd', 'macd_signal', 'macd_hist', 'volume_ratio',
            'net_mf_amount', 'atr', 'intraday_range', 'gap', 'turnover_rate'
        ]
        
        if include_enhanced:
            enhanced_cols = [
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width',
                'obv', 'obv_ratio', 'cci', 'price_ma_ratio', 'price_ema_ratio',
                'momentum_5', 'roc_5', 'willr'
            ]
            feature_cols.extend([col for col in enhanced_cols if col in self.df.columns])
        
        # 创建结果DataFrame
        result_dfs = []
        
        for ts_code, group in self.df.groupby('ts_code'):
            group = group.copy()
            
            # 创建目标变量（下一天收益率）
            group['next_day_return'] = group['close'].pct_change().shift(-1)
            group['target'] = (group['next_day_return'] > target_threshold).astype(int)
            
            # 滞后所有特征1天（防止未来数据泄露）
            feature_data = group[['ts_code', 'trade_date', 'next_day_return', 'target']].copy()
            
            for col in feature_cols:
                if col in group.columns:
                    feature_data[f'{col}_lag1'] = group[col].shift(1)
            
            result_dfs.append(feature_data)
        
        final_df = pd.concat(result_dfs, ignore_index=True)
        
        # 清理数据
        lag_cols = [col for col in final_df.columns if col.endswith('_lag1')]
        final_df = final_df.dropna(subset=lag_cols + ['target'])
        
        return final_df
    
    def get_feature_importance_candidates(self, ml_data) -> pd.DataFrame:
        """
        获取特征重要性分析候选数据
        """
        # 分离特征和目标
        feature_cols = [col for col in ml_data.columns if col.endswith('_lag1')]
        X = ml_data[feature_cols]
        y = ml_data['target']
        
        return X, y, ml_data
    
    def prepare_gru_dataset(self, seq_len=10, target_threshold=0.01, include_enhanced=False):
        """
        生成适配 GRU/LSTM 的三维数据集
        返回: X_tensor, y_tensor, feature_names
        """
        # 先用原来的 prepare_ml_dataset 得到特征和标签
        ml_df = self.prepare_ml_dataset(target_threshold=target_threshold, include_enhanced=include_enhanced)
        feature_cols = [col for col in ml_df.columns if col.endswith('_lag1')]
        
        X_list, y_list = [], []
        
        # 按股票代码分组，保证时间序列连续
        for ts_code, group in ml_df.groupby('ts_code'):
            group = group.sort_values('trade_date')
            X_vals = group[feature_cols].values
            y_vals = group['target'].values
            
            # 滑动窗口
            for i in range(len(group) - seq_len):
                X_seq = X_vals[i:i+seq_len]
                y_target = y_vals[i+seq_len]  # 预测 seq_len 之后的一天
                X_list.append(X_seq)
                y_list.append(y_target)
        
        # 转成 numpy
        X_array = np.array(X_list, dtype=np.float32)  # (batch, seq_len, feature_dim)
        y_array = np.array(y_list, dtype=np.float32)  # (batch,)
        
        # 转成 PyTorch tensor（如果用 TensorFlow/Keras，就不需要转）
        
        X_tensor = torch.tensor(X_array)
        y_tensor = torch.tensor(y_array)
        
        return X_tensor, y_tensor, feature_cols


# 使用示例
if __name__ == "__main__":
    # 示例用法
    # df = your_dataframe  # 包含基础数据
    # ti = TechnicalIndicators(df)
    # 
    # # 计算核心特征
    # core_features = ti.calculate_core_features()
    # 
    # # 准备ML数据集
    # ml_data = ti.prepare_ml_dataset(target_threshold=0.01, include_enhanced=True)
    # 
    # # 获取特征重要性数据
    # X, y, full_data = ti.get_feature_importance_candidates()
    pass