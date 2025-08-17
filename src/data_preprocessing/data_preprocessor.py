from typing import Tuple, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from src.utils.log_helper import LogHelper

class DataPreprocessor:
    """
    专为HybridPredictor准备训练数据的数据预处理器
    接收TechnicalIndicators的输出，专注于数据准备和格式化
    不再计算技术指标，仅做数据清洗、特征选择和时序数据构建
    """
    
    def __init__(self, 
                 lookback_window: int = 20,
                 prediction_horizon: int = 1,
                 feature_scaler_type: str = 'standard',
                 target_scaler_type: str = 'standard',
                 test_size: float = 0.2,
                 validation_split: float = 0.1):
        """
        初始化数据预处理器
        
        Args:
            lookback_window: 回看窗口大小（用于时序特征）
            prediction_horizon: 预测周期（T+1就是1）
            feature_scaler_type: 特征缩放器类型 ('standard' 或 'minmax')
            target_scaler_type: 目标变量缩放器类型 ('standard' 或 'minmax')
            test_size: 测试集比例
            validation_split: 验证集比例
        """
        self.logger = LogHelper.get_logger(__name__)
        
        # 配置参数
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.validation_split = validation_split
        
        # 缩放器
        self.feature_scaler = self._get_scaler(feature_scaler_type)
        self.target_scaler = self._get_scaler(target_scaler_type)
        
        # 特征信息
        self.feature_columns = None
        self.target_column = 'target_return_1d'
        
        # 数据缓存
        self.data_info = {}
        
    def _get_scaler(self, scaler_type: str):
        """获取缩放器实例"""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基于TechnicalIndicators的输出创建数据集
        专注于数据清洗和格式转换，不进行技术指标计算
        """
        result_df = df.copy()
    
        # 确保数据按时间排序
        result_df = result_df.sort_values(['ts_code', 'trade_date'])
    
        # 技术信号综合（保留为数据预处理步骤）
        result_df = self._create_technical_signals(result_df)
    
        return result_df
    
    def _create_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术信号综合指标"""
        result_df = df.copy()
        
        # 买入信号计数
        buy_signals = []
        
        # RSI超卖信号
        if 'RSI6' in result_df.columns:
            buy_signals.append((result_df['RSI6'] < 30).astype(int))
        
        # MACD金叉信号
        if 'macd_signal_diff' in result_df.columns:
            macd_golden_cross = ((result_df['macd_signal_diff'] > 0) & 
                               (result_df['macd_signal_diff'].shift(1) <= 0)).astype(int)
            buy_signals.append(macd_golden_cross)
        
        # 布林带突破下轨信号
        if 'bb_position' in result_df.columns:
            bb_oversold = (result_df['bb_position'] < 0.2).astype(int)
            buy_signals.append(bb_oversold)
        
        # KDJ超卖信号
        if 'KDJ_J' in result_df.columns:
            kdj_oversold = (result_df['KDJ_J'] < 20).astype(int)
            buy_signals.append(kdj_oversold)
        
        if buy_signals:
            result_df['buy_signal_count'] = pd.concat(buy_signals, axis=1).sum(axis=1)
        
        return result_df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建T+1收益率目标变量
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            添加目标变量的DataFrame
        """
        result_df = df.copy()
        
        # 计算T+1收益率
        result_df[self.target_column] = result_df.groupby('ts_code')['close'].shift(-self.prediction_horizon) / result_df['close'] - 1
        
        return result_df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        选择用于训练的特征
        
        Args:
            df: 包含所有特征的DataFrame
            
        Returns:
            选定的特征列名列表
        """
        # 排除非特征列
        exclude_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', self.target_column]
        
        # 选择数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 移除包含太多NaN的特征
        valid_features = []
        for col in feature_cols:
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio < 0.1:  # NaN比例小于10%
                valid_features.append(col)
        
        self.feature_columns = valid_features
        self.logger.info(f"Selected {len(valid_features)} features")
        
        return valid_features
    
    def prepare_single_stock(self, df: pd.DataFrame, ts_code: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        为单只股票准备时序数据
        
        Args:
            df: 包含技术指标的DataFrame
            ts_code: 股票代码
            
        Returns:
            X_sequence: 时序特征 (samples, timesteps, features)
            y: 目标变量 (samples,)
        """
        stock_df = df[df['ts_code'] == ts_code].copy()
        
        if len(stock_df) < self.lookback_window + 1:
            return np.array([]), np.array([])
        
        # 选择特征
        features = self.select_features(stock_df)
        
        # 创建时序数据
        X_sequence = []
        y = []
        
        for i in range(self.lookback_window, len(stock_df)):
            # 检查是否有足够的历史数据
            sequence = stock_df[features].iloc[i-self.lookback_window:i].values
            target = stock_df[self.target_column].iloc[i]
            
            # 跳过包含NaN的数据
            if not np.isnan(sequence).any() and not np.isnan(target):
                X_sequence.append(sequence)
                y.append(target)
        
        return np.array(X_sequence), np.array(y)
    
    def prepare_all_stocks(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        为所有股票准备训练数据
        
        Args:
            df: 包含技术指标的DataFrame
            
        Returns:
            X_sequence: 时序特征 (samples, timesteps, features)
            y: 目标变量 (samples,)
            stock_codes: 股票代码列表
            dates: 日期列表
        """
        self.logger.info("Preparing data for all stocks...")
        
        # 1. 创建特征
        df = self.create_features(df)
        
        # 2. 创建目标变量
        df = self.create_target(df)
        
        # 3. 选择特征
        features = self.select_features(df)
        
        # 4. 收集所有股票的时序数据
        X_all = []
        y_all = []
        stock_codes = []
        dates = []
        
        for ts_code in df['ts_code'].unique():
            X_seq, y_seq = self.prepare_single_stock(df, ts_code)
            
            if len(X_seq) > 0:
                X_all.append(X_seq)
                y_all.extend(y_seq)
                
                # 记录股票代码和对应日期
                stock_dates = df[df['ts_code'] == ts_code]['trade_date'].iloc[self.lookback_window:].tolist()
                stock_codes.extend([ts_code] * len(y_seq))
                dates.extend(stock_dates)
        
        if not X_all:
            raise ValueError("No valid data found for training")
        
        # 合并所有数据
        X_sequence = np.concatenate(X_all, axis=0)
        y = np.array(y_all)
        
        self.logger.info(f"Prepared {len(y)} samples from {len(df['ts_code'].unique())} stocks")
        
        return X_sequence, y, stock_codes, dates
    
    def fit_scalers(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        拟合特征和目标变量的缩放器
        
        Args:
            X: 时序特征 (samples, timesteps, features)
            y: 目标变量 (samples,)
        """
        self.logger.info("开始数据清洗和缩放器拟合...")
        
        # 重塑X以拟合缩放器
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # 检查并清洗数据
        # 1. 检查无穷大值和NaN值
        X_finite = np.isfinite(X_reshaped)
        y_finite = np.isfinite(y)
        
        if not np.all(X_finite):
            inf_count = np.sum(~X_finite)
            self.logger.warning(f"发现{inf_count}个特征数据中的无穷大或NaN值，将进行替换处理")
            # 用中位数替换无穷大值和NaN值
            for col in range(X_reshaped.shape[1]):
                col_data = X_reshaped[:, col]
                finite_mask = np.isfinite(col_data)
                if np.any(finite_mask):
                    median_val = np.median(col_data[finite_mask])
                    col_data[~finite_mask] = median_val
        
        if not np.all(y_finite):
            inf_count = np.sum(~y_finite)
            self.logger.warning(f"发现{inf_count}个目标变量中的无穷大或NaN值，将进行替换处理")
            median_y = np.median(y[np.isfinite(y)])
            y[~y_finite] = median_y
        
        # 2. 检查异常大值（超出float64范围）
        float64_max = np.finfo(np.float64).max
        float64_min = np.finfo(np.float64).min
        
        # 限制异常大值
        X_clipped = np.clip(X_reshaped, float64_min, float64_max)
        
        # 3. 检查极端异常值（使用IQR方法）
        for col in range(X_clipped.shape[1]):
            col_data = X_clipped[:, col]
            finite_mask = np.isfinite(col_data)
            if np.sum(finite_mask) > 10:  # 确保有足够的数据
                Q1 = np.percentile(col_data[finite_mask], 1)
                Q3 = np.percentile(col_data[finite_mask], 99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # 替换极端异常值
                extreme_mask = (col_data < lower_bound) | (col_data > upper_bound)
                if np.any(extreme_mask):
                    self.logger.debug(f"列{col}发现{np.sum(extreme_mask)}个极端异常值")
                    col_data[extreme_mask] = np.median(col_data[finite_mask])
        
        # 4. 检查并处理目标变量异常值
        y_clipped = np.clip(y, float64_min, float64_max)
        
        # 5. 再次验证数据完整性
        if np.any(~np.isfinite(X_clipped)) or np.any(~np.isfinite(y_clipped)):
            raise ValueError("数据清洗后仍存在无效值")
        
        # 拟合特征缩放器
        self.feature_scaler.fit(X_clipped)
        
        # 拟合目标变量缩放器
        y_reshaped = y_clipped.reshape(-1, 1)
        self.target_scaler.fit(y_reshaped)
        
        self.logger.info("数据清洗完成，缩放器拟合成功")
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        转换特征
        
        Args:
            X: 原始特征 (samples, timesteps, features)
            
        Returns:
            标准化后的特征
        """
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.feature_scaler.transform(X_reshaped)
        return X_scaled.reshape(X.shape)
    
    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """转换目标变量"""
        y_reshaped = y.reshape(-1, 1)
        y_scaled = self.target_scaler.transform(y_reshaped)
        return y_scaled.flatten()
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """逆转换目标变量"""
        y_reshaped = y_scaled.reshape(-1, 1)
        y_original = self.target_scaler.inverse_transform(y_reshaped)
        return y_original.flatten()
    
    def create_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                               split_method: str = 'time_series') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        创建训练测试集分割
        
        Args:
            X: 特征数据
            y: 目标变量
            split_method: 分割方法 ('time_series', 'random', 'by_stock')
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if split_method == 'time_series':
            # 时间序列分割
            split_index = int(len(X) * (1 - self.test_size))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
        elif split_method == 'random':
            # 随机分割
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42, shuffle=True
            )
            
        elif split_method == 'by_stock':
            # 按股票分割（避免数据泄露）
            # 这里简化实现，实际应该按股票代码分割
            split_index = int(len(X) * (1 - self.test_size))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
        
        self.logger.info(f"Train/test split completed: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return self.feature_columns or []
    
    def get_data_info(self) -> Dict:
        """获取数据信息"""
        return {
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'scaler_types': {
                'feature': type(self.feature_scaler).__name__,
                'target': type(self.target_scaler).__name__
            }
        }
    
    def save_preprocessor(self, path: str) -> None:
        """保存预处理器状态"""
        save_data = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'data_info': self.data_info
        }
        joblib.dump(save_data, path)
        self.logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str) -> None:
        """加载预处理器状态"""
        save_data = joblib.load(path)
        self.feature_scaler = save_data['feature_scaler']
        self.target_scaler = save_data['target_scaler']
        self.feature_columns = save_data['feature_columns']
        self.lookback_window = save_data['lookback_window']
        self.prediction_horizon = save_data['prediction_horizon']
        self.data_info = save_data['data_info']
        self.logger.info(f"Preprocessor loaded from {path}")