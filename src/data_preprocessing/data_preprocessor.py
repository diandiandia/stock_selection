from typing import Tuple, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import joblib
from src.model.lstm_lgbm_predictor import LSTMLGBMPredictor
from src.utils.log_helper import LogHelper
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from concurrent.futures import ThreadPoolExecutor

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
                 validation_split: float = 0.1,
                 imputation_strategy: str = 'hybrid',
                 outlier_detection: str = 'zscore',
                 outlier_threshold: float = 5.0,
                 shap_subset_size: int = 10000):
        """
        初始化数据预处理器
        
        Args:
            lookback_window: 回看窗口大小（用于时序特征）
            prediction_horizon: 预测周期（T+1就是1）
            feature_scaler_type: 特征缩放器类型 ('standard' 或 'minmax')
            target_scaler_type: 目标变量缩放器类型 ('standard' 或 'minmax')
            test_size: 测试集比例
            validation_split: 验证集比例
            imputation_strategy: 缺失值填充策略 ('mean', 'median', 'mode', 'hybrid')
            outlier_detection: 异常值检测方法 ('iqr', 'zscore', 'none')
            outlier_threshold: 异常值检测阈值
            shap_subset_size: 用于SHAP特征选择的子集大小
        """
        self.logger = LogHelper.get_logger(__name__)
        
        # 配置参数
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.validation_split = validation_split
        self.imputation_strategy = imputation_strategy
        self.outlier_detection = outlier_detection
        self.outlier_threshold = outlier_threshold
        self.shap_subset_size = shap_subset_size
        
        # 参数验证
        if self.imputation_strategy not in ['mean', 'median', 'mode', 'hybrid']:
            raise ValueError(f"无效的缺失值填充策略: {self.imputation_strategy}")
        if self.outlier_detection not in ['iqr', 'zscore', 'none']:
            raise ValueError(f"无效的异常值检测方法: {self.outlier_detection}")
        if self.outlier_threshold <= 0:
            raise ValueError(f"异常值阈值必须为正数, 实际值: {self.outlier_threshold}")
        if not (0 < self.test_size < 1):
            raise ValueError(f"测试集比例必须在(0, 1)之间, 实际值: {self.test_size}")
        if not (0 < self.validation_split < 1):
            raise ValueError(f"验证集比例必须在(0, 1)之间, 实际值: {self.validation_split}")
        if self.lookback_window <= 0:
            raise ValueError(f"回溯窗口大小必须为正数, 实际值: {self.lookback_window}")
        if self.shap_subset_size <= 0:
            raise ValueError(f"SHAP子集大小必须为正数, 实际值: {self.shap_subset_size}")
        
        # 缩放器
        self.feature_scaler = self._get_scaler(feature_scaler_type)
        self.target_scaler = self._get_scaler(target_scaler_type)
        
        # 特征信息
        self.feature_columns = None
        self.target_column = 'target_return_1d'
        
        # 数据缓存
        self.data_info = {}
        self.is_fitted = False
        
        # 数据预处理参数
        self.imputation_values = None
        self.outlier_lower_bounds = None
        self.outlier_upper_bounds = None
        
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
        self.logger.info(f"开始创建特征，输入数据形状: {df.shape}")
        result_df = df.copy()
    
        # 确保数据按时间排序
        result_df = result_df.sort_values(['ts_code', 'trade_date'])
        self.logger.info(f"数据排序完成，股票数量: {result_df['ts_code'].nunique()}")
    
        # 技术信号综合（保留为数据预处理步骤）
        result_df = self._create_technical_signals(result_df)
        self.logger.info(f"技术信号创建完成，新增特征数量: {len(result_df.columns) - len(df.columns)}")
    
        return result_df
    
    def _create_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术信号综合指标（使用TechnicalIndicators已计算的指标）"""
        result_df = df.copy()
        
        # 买入信号计数（直接使用TechnicalIndicators已计算的指标）
        buy_signals = []
        
        # RSI超卖信号（使用TechnicalIndicators计算的RSI6）
        if 'RSI6' in result_df.columns:
            buy_signals.append((result_df['RSI6'] < 30).astype(int))
        
        # MACD金叉信号（使用TechnicalIndicators计算的macd_signal_diff）
        if 'macd_signal_diff' in result_df.columns:
            macd_golden_cross = ((result_df['macd_signal_diff'] > 0) & 
                               (result_df['macd_signal_diff'].shift(1) <= 0)).astype(int)
            buy_signals.append(macd_golden_cross)
        
        # 布林带突破下轨信号（使用TechnicalIndicators计算的bb_position）
        if 'bb_position' in result_df.columns:
            bb_oversold = (result_df['bb_position'] < 0.2).astype(int)
            buy_signals.append(bb_oversold)
        
        # KDJ超卖信号（使用TechnicalIndicators计算的KDJ_J）
        if 'KDJ_J' in result_df.columns:
            kdj_oversold = (result_df['KDJ_J'] < 20).astype(int)
            buy_signals.append(kdj_oversold)
        
        if buy_signals:
            result_df['buy_signal_count'] = pd.concat(buy_signals, axis=1).sum(axis=1)
        
        return result_df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建T+1收益率目标变量
        """
        self.logger.info(f"开始创建目标变量，预测周期: {self.prediction_horizon}")
        result_df = df.copy()
        
        # 计算T+1收益率
        result_df[self.target_column] = result_df.groupby('ts_code')['close'].shift(-self.prediction_horizon) / result_df['close'] - 1
        
        # 记录目标变量的统计信息
        target_data = result_df[self.target_column].dropna()
        self.logger.info(f"目标变量创建完成，有效样本数: {len(target_data)}, 均值: {target_data.mean():.4f}, 标准差: {target_data.std():.4f}")
        
        return result_df
    
    def select_features(self, df: pd.DataFrame, model: Optional['LSTMLGBMPredictor'] = None) -> List[str]:
        """
        选择用于训练的特征，结合缺失值分析、多重共线性检测和SHAP特征重要性
        """
        self.logger.info("开始特征选择...")
        
        # 排除非特征列
        exclude_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', self.target_column]
        
        # 选择数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        self.logger.info(f"原始特征数量: {len(feature_cols)}")
        
        # 步骤1: 移除包含太多NaN的特征（向量化优化）
        nan_ratios = df[feature_cols].isna().mean()
        valid_features = nan_ratios[nan_ratios < 0.2].index.tolist()
        self.logger.info(f"缺失值过滤后特征数量: {len(valid_features)}")
        
        # 步骤2: 移除多重共线性特征
        final_features = self._remove_multicollinearity(df[valid_features], vif_threshold=7.0)
        self.logger.info(f"多重共线性检测后特征数量: {len(final_features)}")
        
        # 步骤3: 如果提供了模型，则使用SHAP特征重要性进一步筛选
        if model and model.feature_importance_cache:
            try:
                top_features = model.get_top_features(top_n=30)
                lgbm_features = [valid_features[idx] for idx, _ in top_features.get('lgbm', []) if idx < len(valid_features)]
                lstm_features = [valid_features[idx] for idx, _ in top_features.get('lstm', []) if idx < len(valid_features)]
                shap_features = list(set(lgbm_features + lstm_features) & set(final_features))
                if shap_features:
                    final_features = shap_features[:30]
                    self.logger.info(f"SHAP特征选择后特征数量: {len(final_features)}")
                else:
                    self.logger.warning("SHAP特征选择返回空特征集，使用VIF筛选结果")
            except Exception as e:
                self.logger.warning(f"SHAP特征选择失败: {str(e)}, 使用VIF筛选结果")
        
        self.feature_columns = final_features
        self.logger.info(f"特征选择完成: 原始{len(feature_cols)} → 缺失值过滤{len(valid_features)} → 去多重共线性{len(final_features)}")
        
        return final_features
    
    def _remove_multicollinearity(self, feature_df: pd.DataFrame, vif_threshold: float = 7.0) -> List[str]:
        """使用VIF移除多重共线性特征 - 优化版: 子采样加速"""
        self.logger.info(f"开始多重共线性检测，输入特征数量: {len(feature_df.columns)}, VIF阈值: {vif_threshold}")
        sample_size = min(10000, len(feature_df))
        features = feature_df.sample(n=sample_size, random_state=42).copy()
        selected_features = []
        
        # 预过滤高相关特征
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        if to_drop:
            self.logger.info(f"预过滤高相关特征: {len(to_drop)} 个 (相关系数>0.8)")
        features = features.drop(columns=to_drop)
        
        # 计算VIF并迭代移除高VIF特征
        iteration = 0
        while len(features.columns) > 0 and len(selected_features) < 30:
            iteration += 1
            vif_data = pd.DataFrame()
            vif_data["feature"] = features.columns
            
            def calculate_vif(args):
                features, idx = args
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        return variance_inflation_factor(features.values, idx)
                except:
                    return np.inf
            
            with ThreadPoolExecutor() as executor:
                vifs = list(executor.map(calculate_vif, [(features, i) for i in range(features.shape[1])]))
            
            vif_data["VIF"] = vifs
            vif_data = vif_data.sort_values("VIF")
            
            self.logger.debug(f"第{iteration}轮VIF计算: 最小VIF={vif_data['VIF'].iloc[0]:.2f}, 最大VIF={vif_data['VIF'].iloc[-1]:.2f}")
            
            if vif_data["VIF"].iloc[0] > vif_threshold:
                if not selected_features:
                    selected_features.append(vif_data["feature"].iloc[0])
                    self.logger.info(f"所有VIF>{vif_threshold}, 保留最小VIF特征: {vif_data['feature'].iloc[0]}")
                break
            
            selected_feature = vif_data["feature"].iloc[0]
            selected_features.append(selected_feature)
            features = features.drop(columns=[selected_feature])
        
        self.logger.info(f"多重共线性检测完成: 最终选择 {len(selected_features)} 个特征")
        return [col for col in feature_df.columns if col in selected_features[:30]]
    
    def prepare_single_stock(self, df: pd.DataFrame, ts_code: str, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """为单支股票准备时序数据"""
        stock_df = df[df['ts_code'] == ts_code].copy()
        if len(stock_df) < self.lookback_window + self.prediction_horizon:
            self.logger.warning(f"股票 {ts_code} 数据不足: {len(stock_df)} < {self.lookback_window + self.prediction_horizon}")
            return np.array([]), np.array([])
        
        X_stock = []
        y_stock = []
        for i in range(self.lookback_window, len(stock_df) - self.prediction_horizon + 1):
            X_stock.append(stock_df[features].iloc[i-self.lookback_window:i].values)
            y_stock.append(stock_df[self.target_column].iloc[i+self.prediction_horizon-1])
        
        X_stock = np.array(X_stock)
        y_stock = np.array(y_stock)
        
        return X_stock, y_stock
    
    def create_train_test_split(self, X_sequence: np.ndarray, y: np.ndarray, stock_codes: np.ndarray, 
                              split_method: str = 'time_series') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """创建训练和测试数据集"""
        self.logger.info(f"开始创建训练测试集，样本数量: {len(X_sequence)}")
        
        if split_method == 'random':
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequence, y, test_size=self.test_size, random_state=42
            )
        elif split_method == 'time_series':
            tscv = TimeSeriesSplit(n_splits=int(1/self.test_size))
            train_idx, test_idx = list(tscv.split(X_sequence))[-1]
            X_train, X_test = X_sequence[train_idx], X_sequence[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            raise ValueError(f"不支持的分割方法: {split_method}")
        
        self.logger.info(f"训练测试分割完成: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """转换特征数据"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_cleaned = self._clean_data(X_reshaped)
        X_scaled = self.feature_scaler.transform(X_cleaned)
        return X_scaled.reshape(X.shape)
    
    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """转换目标变量"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        y_cleaned = y.copy()
        y_finite = np.isfinite(y_cleaned)
        if not np.all(y_finite):
            y_cleaned[~y_finite] = self.target_imputation_value
        y_cleaned = np.clip(y_cleaned, self.target_outlier_lower, self.target_outlier_upper)
        return self.target_scaler.transform(y_cleaned.reshape(-1, 1)).flatten()
    
    def _clean_data(self, X: np.ndarray) -> np.ndarray:
        """清洗数据，处理缺失值和异常值"""
        X_cleaned = X.copy()
        for col in range(X.shape[1]):
            mask = ~np.isfinite(X_cleaned[:, col])
            if np.any(mask):
                X_cleaned[mask, col] = self.imputation_values.get(col, 0.0)
            X_cleaned[:, col] = np.clip(
                X_cleaned[:, col],
                self.outlier_lower_bounds.get(col, -np.inf),
                self.outlier_upper_bounds.get(col, np.inf)
            )
        return X_cleaned
    
    def fit(self, df: pd.DataFrame, model: Optional['LSTMLGBMPredictor'] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        一站式数据预处理和特征选择
        """
        self.logger.info("开始数据预处理和拟合...")
        result_df = self.create_features(df)
        result_df = self.create_target(result_df)
        
        # 选择特征
        self.feature_columns = self.select_features(result_df, model)
        self.logger.info(f"最终选择的特征: {self.feature_columns}")
        
        # 创建时序数据
        self.logger.info("创建时序数据...")
        X_sequence, y, stock_codes, dates = self.create_sequences(result_df, self.feature_columns)
        
        if len(X_sequence) == 0:
            self.logger.error("没有有效的时序数据")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 重塑数据以拟合缩放器
        X_reshaped = X_sequence.reshape(-1, X_sequence.shape[-1])
        
        # 数据清洗：处理缺失值和异常值
        self.logger.info("开始数据清洗...")
        X_cleaned = self._clean_data(X_reshaped)
        
        # 拟合特征缩放器
        self.logger.info("拟合特征缩放器...")
        self.feature_scaler.fit(X_cleaned)
        
        # 处理目标变量
        self.logger.info("处理目标变量...")
        y_cleaned = y.copy()
        y_finite = np.isfinite(y_cleaned)
        
        if not np.all(y_finite):
            finite_y = y_cleaned[y_finite]
            missing_count = np.sum(~y_finite)
            self.logger.info(f"目标变量缺失值: {missing_count}/{len(y)} ({missing_count/len(y)*100:.1f}%)")
            
            if self.imputation_strategy == 'mean':
                self.target_imputation_value = np.mean(finite_y)
            elif self.imputation_strategy == 'mode':
                self.target_imputation_value = pd.Series(finite_y).mode()[0]
            elif self.imputation_strategy == 'hybrid':
                y_temp = pd.Series(y_cleaned).ffill().values
                mask = ~np.isfinite(y_temp)
                if np.any(mask):
                    y_temp[mask] = np.median(finite_y)
                y_cleaned = y_temp
                self.target_imputation_value = np.median(finite_y)
            else:  # median
                self.target_imputation_value = np.median(finite_y)
            y_cleaned[~y_finite] = self.target_imputation_value
            self.logger.info(f"目标变量填充值: {self.target_imputation_value:.4f}")
        else:
            if self.imputation_strategy == 'mean':
                self.target_imputation_value = np.mean(y_cleaned)
            elif self.imputation_strategy == 'mode':
                self.target_imputation_value = pd.Series(y_cleaned).mode()[0]
            else:  # median or hybrid
                self.target_imputation_value = np.median(y_cleaned)
            self.logger.info(f"目标变量无需填充，均值: {self.target_imputation_value:.4f}")
        
        # 拟合目标变量缩放器
        self.logger.info("拟合目标变量缩放器...")
        y_reshaped = y_cleaned.reshape(-1, 1)
        self.target_scaler.fit(y_reshaped)
        
        # 设置is_fitted标志
        self.is_fitted = True
        
        # 现在可以安全地进行 Collateral
        # 执行SHAP特征选择（如果需要）
        if model is not None and len(X_sequence) > self.shap_subset_size:
            self.logger.info(f"对{self.shap_subset_size}个样本执行SHAP特征选择...")
            subset_indices = np.random.choice(len(X_sequence), size=self.shap_subset_size, replace=False)
            X_subset = X_sequence[subset_indices]
            y_subset = y[subset_indices]
            
            # 现在可以安全调用transform_features
            X_subset_scaled = self.transform_features(X_subset)
            shap_values = model.compute_shap_values(X_subset_scaled, y_subset)
            feature_importance = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': feature_importance
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            selected_features = importance_df['feature'].iloc[:int(len(self.feature_columns) * 0.5)].tolist()
            self.feature_columns = [f for f in self.feature_columns if f in selected_features]
            self.logger.info(f"SHAP特征选择完成: 保留{len(self.feature_columns)}个特征")
        
        # 转换数据
        X_sequence_scaled = self.transform_features(X_sequence)
        y_scaled = self.transform_target(y)
        
        self.logger.info("数据预处理和模型拟合完成")
        return X_sequence_scaled, y_scaled, stock_codes, dates
        
        def process_stock(ts_code):
            X_stock, y_stock = self.prepare_single_stock(df, ts_code, self.feature_columns)
            if len(X_stock) > 0:
                return X_stock, y_stock, [ts_code] * len(X_stock), df[df['ts_code'] == ts_code]['trade_date'].iloc[self.lookback_window:].values
            return np.array([]), np.array([]), [], []
        
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(process_stock)(ts_code) for ts_code in tqdm(df['ts_code'].unique(), desc="Processing stocks")
        )
        
        for X_stock, y_stock, codes, stock_dates in results:
            if len(X_stock) > 0:
                X_sequence.append(X_stock)
                y.append(y_stock)
                stock_codes.extend(codes)
                dates.extend(stock_dates)
        
        if not X_sequence:
            raise ValueError("没有有效的时序数据，请检查输入数据或参数设置")
        
        X_sequence = np.concatenate(X_sequence, axis=0)
        y = np.concatenate(y, axis=0)
        stock_codes = np.array(stock_codes)
        dates = np.array(dates)
        
        self.logger.info(f"时序数据准备完成: {len(X_sequence)} 个样本，{len(np.unique(stock_codes))} 支股票")
        
        # 计算特征缺失值填充值和异常值阈值
        self.logger.info("计算特征填充值和异常值阈值...")
        X_reshaped = X_sequence.reshape(-1, X_sequence.shape[-1])
        n_samples, n_features = X_reshaped.shape
        self.imputation_values = {}
        self.outlier_lower_bounds = {}
        self.outlier_upper_bounds = {}
        outlier_counts = []
        
        for col in range(n_features):
            finite_data = X_reshaped[np.isfinite(X_reshaped[:, col]), col]
            if len(finite_data) == 0:
                self.imputation_values[col] = 0.0
                self.outlier_lower_bounds[col] = -np.inf
                self.outlier_upper_bounds[col] = np.inf
                outlier_counts.append(0)
                continue
            
            if self.imputation_strategy == 'mean':
                self.imputation_values[col] = np.mean(finite_data)
            elif self.imputation_strategy == 'mode':
                self.imputation_values[col] = pd.Series(finite_data).mode()[0]
            else:  # median or hybrid
                self.imputation_values[col] = np.median(finite_data)
            
            if self.outlier_detection == 'iqr':
                q25, q75 = np.percentile(finite_data, [25, 75])
                iqr = q75 - q25
                self.outlier_lower_bounds[col] = q25 - self.outlier_threshold * iqr
                self.outlier_upper_bounds[col] = q75 + self.outlier_threshold * iqr
                outliers = (finite_data < self.outlier_lower_bounds[col]) | (finite_data > self.outlier_upper_bounds[col])
                outlier_counts.append(np.sum(outliers))
            elif self.outlier_detection == 'zscore':
                mean = np.mean(finite_data)
                std = np.std(finite_data)
                self.outlier_lower_bounds[col] = mean - self.outlier_threshold * std
                self.outlier_upper_bounds[col] = mean + self.outlier_threshold * std
                outliers = (finite_data < self.outlier_lower_bounds[col]) | (finite_data > self.outlier_upper_bounds[col])
                outlier_counts.append(np.sum(outliers))
            else:
                self.outlier_lower_bounds[col] = -np.inf
                self.outlier_upper_bounds[col] = np.inf
                outlier_counts.append(0)
        
        total_outliers = np.sum(outlier_counts)
        self.logger.info(f"异常值统计: 总异常值 {total_outliers}, 平均每特征 {total_outliers/n_features:.1f}")
        
        # 处理训练数据并拟合缩放器
        self.logger.info("开始数据清洗...")
        X_cleaned = self._clean_data(X_reshaped)
        
        # 处理目标变量缺失值
        self.logger.info("处理目标变量...")
        y_cleaned = y.copy()
        y_finite = np.isfinite(y_cleaned)
        
        if not np.all(y_finite):
            finite_y = y_cleaned[y_finite]
            missing_count = np.sum(~y_finite)
            self.logger.info(f"目标变量缺失值: {missing_count}/{len(y)} ({missing_count/len(y)*100:.1f}%)")
            
            if self.imputation_strategy == 'mean':
                self.target_imputation_value = np.mean(finite_y)
            elif self.imputation_strategy == 'mode':
                self.target_imputation_value = pd.Series(finite_y).mode()[0]
            elif self.imputation_strategy == 'hybrid':
                y_temp = pd.Series(y_cleaned).ffill().values
                mask = ~np.isfinite(y_temp)
                if np.any(mask):
                    y_temp[mask] = np.median(finite_y)
                y_cleaned = y_temp
                self.target_imputation_value = np.median(finite_y)
            else:  # median
                self.target_imputation_value = np.median(finite_y)
            y_cleaned[~y_finite] = self.target_imputation_value
            self.logger.info(f"目标变量填充值: {self.target_imputation_value:.4f}")
        else:
            if self.imputation_strategy == 'mean':
                self.target_imputation_value = np.mean(y_cleaned)
            elif self.imputation_strategy == 'mode':
                self.target_imputation_value = pd.Series(y_cleaned).mode()[0]
            else:  # median or hybrid
                self.target_imputation_value = np.median(y_cleaned)
            self.logger.info(f"目标变量无需填充，均值: {self.target_imputation_value:.4f}")
        
        # 计算目标变量异常值阈值
        self.logger.info("计算目标变量异常值阈值...")
        if self.outlier_detection == 'iqr':
            q25, q75 = np.percentile(y_cleaned, [25, 75])
            iqr_y = q75 - q25
            self.target_outlier_lower = q25 - self.outlier_threshold * iqr_y
            self.target_outlier_upper = q75 + self.outlier_threshold * iqr_y
        elif self.outlier_detection == 'zscore':
            mean_y = np.mean(y_cleaned)
            std_y = np.std(y_cleaned)
            self.target_outlier_lower = mean_y - self.outlier_threshold * std_y
            self.target_outlier_upper = mean_y + self.outlier_threshold * std_y
        else:
            self.target_outlier_lower = -np.inf
            self.target_outlier_upper = np.inf
        
        self.logger.info(f"目标变量异常值范围: [{self.target_outlier_lower:.4f}, {self.target_outlier_upper:.4f}]")
        
        # 处理目标变量异常值
        original_count = len(y_cleaned)
        y_cleaned = np.clip(y_cleaned, self.target_outlier_lower, self.target_outlier_upper)
        clipped_count = np.sum((y < self.target_outlier_lower) | (y > self.target_outlier_upper))
        self.logger.info(f"目标变量异常值处理: 裁剪 {clipped_count}/{original_count} 个异常值")
        
        # 拟合特征缩放器
        self.logger.info("拟合特征缩放器...")
        self.feature_scaler.fit(X_cleaned)
        
        # 拟合目标变量缩放器
        self.logger.info("拟合目标变量缩放器...")
        y_reshaped = y_cleaned.reshape(-1, 1)
        self.target_scaler.fit(y_reshaped)
        
        # 转换数据
        X_sequence_scaled = self.transform_features(X_sequence)
        y_scaled = self.transform_target(y)
        
        self.is_fitted = True
        self.logger.info("数据预处理和模型拟合完成")
        
        return X_sequence_scaled, y_scaled, stock_codes, dates
    
    def fit_scalers(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        [已弃用] 仅拟合特征和目标变量的缩放器
        """
        warnings.warn(
            "fit_scalers() is deprecated. Use fit() instead for complete preprocessing.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.logger.info("开始拟合缩放器...")
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_finite = np.isfinite(X_reshaped)
        y_finite = np.isfinite(y)
        
        if not np.all(X_finite) or not np.all(y_finite):
            raise ValueError("数据中存在NaN或无穷大值，请先清洗数据")
        
        self.feature_scaler.fit(X_reshaped)
        self.target_scaler.fit(y.reshape(-1, 1))
        
        self.is_fitted = True
        self.logger.info("缩放器拟合完成")
    
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
            'data_info': self.data_info,
            'imputation_values': self.imputation_values,
            'outlier_lower_bounds': self.outlier_lower_bounds,
            'outlier_upper_bounds': self.outlier_upper_bounds,
            'target_imputation_value': self.target_imputation_value,
            'target_outlier_lower': self.target_outlier_lower,
            'target_outlier_upper': self.target_outlier_upper
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
        self.imputation_values = save_data.get('imputation_values', None)
        self.outlier_lower_bounds = save_data.get('outlier_lower_bounds', None)
        self.outlier_upper_bounds = save_data.get('outlier_upper_bounds', None)
        self.target_imputation_value = save_data.get('target_imputation_value', None)
        self.target_outlier_lower = save_data.get('target_outlier_lower', None)
        self.target_outlier_upper = save_data.get('target_outlier_upper', None)
        self.is_fitted = True
        self.logger.info(f"Preprocessor loaded from {path}")