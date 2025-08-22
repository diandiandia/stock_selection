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
                 shap_subset_size: int = 10000,
                 max_workers: int = 4):
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
            max_workers: 并行处理的最大工作线程数
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
        self.max_workers = max_workers
        
        # 初始化缩放器
        self.feature_scaler = StandardScaler() if feature_scaler_type == 'standard' else MinMaxScaler()
        self.target_scaler = StandardScaler() if target_scaler_type == 'standard' else MinMaxScaler()
        
        # 初始化其他属性
        self.feature_columns = []
        self.is_fitted = False
        self.data_info = {}
        self.imputation_values = None
        self.outlier_lower_bounds = None
        self.outlier_upper_bounds = None
        self.target_imputation_value = None
        self.target_outlier_lower = None
        self.target_outlier_upper = None
    
    def create_sequences(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        创建时序序列数据
        """
        self.logger.info("创建时序数据...")
        self.logger.info(f"创建时序数据，特征数量: {len(self.feature_columns)}")
        
        sequences = []
        targets = []
        stock_codes = []
        dates = []
        skipped_insufficient = 0
        skipped_nan = 0
        
        def process_stock(group):
            group = group.sort_values('trade_date')
            data = group[self.feature_columns].values
            target = group[target_col].values
            stock_code = group['ts_code'].iloc[0]
            group_dates = group['trade_date'].values
            
            min_data_points = self.lookback_window + self.prediction_horizon
            if len(group) < min_data_points:
                return None, None, None, None
            
            group_sequences = []
            group_targets = []
            group_codes = []
            group_dates_out = []
            
            for i in range(len(group) - self.lookback_window - self.prediction_horizon + 1):
                seq = data[i:i + self.lookback_window]
                tgt = target[i + self.lookback_window + self.prediction_horizon - 1]
                
                # 检查序列和目标中的NaN
                if np.any(np.isnan(seq)) or np.isnan(tgt):
                    return None, None, None, None
                
                group_sequences.append(seq)
                group_targets.append(tgt)
                group_codes.append(stock_code)
                group_dates_out.append(group_dates[i + self.lookback_window - 1])
            
            return np.array(group_sequences), np.array(group_targets), group_codes, np.array(group_dates_out)
        
        self.logger.info(f"Processing stocks: ")
        groups = [group for _, group in df.groupby('ts_code')]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(executor.map(process_stock, groups), total=len(groups)))
        
        for seq, tgt, codes, group_dates in results:
            if seq is None or len(seq) == 0:
                skipped_insufficient += 1
                continue
            if np.any(np.isnan(seq)) or np.any(np.isnan(tgt)):
                skipped_nan += 1
                continue
            sequences.append(seq)
            targets.append(tgt)
            stock_codes.extend(codes)
            dates.append(group_dates)
        
        if sequences:
            X_sequence = np.concatenate(sequences, axis=0)
            y = np.concatenate(targets, axis=0)
            dates = np.concatenate(dates, axis=0)
        else:
            X_sequence = np.array([])
            y = np.array([])
            dates = np.array([])
        
        self.logger.info(f"序列诊断: 总股票 {len(groups)}, 跳过数据不足 {skipped_insufficient}, 跳过NaN序列 {skipped_nan}, 有效序列 {len(X_sequence)}")
        self.logger.info(f"生成序列完成: {len(X_sequence)}个样本，维度: {X_sequence.shape if X_sequence.size else 'empty'}")
        
        return X_sequence, y, stock_codes, dates
    
    def fit(self, df: pd.DataFrame, model: Optional[LSTMLGBMPredictor] = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        拟合数据预处理器并返回处理后的数据
        """
        self.logger.info("开始数据预处理和拟合...")
        self.logger.info(f"输入数据形状: {df.shape}")
        
        # 数据排序
        self.logger.info("数据排序完成...")
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        self.logger.info(f"数据排序完成，股票数量: {df['ts_code'].nunique()}")
        
        # 创建技术信号
        self.logger.info("创建技术信号...")
        df['signal'] = df['close'].pct_change(periods=self.prediction_horizon).shift(-self.prediction_horizon)
        self.logger.info(f"技术信号创建完成，新增特征数量: 1")
        
        # 创建目标变量
        self.logger.info(f"开始创建目标变量，预测周期: {self.prediction_horizon}")
        target_col = 'signal'
        self.data_info['target_mean'] = df[target_col].mean()
        self.data_info['target_std'] = df[target_col].std()
        self.logger.info(f"目标变量创建完成，有效样本数: {df[target_col].notna().sum()}, 均值: {self.data_info['target_mean']:.4f}, 标准差: {self.data_info['target_std']:.4f}")
        
        # 特征选择
        self.logger.info("开始特征选择...")
        all_features = [col for col in df.columns if col not in ['trade_date', 'ts_code', target_col]]
        self.logger.info(f"原始特征数量: {len(all_features)}")
        
        # 缺失值过滤
        missing_ratio = df[all_features].isna().mean()
        selected_features = [col for col in all_features if missing_ratio[col] < 0.5]
        self.logger.info(f"缺失值过滤后特征数量: {len(selected_features)}")
        
        # 多重共线性检测
        self.logger.info(f"开始多重共线性检测，输入特征数量: {len(selected_features)}, VIF阈值: 7.0")
        correlation_matrix = df[selected_features].corr().abs()
        high_corr_pairs = []
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                if correlation_matrix.iloc[i, j] > 0.8:
                    high_corr_pairs.append((selected_features[i], selected_features[j]))
        self.logger.info(f"预过滤高相关特征: {len(high_corr_pairs)} 个 (相关系数>0.8)")
        
        # VIF计算
        X = df[selected_features].fillna(df[selected_features].mean())
        vif_data = pd.DataFrame()
        vif_data['feature'] = selected_features
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        selected_features = vif_data[vif_data['VIF'] < 7.0]['feature'].tolist()
        self.logger.info(f"多重共线性检测完成: 最终选择 {len(selected_features)} 个特征")
        self.feature_columns = selected_features
        self.logger.info(f"多重共线性检测后特征数量: {len(self.feature_columns)}")
        
        self.logger.info(f"特征选择完成: 原始{len(all_features)} → 缺失值过滤{len(all_features)} → 去多重共线性{len(self.feature_columns)}")
        
        # 创建时序数据
        X_sequence, y, stock_codes, dates = self.create_sequences(df, target_col)
        
        # 计算特征异常值边界
        self.logger.info("计算特征异常值边界...")
        self.imputation_values = {}
        self.outlier_lower_bounds = {}
        self.outlier_upper_bounds = {}
        for col in self.feature_columns:
            col_data = df[col].dropna()
            if self.imputation_strategy == 'mean':
                self.imputation_values[col] = col_data.mean()
            elif self.imputation_strategy == 'median':
                self.imputation_values[col] = col_data.median()
            elif self.imputation_strategy == 'mode':
                self.imputation_values[col] = col_data.mode()[0]
            elif self.imputation_strategy == 'hybrid':
                self.imputation_values[col] = col_data.median()
            
            if self.outlier_detection == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                self.outlier_lower_bounds[col] = mean - self.outlier_threshold * std
                self.outlier_upper_bounds[col] = mean + self.outlier_threshold * std
            elif self.outlier_detection == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_lower_bounds[col] = Q1 - 1.5 * IQR
                self.outlier_upper_bounds[col] = Q3 + 1.5 * IQR
            else:
                self.outlier_lower_bounds[col] = col_data.min()
                self.outlier_upper_bounds[col] = col_data.max()
        
        self.logger.info(f"特征异常值边界计算完成: {len(self.feature_columns)} 个特征")
        
        # 数据清洗
        self.logger.info("开始数据清洗...")
        self.logger.info("开始清洗数据...")
        for i in range(X_sequence.shape[2]):
            feature = self.feature_columns[i]
            feature_data = X_sequence[:, :, i]
            if self.outlier_detection != 'none':
                outliers = (feature_data < self.outlier_lower_bounds[feature]) | (feature_data > self.outlier_upper_bounds[feature])
                X_sequence[:, :, i][outliers] = np.clip(
                    feature_data[outliers],
                    self.outlier_lower_bounds[feature],
                    self.outlier_upper_bounds[feature]
                )
                self.logger.info(f"特征 {feature}: 裁剪 {outliers.sum()} 个异常值")
        
        self.logger.info("数据清洗完成")
        
        # 处理目标变量
        self.logger.info("处理目标变量...")
        if np.any(np.isnan(y)):
            if self.imputation_strategy == 'mean':
                self.target_imputation_value = np.nanmean(y)
            elif self.imputation_strategy == 'median':
                self.target_imputation_value = np.nanmedian(y)
            elif self.imputation_strategy == 'mode':
                self.target_imputation_value = pd.Series(y).mode()[0]
            elif self.imputation_strategy == 'hybrid':
                self.target_imputation_value = np.nanmedian(y)
            y[np.isnan(y)] = self.target_imputation_value
            self.logger.info(f"目标变量填充: {np.isnan(y).sum()} 个NaN值")
        else:
            self.logger.info(f"目标变量无需填充，均值: {np.mean(y):.4f}")
        
        # 计算目标变量异常值阈值
        self.logger.info("计算目标变量异常值阈值...")
        if self.outlier_detection == 'zscore':
            mean = np.mean(y)
            std = np.std(y)
            self.target_outlier_lower = mean - self.outlier_threshold * std
            self.target_outlier_upper = mean + self.outlier_threshold * std
        elif self.outlier_detection == 'iqr':
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            self.target_outlier_lower = Q1 - 1.5 * IQR
            self.target_outlier_upper = Q3 + 1.5 * IQR
        else:
            self.target_outlier_lower = np.min(y)
            self.target_outlier_upper = np.max(y)
        
        self.logger.info(f"目标变量异常值范围: [{self.target_outlier_lower:.4f}, {self.target_outlier_upper:.4f}]")
        
        outliers = (y < self.target_outlier_lower) | (y > self.target_outlier_upper)
        y[outliers] = np.clip(y[outliers], self.target_outlier_lower, self.target_outlier_upper)
        self.logger.info(f"目标变量异常值处理: 裁剪 {outliers.sum()}/{len(y)} 个异常值")
        
        # 拟合特征缩放器
        self.logger.info("拟合特征缩放器...")
        X_2d = X_sequence.reshape(-1, X_sequence.shape[2])
        self.feature_scaler.fit(X_2d)
        X_sequence_scaled = self.feature_scaler.transform(X_2d).reshape(X_sequence.shape)
        
        # 拟合目标变量缩放器
        self.logger.info("拟合目标变量缩放器...")
        self.target_scaler.fit(y.reshape(-1, 1))
        y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # SHAP特征选择
        if model is not None:
            self.logger.info(f"对{self.shap_subset_size}个样本执行SHAP特征选择...")
            subset_idx = np.random.choice(len(X_sequence_scaled), size=min(self.shap_subset_size, len(X_sequence_scaled)), replace=False)
            X_subset_scaled = X_sequence_scaled[subset_idx]
            y_subset_scaled = y_scaled[subset_idx]
            
            # 训练模型
            model.fit(X_sequence=X_subset_scaled, y=y_subset_scaled)
            
            # 计算SHAP值
            shap_values = model.compute_shap_values(X_subset_scaled)
            shap_importance = np.abs(shap_values).mean(axis=(0, 1))
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': shap_importance
            }).sort_values('importance', ascending=False)
            
            # 选择前12个特征
            selected_features = importance_df['feature'].iloc[:12].tolist()
            self.feature_columns = selected_features
            feature_indices = [self.feature_columns.index(f) for f in selected_features]
            X_sequence_scaled = X_sequence_scaled[:, :, feature_indices]
            
            self.logger.info(f"SHAP特征选择完成: 保留{len(self.feature_columns)}个特征")
        
        self.is_fitted = True
        return X_sequence_scaled, y_scaled, stock_codes, dates
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        转换数据
        """
        if not self.is_fitted:
            raise ValueError("预处理器未拟合，请先调用fit()")
        
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        df['signal'] = df['close'].pct_change(periods=self.prediction_horizon).shift(-self.prediction_horizon)
        target_col = 'signal'
        
        X_sequence, y, stock_codes, dates = self.create_sequences(df, target_col)
        
        for i in range(X_sequence.shape[2]):
            feature = self.feature_columns[i]
            feature_data = X_sequence[:, :, i]
            if self.outlier_detection != 'none':
                outliers = (feature_data < self.outlier_lower_bounds[feature]) | (feature_data > self.outlier_upper_bounds[feature])
                X_sequence[:, :, i][outliers] = np.clip(
                    feature_data[outliers],
                    self.outlier_lower_bounds[feature],
                    self.outlier_upper_bounds[feature]
                )
        
        if np.any(np.isnan(y)):
            y[np.isnan(y)] = self.target_imputation_value
        
        outliers = (y < self.target_outlier_lower) | (y > self.target_outlier_upper)
        y[outliers] = np.clip(y[outliers], self.target_outlier_lower, self.target_outlier_upper)
        
        X_2d = X_sequence.reshape(-1, X_sequence.shape[2])
        X_sequence_scaled = self.feature_scaler.transform(X_2d).reshape(X_sequence.shape)
        y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        return X_sequence_scaled, y_scaled, stock_codes, dates
    
    def save_preprocessor(self, path: str) -> None:
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
        save_data = joblib.load(path)
        self.feature_scaler = save_data['feature_scaler']
        self.target_scaler = save_data['target_scaler']
        self.feature_columns = save_data['feature_columns']
        self.lookback_window = save_data['lookback_window']
        self.prediction_horizon = save_data['prediction_horizon']
        self.data_info = save_data.get('data_info', {})
        self.imputation_values = save_data.get('imputation_values', None)
        self.outlier_lower_bounds = save_data.get('outlier_lower_bounds', None)
        self.outlier_upper_bounds = save_data.get('outlier_upper_bounds', None)
        self.target_imputation_value = save_data.get('target_imputation_value', None)
        self.target_outlier_lower = save_data.get('target_outlier_lower', None)
        self.target_outlier_upper = save_data.get('target_outlier_upper', None)
        self.is_fitted = True
        self.logger.info(f"Preprocessor loaded from {path}")