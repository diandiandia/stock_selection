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
                 validation_split: float = 0.1,
                 imputation_strategy: str = 'median',
                 outlier_detection: str = 'iqr',
                 outlier_threshold: float = 3.0):
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
        self.imputation_strategy = imputation_strategy
        self.outlier_detection = outlier_detection
        self.outlier_threshold = outlier_threshold
        
        # 参数验证
        if self.imputation_strategy not in ['mean', 'median', 'mode']:
            raise ValueError(f"无效的缺失值填充策略: {self.imputation_strategy}, 必须是'mean', 'median'或'mode'")
        if self.outlier_detection not in ['iqr', 'zscore', 'none']:
            raise ValueError(f"无效的异常值检测方法: {self.outlier_detection}, 必须是'iqr', 'zscore'或'none'")
        if self.outlier_threshold <= 0:
            raise ValueError(f"异常值阈值必须为正数, 实际值: {self.outlier_threshold}")
        if not (0 < self.test_size < 1):
            raise ValueError(f"测试集比例必须在(0, 1)之间, 实际值: {self.test_size}")
        if not (0 < self.validation_split < 1):
            raise ValueError(f"验证集比例必须在(0, 1)之间, 实际值: {self.validation_split}")
        if self.lookback_window <= 0:
            raise ValueError(f"回溯窗口大小必须为正数, 实际值: {self.lookback_window}")
        
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
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            添加目标变量的DataFrame
        """
        self.logger.info(f"开始创建目标变量，预测周期: {self.prediction_horizon}")
        result_df = df.copy()
        
        # 计算T+1收益率
        result_df[self.target_column] = result_df.groupby('ts_code')['close'].shift(-self.prediction_horizon) / result_df['close'] - 1
        
        # 记录目标变量的统计信息
        target_data = result_df[self.target_column].dropna()
        self.logger.info(f"目标变量创建完成，有效样本数: {len(target_data)}, 均值: {target_data.mean():.4f}, 标准差: {target_data.std():.4f}")
        
        return result_df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        选择用于训练的特征，结合缺失值分析和多重共线性检测
        
        Args:
            df: 包含所有特征的DataFrame
            
        Returns:
            选定的特征列名列表
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
        valid_features = nan_ratios[nan_ratios < 0.2].index.tolist()  # 放宽缺失值阈值至0.2
        self.logger.info(f"缺失值过滤后特征数量: {len(valid_features)} (过滤了 {len(feature_cols) - len(valid_features)} 个)")
        
        # 步骤2: 移除多重共线性特征（跳过特征重要性选择，因为没有训练数据）
        final_features = self._remove_multicollinearity(df[valid_features])
        self.logger.info(f"多重共线性检测后最终特征数量: {len(final_features)}")
        
        self.feature_columns = final_features
        self.logger.info(f"特征选择完成: 原始{len(feature_cols)} → 缺失值过滤{len(valid_features)} → 去多重共线性{len(final_features)}")
        
        return final_features
    
    def _remove_multicollinearity(self, feature_df: pd.DataFrame) -> List[str]:
        """使用VIF移除多重共线性特征 - 优化相关性阈值和VIF阈值"""
        import warnings
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        self.logger.info(f"开始多重共线性检测，输入特征数量: {len(feature_df.columns)}")
        features = feature_df.copy()
        selected_features = []
        
        # 预过滤高相关特征（优化1）
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]  # 保持相关性阈值为0.9
        if to_drop:
            self.logger.info(f"预过滤高相关特征: {len(to_drop)} 个 (相关系数>0.9)")
        features = features.drop(columns=to_drop)
        
        # 计算VIF并迭代移除高VIF特征（并行计算优化）
        iteration = 0
        while len(features.columns) > 0 and len(selected_features) < 30:
            iteration += 1
            vif_data = pd.DataFrame()
            vif_data["feature"] = features.columns
            
            # 使用并行计算VIF（优化2）
            from concurrent.futures import ThreadPoolExecutor
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
            
            # 记录VIF统计信息
            self.logger.debug(f"第{iteration}轮VIF计算: 最小VIF={vif_data['VIF'].iloc[0]:.2f}, 最大VIF={vif_data['VIF'].iloc[-1]:.2f}")
            
            # 如果最小VIF大于15，没有可保留的特征
            if vif_data["VIF"].iloc[0] > 15:  # 保持VIF阈值为15
                if not selected_features:
                    selected_features.append(vif_data["feature"].iloc[0])
                    self.logger.info(f"所有VIF>15，保留最小VIF特征: {vif_data['feature'].iloc[0]}")
                break
            
            # 添加最小VIF特征
            selected_feature = vif_data["feature"].iloc[0]
            selected_features.append(selected_feature)
            
            # 移除已选择特征并继续
            features = features.drop(columns=[selected_feature])
        
        self.logger.info(f"多重共线性检测完成: 最终选择 {len(selected_features)} 个特征")
        
        # 按原始顺序返回，最多保留30个特征
        selected_features_limited = selected_features[:30]  # 增加特征数量限制至30
        self.logger.info(f"特征数量限制: 从 {len(selected_features)} 个减少到 {len(selected_features_limited)} 个")
        return [col for col in feature_df.columns if col in selected_features_limited]
    
    def prepare_single_stock(self, df: pd.DataFrame, ts_code: str, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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
            self.logger.debug(f"股票 {ts_code} 数据不足: {len(stock_df)} < {self.lookback_window + 1}")
            return np.array([]), np.array([])
        
        # 使用全局选择的特征
        # 过滤掉股票数据中不存在的特征
        available_features = [f for f in features if f in stock_df.columns]
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            self.logger.warning(f"股票 {ts_code} 缺少 {len(missing)} 个特征: {missing}")
            features = available_features
        
        # 创建时序数据
        X_sequence = []
        y = []
        
        valid_samples = 0
        for i in range(self.lookback_window, len(stock_df)):
            # 检查是否有足够的历史数据
            sequence = stock_df[features].iloc[i-self.lookback_window:i].values
            target = stock_df[self.target_column].iloc[i]
            
            # 跳过包含NaN的数据
            if not np.isnan(sequence).any() and not np.isnan(target):
                X_sequence.append(sequence)
                y.append(target)
                valid_samples += 1
        
        self.logger.debug(f"股票 {ts_code}: 原始数据 {len(stock_df)} 条，有效样本 {valid_samples} 条")
        
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
        self.logger.info("开始为所有股票准备训练数据...")
        
        # 1. 创建特征
        self.logger.info("步骤1: 创建特征...")
        df = self.create_features(df)
        
        # 2. 创建目标变量
        self.logger.info("步骤2: 创建目标变量...")
        df = self.create_target(df)
        
        # 记录数据概览
        total_stocks = df['ts_code'].nunique()
        total_records = len(df)
        self.logger.info(f"数据概览: {total_stocks} 只股票, {total_records} 条记录")
        
        # 3. 全局特征选择（所有股票使用相同特征集）
        self.logger.info("步骤3: 全局特征选择...")
        global_features = self.select_features(df)
        self.logger.info(f"选择的特征: {global_features}")
        
        # 4. 收集所有股票的时序数据
        self.logger.info("步骤4: 收集时序数据...")
        X_all = []
        y_all = []
        stock_codes = []
        dates = []
        
        processed_stocks = 0
        valid_stocks = 0
        
        for ts_code in df['ts_code'].unique():
            processed_stocks += 1
            if processed_stocks % 50 == 0:
                self.logger.info(f"处理进度: {processed_stocks}/{total_stocks} 只股票")
            
            X_seq, y_seq = self.prepare_single_stock(df, ts_code, global_features)
            
            if len(X_seq) > 0:
                valid_stocks += 1
                X_all.append(X_seq)
                y_all.extend(y_seq)
                
                # 记录股票代码和对应日期
                stock_dates = df[df['ts_code'] == ts_code]['trade_date'].iloc[self.lookback_window:].tolist()
                stock_codes.extend([ts_code] * len(y_seq))
                dates.extend(stock_dates)
        
        if not X_all:
            self.logger.error("没有找到有效的训练数据")
            raise ValueError("No valid data found for training")
        
        # 合并所有数据
        X_sequence = np.concatenate(X_all, axis=0)
        y = np.array(y_all)
        
        self.logger.info(f"数据准备完成:")
        self.logger.info(f"- 处理股票数: {valid_stocks}/{total_stocks}")
        self.logger.info(f"- 总样本数: {len(y)}")
        self.logger.info(f"- 特征维度: {X_sequence.shape[2]}")
        self.logger.info(f"- 时间步长: {X_sequence.shape[1]}")
        
        return X_sequence, y, stock_codes, dates
    
    def prepare_and_fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        一站式数据准备和模型拟合方法
        
        整合prepare_all_stocks、fit、transform_features和transform_target的完整流程
        简化main.py中的调用逻辑
        
        Args:
            df: 包含技术指标的DataFrame
            
        Returns:
            X_sequence_scaled: 标准化后的时序特征 (samples, timesteps, features)
            y_scaled: 标准化后的目标变量 (samples,)
            stock_codes: 股票代码列表
            dates: 日期列表
        """
        self.logger.info("开始一站式数据准备和模型拟合...")
        
        # 1. 准备所有股票的训练数据
        X_sequence, y, stock_codes, dates = self.prepare_all_stocks(df)
        
        if len(X_sequence) == 0:
            raise ValueError("没有有效的训练数据，请检查数据质量和参数设置")
        
        # 2. 拟合预处理模型（包含缩放器拟合）
        self.fit(X_sequence, y)
        
        # 3. 转换数据
        X_sequence_scaled = self.transform_features(X_sequence)
        y_scaled = self.transform_target(y)
        
        self.logger.info(f"一站式处理完成: 获得{len(X_sequence_scaled)}个样本，{len(stock_codes)}支股票")
        self.logger.info(f"最终数据维度: X={X_sequence_scaled.shape}, y={y_scaled.shape}")
        
        return X_sequence_scaled, y_scaled, stock_codes, dates
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合预处理模型，计算并存储所有必要的预处理参数
        
        参数:
            X: 训练特征数据，形状为 (samples, timesteps, features)
            y: 训练目标变量，形状为 (samples,)
        
        注意:
            该方法仅应使用训练数据调用，以避免数据泄露
            所有预处理参数（缺失值填充值、异常值阈值等）将从训练数据计算
        """
        self.logger.info("开始模型拟合，计算预处理参数...")
        self.logger.info(f"训练数据形状: X={X.shape}, y={y.shape}")
        
        # 重塑数据维度以便处理
        X_reshaped = X.reshape(-1, X.shape[-1])
        n_features = X_reshaped.shape[1]
        
        self.logger.info(f"重塑后数据形状: {X_reshaped.shape}")
        
        # 1. 计算缺失值填充值并存储
        self.logger.info("计算缺失值填充值...")
        self.imputation_values = np.zeros(n_features)
        X_finite = np.isfinite(X_reshaped)
        
        nan_counts = []
        for col in range(n_features):
            col_data = X_reshaped[:, col]
            finite_mask = X_finite[:, col]
            nan_count = np.sum(~finite_mask)
            nan_counts.append(nan_count)
            
            if np.any(finite_mask):
                if self.imputation_strategy == 'mean':
                    self.imputation_values[col] = np.mean(col_data[finite_mask])
                elif self.imputation_strategy == 'median':
                    self.imputation_values[col] = np.median(col_data[finite_mask])
                elif self.imputation_strategy == 'mode':
                    self.imputation_values[col] = pd.Series(col_data[finite_mask]).mode()[0]
                else:
                    self.imputation_values[col] = np.median(col_data[finite_mask])
            else:
                self.imputation_values[col] = 0  # 所有值都是NaN时使用0
        
        total_nans = np.sum(nan_counts)
        self.logger.info(f"缺失值统计: 总缺失值 {total_nans}, 平均每特征 {total_nans/n_features:.1f}")
        
        # 2. 计算异常值阈值并存储
        self.logger.info("计算异常值阈值...")
        self.outlier_lower_bounds = np.zeros(n_features)
        self.outlier_upper_bounds = np.zeros(n_features)
        
        outlier_counts = []
        for col in range(n_features):
            col_data = X_reshaped[:, col]
            finite_mask = np.isfinite(col_data)
            if np.sum(finite_mask) > 10:
                if self.outlier_detection == 'iqr':
                    Q1 = np.percentile(col_data[finite_mask], 25)
                    Q3 = np.percentile(col_data[finite_mask], 75)
                    IQR = Q3 - Q1
                    self.outlier_lower_bounds[col] = Q1 - self.outlier_threshold * IQR
                    self.outlier_upper_bounds[col] = Q3 + self.outlier_threshold * IQR
                elif self.outlier_detection == 'zscore':
                    mean = np.mean(col_data[finite_mask])
                    std = np.std(col_data[finite_mask])
                    self.outlier_lower_bounds[col] = mean - self.outlier_threshold * std
                    self.outlier_upper_bounds[col] = mean + self.outlier_threshold * std
                else:
                    self.outlier_lower_bounds[col] = -np.inf
                    self.outlier_upper_bounds[col] = np.inf
                    
                # 计算异常值数量
                outliers = (col_data[finite_mask] < self.outlier_lower_bounds[col]) | (col_data[finite_mask] > self.outlier_upper_bounds[col])
                outlier_counts.append(np.sum(outliers))
            else:
                self.outlier_lower_bounds[col] = -np.inf
                self.outlier_upper_bounds[col] = np.inf
                outlier_counts.append(0)
        
        total_outliers = np.sum(outlier_counts)
        self.logger.info(f"异常值统计: 总异常值 {total_outliers}, 平均每特征 {total_outliers/n_features:.1f}")
        
        # 3. 处理训练数据并拟合缩放器
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
            else:  # median
                self.target_imputation_value = np.median(finite_y)
            y_cleaned[~y_finite] = self.target_imputation_value
            self.logger.info(f"目标变量填充值: {self.target_imputation_value:.4f}")
        else:
            # 所有值都是有限的，直接计算填充值
            if self.imputation_strategy == 'mean':
                self.target_imputation_value = np.mean(y_cleaned)
            elif self.imputation_strategy == 'mode':
                self.target_imputation_value = pd.Series(y_cleaned).mode()[0]
            else:  # median
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
        
        self.is_fitted = True
        self.logger.info("模型拟合完成，所有预处理参数已计算")
    
    def _clean_data(self, X_reshaped: np.ndarray) -> np.ndarray:
        """
        使用拟合阶段计算的参数清洗数据
        
        参数:
            X_reshaped: 展平后的特征数据，形状为 (samples*timesteps, features)
        
        返回:
            清洗后的特征数据
        """
        X_cleaned = X_reshaped.copy()
        n_samples, n_features = X_cleaned.shape
        
        # 1. 使用存储的填充值处理缺失值
        for col in range(n_features):
            mask = ~np.isfinite(X_cleaned[:, col])
            if np.any(mask):
                X_cleaned[mask, col] = self.imputation_values[col]
        
        # 2. 使用存储的阈值处理异常值
        for col in range(n_features):
            lower = self.outlier_lower_bounds[col]
            upper = self.outlier_upper_bounds[col]
            if not np.isinf(lower) and not np.isinf(upper):
                mask = (X_cleaned[:, col] < lower) | (X_cleaned[:, col] > upper)
                if np.any(mask):
                    X_cleaned[mask, col] = np.clip(X_cleaned[mask, col], lower, upper)
        
        return X_cleaned
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        转换特征
        
        Args:
            X: 原始特征 (samples, timesteps, features)
            
        Returns:
            标准化后的特征
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")
        
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_cleaned = self._clean_data(X_reshaped)
        X_scaled = self.feature_scaler.transform(X_cleaned)
        return X_scaled.reshape(X.shape)
    
    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        转换目标变量
        
        Args:
            y: 原始目标变量
            
        Returns:
            标准化后的目标变量
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")
        
        y_reshaped = y.reshape(-1, 1)
        # 对目标变量应用相同的数据清洗逻辑
        y_cleaned = self._clean_target(y_reshaped)
        y_scaled = self.target_scaler.transform(y_cleaned)
        return y_scaled.flatten()
    
    def _clean_target(self, y_reshaped: np.ndarray) -> np.ndarray:
        """
        清洗目标变量数据
        
        参数:
            y_reshaped: 展平后的目标变量，形状为 (samples, 1)
        
        返回:
            清洗后的目标变量
        """
        y_cleaned = y_reshaped.copy()
        
        # 处理缺失值
        mask = ~np.isfinite(y_cleaned)
        if np.any(mask):
            # 使用存储的目标变量填充值
            y_cleaned[mask] = self.imputation_values[-1]  # 假设最后一个是目标变量的填充值
        
        # 处理异常值
        lower = self.outlier_lower_bounds[-1] if self.outlier_lower_bounds is not None else -np.inf
        upper = self.outlier_upper_bounds[-1] if self.outlier_upper_bounds is not None else np.inf
        if not np.isinf(lower) and not np.isinf(upper):
            mask = (y_cleaned < lower) | (y_cleaned > upper)
            if np.any(mask):
                y_cleaned[mask] = np.clip(y_cleaned[mask], lower, upper)
        
        return y_cleaned
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        目标变量逆转换
        
        Args:
            y_scaled: 标准化后的目标变量
            
        Returns:
            原始尺度的目标变量
        """
        y_reshaped = y_scaled.reshape(-1, 1)
        y_inv = self.target_scaler.inverse_transform(y_reshaped)
        return y_inv.flatten()
    
    def create_train_test_split(self, X: np.ndarray, y: np.ndarray, stock_codes: Optional[List[str]] = None, 
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
            # 获取唯一股票代码
            unique_stocks = np.unique(stock_codes)
            np.random.seed(42)
            np.random.shuffle(unique_stocks)
            split_idx = int(len(unique_stocks) * (1 - self.test_size))
            train_stocks = set(unique_stocks[:split_idx])
            
            # 创建掩码
            train_mask = np.array([code in train_stocks for code in stock_codes])
            test_mask = ~train_mask
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        
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

    def fit_scalers(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        [已弃用] 仅拟合特征和目标变量的缩放器
        
        注意: 此方法已弃用，建议使用fit()方法，它提供更完整的数据预处理功能
        
        Args:
            X: 时序特征 (samples, timesteps, features)
            y: 目标变量 (samples,)
        """
        import warnings
        warnings.warn(
            "fit_scalers() is deprecated. Use fit() instead for complete preprocessing including outlier detection and imputation.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.logger.info("开始拟合缩放器...")
        
        # 重塑X以拟合缩放器
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # 确保数据有效性
        X_finite = np.isfinite(X_reshaped)
        y_finite = np.isfinite(y)
        
        if not np.all(X_finite) or not np.all(y_finite):
            raise ValueError("数据中存在NaN或无穷大值，请先清洗数据")
        
        # 直接拟合缩放器（假设数据已清洗）
        self.feature_scaler.fit(X_reshaped)
        self.target_scaler.fit(y.reshape(-1, 1))
        
        self.is_fitted = True
        self.logger.info("缩放器拟合完成")