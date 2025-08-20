from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from src.utils.log_helper import LogHelper

class LSTMLGBMPredictor:
    """LSTM + LightGBM混合预测模型，专为T+1股票推荐设计"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化混合预测模型
        
        Args:
            config: 模型配置字典
        """
        self.logger = LogHelper.get_logger(__name__)
        self.config = config or self._get_default_config()
        
        # 初始化模型
        self.lgbm_model = None
        self.lstm_model = None
        self.ensemble_weights = {'lgbm': 0.5, 'lstm': 0.5}
        
        # 特征重要性缓存
        self.feature_importance_cache = {}
        
    def _get_default_config(self) -> Dict:
        """获取默认模型配置"""
        return {
            # 数据配置
            'lookback_window': 20,
            'test_size': 0.2,
            
            # LightGBM配置
            'lgbm_params': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 10,
                'num_leaves': 64,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1
            },
            
            # LSTM配置
            'lstm_units': [64, 32],  # 双层LSTM
            'dropout_rate': 0.3,
            'recurrent_dropout': 0.2,
            'batch_size': 128,
            'epochs': 100,
            'learning_rate': 0.001,
            'patience': 15,
            
            # 集成配置
            'ensemble_method': 'weighted',  # 'weighted' 或 'stacking'
            'weight_optimization': True,
            
            # 训练配置
            'validation_split': 0.15,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 8,
            'min_lr': 1e-7
        }
    
    def _build_lgbm_model(self) -> LGBMRegressor:
        """构建LightGBM模型"""
        return LGBMRegressor(**self.config['lgbm_params'])
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """构建改进的LSTM模型"""
        inputs = Input(shape=input_shape, name='sequence_input')
        
        # 双层LSTM
        x = LSTM(
            self.config['lstm_units'][0],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(inputs)
        
        x = BatchNormalization()(x)
        
        x = LSTM(
            self.config['lstm_units'][1],
            return_sequences=False,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
        
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        
        # 全连接层
        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = Dropout(self.config['dropout_rate'] / 2)(x)
        
        # 输出层
        outputs = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_lstm_data(self, X_sequence: np.ndarray) -> np.ndarray:
        """
        准备LSTM的输入数据
        
        Args:
            X_sequence: 时序特征
            
        Returns:
            处理后的时序数据
        """
        if len(X_sequence) == 0:
            return X_sequence
            
        # 确保数据形状正确
        if len(X_sequence.shape) == 3:
            return X_sequence
        else:
            raise ValueError(f"Expected 3D input for LSTM, got {len(X_sequence.shape)}D")
    
    def fit(self, X_sequence: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        训练混合模型
        
        Args:
            X_sequence: 时序特征，形状为 (samples, timesteps, features)
            y: 目标变量，形状为 (samples,)
            
        Returns:
            训练结果字典
        """
        self.logger.info(f"开始混合模型训练，样本数量: {len(y)}")
        self.logger.info(f"输入序列形状: {X_sequence.shape}")
        
        # 数据验证
        if len(X_sequence) != len(y):
            raise ValueError("X_sequence and y must have the same length")
        
        if len(X_sequence.shape) != 3:
            raise ValueError(f"Expected 3D input for X_sequence, got {len(X_sequence.shape)}D")
        
        # 训练LightGBM模型 - 使用每个序列的最后一个时间步
        self.logger.info("训练LightGBM模型...")
        self.lgbm_model = self._build_lgbm_model()
        
        X_static_last = X_sequence[:, -1, :]  # 使用LSTM输入的最后一帧
        self.logger.info(f"LightGBM input shape: {X_static_last.shape}")
        
        # 训练LightGBM
        self.lgbm_model.fit(
            X_static_last, 
            y,
            eval_set=[(X_static_last, y)],
            callbacks=[early_stopping(10)],
            verbose=False
        )
        
        # 训练LSTM模型
        self.logger.info("训练LSTM模型...")
        self.lstm_model = self._build_lstm_model(X_sequence.shape[1:])
        
        # 改进的回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_lstm_model.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # 训练LSTM
        history = self.lstm_model.fit(
            X_sequence, y,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_split'],
            callbacks=callbacks,
            verbose=1
        )
        
        # 优化集成权重
        if self.config['weight_optimization']:
            self._optimize_ensemble_weights(X_static_last, X_sequence, y)
        
        # 计算特征重要性
        self._calculate_feature_importance(X_sequence)
        
        # 返回详细的训练结果
        train_results = {
            'lstm_best_epoch': len(history.history['loss']) - self.config['early_stopping_patience'],
            'lstm_final_loss': min(history.history['val_loss']),
            'lstm_final_mae': min(history.history['val_mae']),
            'ensemble_weights': self.ensemble_weights,
            'training_samples': len(y),
            'feature_count': X_sequence.shape[-1],
            'sequence_length': X_sequence.shape[1]
        }
        
        self.logger.info("混合模型训练完成")
        self.logger.info(f"训练结果: {train_results}")
        
        return train_results
    
    def _optimize_ensemble_weights(self, X_static: np.ndarray, 
                                  X_sequence: np.ndarray, 
                                  y: np.ndarray):
        """优化集成权重 - 使用Ridge Regression允许负权重"""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV
        
        # 获取两个模型的预测
        lgbm_pred = self.lgbm_model.predict(X_static)
        lstm_pred = self.lstm_model.predict(X_sequence, verbose=0).flatten()
        
        # 使用网格搜索找到最优Ridge参数
        predictions = np.column_stack([lgbm_pred, lstm_pred])
        
        # 网格搜索优化Ridge回归参数
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        ridge = Ridge(fit_intercept=False)
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(predictions, y)
        
        # 使用最优参数进行权重优化
        best_ridge = grid_search.best_estimator_
        best_ridge.fit(predictions, y)
        
        # 获取权重
        weights = best_ridge.coef_
        
        # 允许负权重，但进行归一化
        weights_sum = np.sum(np.abs(weights))
        if weights_sum < 1e-6:
            weights = np.array([0.5, 0.5])
        else:
            # 归一化权重
            weights = weights / weights_sum
        
        self.ensemble_weights = {
            'lgbm': weights[0],
            'lstm': weights[1]
        }
        
        self.logger.info(f"优化后的集成权重: {self.ensemble_weights}")
    
    def _validate_input_data(func):
        """数据验证装饰器"""
        def wrapper(self, X_sequence, *args, **kwargs):
            if X_sequence is None or len(X_sequence) == 0:
                raise ValueError("Input data cannot be empty")
            
            if len(X_sequence.shape) != 3:
                raise ValueError(f"Expected 3D input (samples, timesteps, features), got {len(X_sequence.shape)}D")
            
            # 检查NaN和无穷大值
            if np.any(np.isnan(X_sequence)) or np.any(np.isinf(X_sequence)):
                raise ValueError("Input data contains NaN or infinite values")
            
            return func(self, X_sequence, *args, **kwargs)
        return wrapper
    
    @_validate_input_data
    def predict(self, X_sequence: np.ndarray) -> np.ndarray:
        """
        使用混合模型进行预测
        
        Args:
            X_sequence: 时序特征，形状为 (samples, timesteps, features)
            
        Returns:
            预测结果，形状为 (samples,)
        """
        if self.lgbm_model is None or self.lstm_model is None:
            raise ValueError("Models not trained. Call fit() first.")
        
        # LightGBM预测 - 始终使用X_sequence的最后一帧
        X_static_last = X_sequence[:, -1, :]
        lgbm_pred = self.lgbm_model.predict(X_static_last)
        
        # LSTM预测
        lstm_pred = self.lstm_model.predict(X_sequence, verbose=0).flatten()
        
        # 集成预测
        if self.config['ensemble_method'] == 'weighted':
            final_pred = (
                self.ensemble_weights['lgbm'] * lgbm_pred +
                self.ensemble_weights['lstm'] * lstm_pred
            )
        else:
            # 简单平均
            final_pred = (lgbm_pred + lstm_pred) / 2
        
        return final_pred

    @_validate_input_data  
    def evaluate(self, X_sequence: np.ndarray, y: np.ndarray, dates: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估模型性能 - 包含金融指标
        
        Args:
            X_sequence: 时序特征，形状为 (samples, timesteps, features)
            y: 真实值，形状为 (samples,)
            dates: 可选的日期数组，用于计算时间序列指标
            
        Returns:
            评估指标字典
        """
        if len(X_sequence) != len(y):
            raise ValueError("X_sequence and y must have the same length")
        
        predictions = self.predict(X_sequence)
        
        # 基本统计指标
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mse)
        
        # 方向准确性
        actual_direction = np.sign(y)
        predicted_direction = np.sign(predictions)
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        # 盈利预测准确率
        profitable_mask = y > 0
        if np.sum(profitable_mask) > 0:
            profitable_accuracy = np.mean(predictions[profitable_mask] > 0)
        else:
            profitable_accuracy = 0.0
        
        # 计算R²得分
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # 计算预测收益率统计
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        actual_mean = np.mean(y)
        actual_std = np.std(y)
        
        # 计算相关系数
        correlation = np.corrcoef(y, predictions)[0, 1] if len(y) > 1 else 0.0
        
        # 计算金融指标
        # 夏普比率 (假设无风险收益率为0)
        returns = predictions
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        # 最大回撤
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # 信息比率 (相对于实际收益率)
        tracking_error = np.std(predictions - y)
        information_ratio = np.mean(predictions - y) / tracking_error if tracking_error > 0 else 0.0
        
        # 预测稳定性 (预测标准差与实际标准差的比率)
        std_ratio = pred_std / actual_std if actual_std > 0 else 0.0
        
        # 年化收益率
        annual_return = np.mean(returns) * 252  # 假设一年252个交易日
        
        # 年化波动率
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        # Calmar比率 (年化收益率/最大回撤)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        # Sortino比率 (年化收益率/下行标准差)
        negative_returns = returns[returns < 0]
        downside_deviation = np.sqrt(np.mean(negative_returns**2)) if len(negative_returns) > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else np.inf
        
        # 预测稳定性度量
        pred_stability = 1 - abs(pred_mean - actual_mean) / (actual_std + 1e-8)
        
        evaluation_results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2_score,
            'direction_accuracy': direction_accuracy,
            'profitable_accuracy': profitable_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'prediction_stability': pred_stability,
            'correlation': correlation,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'prediction_mean': pred_mean,
            'prediction_std': pred_std,
            'actual_mean': actual_mean,
            'actual_std': actual_std
        }
        
        # 如果提供了日期，计算时间序列相关指标
        if dates is not None and len(dates) == len(y):
            # 计算预测序列的自相关性
            pred_autocorr = np.corrcoef(predictions[:-1], predictions[1:])[0, 1] if len(predictions) > 1 else 0.0
            evaluation_results['prediction_autocorrelation'] = pred_autocorr
            
            # 计算滚动窗口指标
            window_size = min(30, len(predictions) // 4)  # 使用1/4数据长度或30天，取较小值
            if window_size > 5:
                rolling_sharpe = []
                for i in range(window_size, len(predictions)):
                    window_returns = predictions[i-window_size:i]
                    if np.std(window_returns) > 0:
                        rolling_sharpe.append(np.mean(window_returns) / np.std(window_returns))
                if rolling_sharpe:
                    evaluation_results['rolling_sharpe_mean'] = np.mean(rolling_sharpe)
                    evaluation_results['rolling_sharpe_std'] = np.std(rolling_sharpe)
        
        self.logger.info("模型评估完成")
        self.logger.info(f"评估指标: {evaluation_results}")
        
        return evaluation_results
    
    def get_model_summary(self) -> Dict:
        """获取模型配置和状态摘要"""
        return {
            'model_type': 'LSTM+LightGBM Hybrid',
            'config': self.config,
            'ensemble_weights': self.ensemble_weights,
            'trained': self.lgbm_model is not None and self.lstm_model is not None,
            'feature_count': len(self.feature_importance_cache) if self.feature_importance_cache else None
        }
    
    def _calculate_feature_importance(self, X_sequence: np.ndarray):
        """计算特征重要性"""
        try:
            # LightGBM特征重要性
            if self.lgbm_model is not None:
                lgbm_importance = self.lgbm_model.feature_importances_
                self.feature_importance_cache['lgbm'] = lgbm_importance
                
            # LSTM特征重要性 (使用排列重要性作为近似)
            if self.lstm_model is not None and len(X_sequence) > 0:
                # 这里简化处理，实际应用中可能需要更复杂的特征重要性计算
                # 对于LSTM，特征重要性计算比较复杂，这里仅作占位符
                self.feature_importance_cache['lstm'] = np.ones(X_sequence.shape[-1])
                
            self.logger.info("特征重要性计算完成")
        except Exception as e:
            self.logger.warning(f"计算特征重要性时出错: {str(e)}")
    
    def get_top_features(self, top_n: int = 20) -> Dict[str, List[Tuple[int, float]]]:
        """
        获取最重要的特征
        
        Args:
            top_n: 返回前N个特征
            
        Returns:
            各模型的特征重要性排序
        """
        top_features = {}
        
        for model_name, importance in self.feature_importance_cache.items():
            # 获取重要性排序
            sorted_idx = np.argsort(importance)[::-1][:top_n]
            top_features[model_name] = [(idx, importance[idx]) for idx in sorted_idx]
            
        return top_features
    
    def save_models(self, model_dir: str = "models"):
        """
        保存模型和配置
        
        Args:
            model_dir: 模型保存目录
        """
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存LightGBM模型
        if self.lgbm_model is not None:
            lgbm_path = os.path.join(model_dir, "lgbm_model.pkl")
            joblib.dump(self.lgbm_model, lgbm_path)
            
        # 保存LSTM模型
        if self.lstm_model is not None:
            lstm_path = os.path.join(model_dir, "lstm_model.keras")
            self.lstm_model.save(lstm_path)
            
        # 保存配置和权重
        config_path = os.path.join(model_dir, "model_config.pkl")
        model_state = {
            'config': self.config,
            'ensemble_weights': self.ensemble_weights,
            'feature_importance': self.feature_importance_cache
        }
        joblib.dump(model_state, config_path)
        
        self.logger.info(f"模型已保存到 {model_dir}")
    
    def load_models(self, model_dir: str = "models"):
        """
        加载模型和配置
        
        Args:
            model_dir: 模型保存目录
        """
        # 检查模型目录
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
            
        # 加载LightGBM模型
        lgbm_path = os.path.join(model_dir, "lgbm_model.pkl")
        if os.path.exists(lgbm_path):
            self.lgbm_model = joblib.load(lgbm_path)
            
        # 加载LSTM模型
        lstm_path = os.path.join(model_dir, "lstm_model.keras")
        if os.path.exists(lstm_path):
            self.lstm_model = tf.keras.models.load_model(lstm_path)
            
        # 加载配置和权重
        config_path = os.path.join(model_dir, "model_config.pkl")
        if os.path.exists(config_path):
            model_state = joblib.load(config_path)
            self.config = model_state.get('config', self.config)
            self.ensemble_weights = model_state.get('ensemble_weights', self.ensemble_weights)
            self.feature_importance_cache = model_state.get('feature_importance', {})
            
        self.logger.info(f"模型已从 {model_dir} 加载")
    
    def get_recommendations(self, X_sequence: np.ndarray, 
                          stock_codes: Optional[List[str]] = None,
                          top_n: int = 50) -> pd.DataFrame:
        """
        生成股票推荐
        
        Args:
            X_sequence: 时序特征，形状为 (samples, timesteps, features)
            stock_codes: 股票代码列表
            top_n: 返回前N个推荐
            
        Returns:
            推荐结果DataFrame
        """
        if self.lgbm_model is None or self.lstm_model is None:
            raise ValueError("模型未训练，请先调用fit()方法")
            
        # 进行预测
        predictions = self.predict(X_sequence)
        
        # 创建推荐DataFrame
        if stock_codes is not None:
            if len(stock_codes) != len(predictions):
                raise ValueError("股票代码数量与预测结果数量不匹配")
            recommendation_df = pd.DataFrame({
                'ts_code': stock_codes,
                'prediction': predictions
            })
        else:
            recommendation_df = pd.DataFrame({
                'index': range(len(predictions)),
                'prediction': predictions
            })
            
        # 按预测值排序
        recommendation_df = recommendation_df.sort_values('prediction', ascending=False)
        
        # 返回前N个推荐
        return recommendation_df.head(top_n)

# LightGBM的early_stopping回调函数
from lightgbm import early_stopping

__all__ = ['LSTMLGBMPredictor']