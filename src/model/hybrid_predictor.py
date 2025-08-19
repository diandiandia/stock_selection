from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import GRU, Dense, Dropout, Input, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from src.utils.log_helper import LogHelper

class HybridPredictor:
    """XGBoost + GRU混合预测模型，专为T+1股票推荐设计"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化混合预测模型
        
        Args:
            config: 模型配置字典
        """
        self.logger = LogHelper.get_logger(__name__)
        self.config = config or self._get_default_config()
        
        # 初始化模型
        self.xgb_model = None
        self.gru_model = None
        self.ensemble_weights = {'xgb': 0.5, 'gru': 0.5}
        
        # 特征重要性缓存
        self.feature_importance_cache = {}
        
    def _get_default_config(self) -> Dict:
        """获取默认模型配置"""
        return {
            # 数据配置
            'lookback_window': 20,
            'test_size': 0.2,
            
            # XGBoost配置 - 优化参数
            'xgb_params': {
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 8,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'objective': 'reg:squarederror',
                'early_stopping_rounds': 20,
                'eval_metric': 'rmse'
            },
            
            # GRU配置 - 优化网络结构
            'gru_units': [256, 128, 64],
            'dropout_rate': 0.4,
            'recurrent_dropout': 0.3,
            'batch_size': 32,
            # 'epochs': 100,
            'epochs': 1, # 测试时设置为1以加快速度
            'learning_rate': 0.001,
            'patience': 15,
            
            # 集成配置
            'ensemble_method': 'weighted',  # 'weighted' 或 'stacking'
            'weight_optimization': True,
            
            # 训练配置
            'validation_split': 0.15,
            'early_stopping_patience': 20,
            'reduce_lr_patience': 8,
            'min_lr': 1e-7
        }
    
    def _build_xgb_model(self) -> XGBRegressor:
        """构建XGBoost模型"""
        return XGBRegressor(**self.config['xgb_params'])
    
    def _build_gru_model(self, input_shape: Tuple[int, int]) -> Model:
        """构建改进的GRU模型"""
        inputs = Input(shape=input_shape, name='sequence_input')
        
        # 第一层GRU
        x = GRU(
            self.config['gru_units'][0],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        
        # 第二层GRU
        if len(self.config['gru_units']) > 1:
            x = GRU(
                self.config['gru_units'][1],
                return_sequences=False,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['recurrent_dropout'],
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config['dropout_rate'])(x)
        
        # 全连接层 - 改进结构
        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'] / 2)(x)
        
        x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = Dropout(self.config['dropout_rate'] / 4)(x)
        
        # 输出层
        outputs = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型 - 添加学习率调度
        initial_learning_rate = self.config['learning_rate']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
        
        optimizer = Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_gru_data(self, X_sequence: np.ndarray) -> np.ndarray:
        """
        准备GRU的输入数据
        
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
            raise ValueError(f"Expected 3D input for GRU, got {len(X_sequence.shape)}D")
    
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
        
        # 训练XGBoost模型 - 使用每个序列的最后一个时间步
        self.logger.info("训练XGBoost模型...")
        self.xgb_model = self._build_xgb_model()
        
        X_static_last = X_sequence[:, -1, :]  # 使用GRU输入的最后一帧
        self.logger.info(f"XGBoost input shape: {X_static_last.shape}")
        
        # 训练XGBoost
        self.xgb_model.fit(
            X_static_last, 
            y,
            eval_set=[(X_static_last, y)],
            verbose=False
        )
        
        # 训练GRU模型
        self.logger.info("训练GRU模型...")
        self.gru_model = self._build_gru_model(X_sequence.shape[1:])
        
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
                'models/best_gru_model.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # 训练GRU
        history = self.gru_model.fit(
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
            'gru_best_epoch': len(history.history['loss']) - self.config['early_stopping_patience'],
            'gru_final_loss': min(history.history['val_loss']),
            'gru_final_mae': min(history.history['val_mae']),
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
        """优化集成权重"""
        from sklearn.linear_model import LinearRegression
        
        # 获取两个模型的预测
        xgb_pred = self.xgb_model.predict(X_static)
        gru_pred = self.gru_model.predict(X_sequence, verbose=0).flatten()
        
        # 使用线性回归找到最优权重
        predictions = np.column_stack([xgb_pred, gru_pred])
        
        # 限制权重在合理范围内
        lr = LinearRegression(fit_intercept=False)
        lr.fit(predictions, y)
        
        # 归一化权重
        weights = lr.coef_
        weights = np.clip(weights, 0.1, 0.9)  # 避免极端权重
        weights = weights / np.sum(weights)
        
        self.ensemble_weights = {
            'xgb': weights[0],
            'gru': weights[1]
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
        if self.xgb_model is None or self.gru_model is None:
            raise ValueError("Models not trained. Call fit() first.")
        
        # XGBoost预测 - 始终使用X_sequence的最后一帧
        X_static_last = X_sequence[:, -1, :]
        xgb_pred = self.xgb_model.predict(X_static_last)
        
        # GRU预测
        gru_pred = self.gru_model.predict(X_sequence, verbose=0).flatten()
        
        # 集成预测
        if self.config['ensemble_method'] == 'weighted':
            final_pred = (
                self.ensemble_weights['xgb'] * xgb_pred +
                self.ensemble_weights['gru'] * gru_pred
            )
        else:
            # 简单平均
            final_pred = (xgb_pred + gru_pred) / 2
        
        return final_pred

    @_validate_input_data  
    def evaluate(self, X_sequence: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_sequence: 时序特征，形状为 (samples, timesteps, features)
            y: 真实值，形状为 (samples,)
            
        Returns:
            评估指标字典
        """
        if len(X_sequence) != len(y):
            raise ValueError("X_sequence and y must have the same length")
        
        predictions = self.predict(X_sequence)
        
        # 计算评估指标
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mse)
        
        # 方向准确率
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
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2_score,
            'direction_accuracy': direction_accuracy,
            'profitable_accuracy': profitable_accuracy,
            'predicted_mean': pred_mean,
            'predicted_std': pred_std,
            'actual_mean': actual_mean,
            'actual_std': actual_std,
            'correlation': correlation,
            'sample_count': len(y)
        }
    
    def get_model_summary(self) -> Dict[str, any]:
        """
        获取模型摘要信息
        
        Returns:
            包含模型配置和状态的摘要字典
        """
        summary = {
            'config': self.config,
            'models_initialized': {
                'xgb': self.xgb_model is not None,
                'gru': self.gru_model is not None
            },
            'ensemble_weights': self.ensemble_weights,
            'feature_importance_computed': bool(self.feature_importance_cache)
        }
        
        if self.xgb_model is not None:
            summary['xgb_params'] = self.xgb_model.get_params()
            
        if self.gru_model is not None:
            summary['gru_model_summary'] = self.gru_model.summary()
            
        return summary
    
    def _calculate_feature_importance(self, X_sequence: np.ndarray):
        """计算特征重要性 - 包括XGBoost和GRU"""
        if self.xgb_model is not None:
            importance = self.xgb_model.feature_importances_
            self.feature_importance_cache['xgb'] = importance
            self.logger.info(f"XGBoost特征重要性计算完成，前5个特征索引: {np.argsort(importance)[-5:]}")
        
        # 为GRU模型计算特征重要性（使用SHAP值的简化版本）
        if self.gru_model is not None:
            try:
                # 使用梯度作为特征重要性的近似
                # 这里简化处理，实际应用中可以使用SHAP或LIME
                self.feature_importance_cache['gru'] = np.ones(X_sequence.shape[-1])  # 修复占位符问题
                self.logger.info("GRU特征重要性占位符已设置")
            except Exception as e:
                self.logger.warning(f"GRU特征重要性计算失败: {e}")
        
        self.logger.info("特征重要性计算完成")
    
    def get_top_features(self, n_features: int = 10) -> Dict[str, np.ndarray]:
        """
        获取最重要的特征
        
        Args:
            n_features: 要返回的特征数量
            
        Returns:
            包含特征索引和重要性分数的字典
        """
        result = {}
        
        # XGBoost特征重要性
        if 'xgb' in self.feature_importance_cache:
            importance = self.feature_importance_cache['xgb']
            indices = np.argsort(importance)[::-1][:n_features]
            result['xgb'] = {
                'indices': indices,
                'importance': importance[indices]
            }
        
        # GRU特征重要性（占位符）
        if 'gru' in self.feature_importance_cache:
            importance = self.feature_importance_cache['gru']
            indices = np.argsort(importance)[::-1][:n_features]
            result['gru'] = {
                'indices': indices,
                'importance': importance[indices]
            }
        
        self.logger.info(f"Retrieved top {n_features} features for both models")
        return result
    
    def save_models(self, path: str) -> None:
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        
        # 保存XGBoost模型
        joblib.dump(self.xgb_model, f"{path}/xgb_model.joblib")
        
        # 保存GRU模型
        self.gru_model.save(f"{path}/gru_model.keras")
        
        # 保存配置和权重
        save_data = {
            'config': self.config,
            'ensemble_weights': self.ensemble_weights,
            'feature_importance': self.feature_importance_cache
        }
        joblib.dump(save_data, f"{path}/model_config.joblib")
        
        self.logger.info(f"模型已保存至: {path}")
    
    def load_models(self, path: str) -> None:
        """加载模型"""
        # 加载XGBoost模型
        self.xgb_model = joblib.load(f"{path}/xgb_model.joblib")
        
        # 加载GRU模型
        self.gru_model = tf.keras.models.load_model(f"{path}/gru_model.keras")
        
        # 加载配置
        save_data = joblib.load(f"{path}/model_config.joblib")
        self.config = save_data['config']
        self.ensemble_weights = save_data['ensemble_weights']
        self.feature_importance_cache = save_data['feature_importance']
        
        self.logger.info(f"模型已从: {path} 加载完成")
    
    def get_recommendations(self, predictions: Dict[str, float], 
                           top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取股票推荐
        
        Args:
            predictions: 股票代码到预测收益率的映射
            top_n: 推荐数量
            
        Returns:
            按预测收益率排序的推荐列表
        """
        # 排序并返回前N个
        sorted_predictions = sorted(
            predictions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_predictions[:top_n]