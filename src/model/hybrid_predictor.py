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
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.utils.log_helper import LogHelper
from typing import Dict, List, Optional

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
            
            # XGBoost配置
            'xgb_params': {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'reg:squarederror',
                'early_stopping_rounds': 10
            },
            
            # GRU配置
            'gru_units': [128, 64],
            'dropout_rate': 0.3,
            'recurrent_dropout': 0.2,
            'batch_size': 64,
            'epochs': 2,
            'learning_rate': 0.001,
            'patience': 15,
            
            # 集成配置
            'ensemble_method': 'weighted',  # 'weighted' 或 'stacking'
            'weight_optimization': True,
            
            # 训练配置
            'validation_split': 0.1,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5
        }
    
    def _build_xgb_model(self) -> XGBRegressor:
        """构建XGBoost模型"""
        return XGBRegressor(**self.config['xgb_params'])
    
    def _build_gru_model(self, input_shape: Tuple[int, int]) -> Model:
        """构建GRU模型"""
        inputs = Input(shape=input_shape, name='sequence_input')
        
        # 第一层GRU
        x = GRU(
            self.config['gru_units'][0],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['recurrent_dropout']
        )(inputs)
        x = BatchNormalization()(x)
        
        # 第二层GRU
        if len(self.config['gru_units']) > 1:
            x = GRU(
                self.config['gru_units'][1],
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['recurrent_dropout']
            )(x)
            x = BatchNormalization()(x)
        
        # 全连接层
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.config['dropout_rate'])(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(self.config['dropout_rate'] / 2)(x)
        
        # 输出层
        outputs = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        optimizer = Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=1.0
        )
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
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
    
    def fit(self, X_static: Optional[np.ndarray], X_sequence: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        训练混合模型
        
        Args:
            X_static: 静态特征（可选，实际上未使用）
            X_sequence: 时序特征
            y: 目标变量
            
        Returns:
            训练结果字典
        """
        self.logger.info(f"Starting hybrid model training with {len(y)} samples")
        
        # 数据验证 - 只验证X_sequence和y的长度
        if len(X_sequence) != len(y):
            raise ValueError("X_sequence and y must have the same length")
        
        # 训练XGBoost模型
        self.logger.info("Training XGBoost model...")
        self.xgb_model = self._build_xgb_model()
        
        # 对于XGBoost，我们使用每个序列的最后一个时间步
        X_static_last = X_sequence[:, -1, :]  # 使用GRU输入的最后一帧
        
        # 训练XGBoost
        self.xgb_model.fit(
            X_static_last, 
            y,
            eval_set=[(X_static_last, y)],
            verbose=False
        )
        
        # 训练GRU模型
        self.logger.info("Training GRU model...")
        self.gru_model = self._build_gru_model(X_sequence.shape[1:])
        
        # 回调函数
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
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_gru_model.keras',
                monitor='val_loss',
                save_best_only=True,
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
        self._calculate_feature_importance()
        
        # 返回训练结果
        train_results = {
            'gru_best_epoch': len(history.history['loss']) - self.config['early_stopping_patience'],
            'gru_final_loss': min(history.history['val_loss']),
            'ensemble_weights': self.ensemble_weights
        }
        
        self.logger.info("Hybrid model training completed")
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
        
        self.logger.info(f"Optimized ensemble weights: {self.ensemble_weights}")
    
    def predict(self, X_static: Optional[np.ndarray], X_sequence: np.ndarray) -> np.ndarray:
        """
        使用混合模型进行预测
        
        Args:
            X_static: 静态特征（可选，实际上未使用）
            X_sequence: 时序特征
            
        Returns:
            预测结果
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

    def evaluate(self, X_static: Optional[np.ndarray], X_sequence: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_static: 静态特征（可选，实际上未使用）
            X_sequence: 时序特征
            y: 真实值
            
        Returns:
            评估指标字典
        """
        predictions = self.predict(X_static, X_sequence)
        
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
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'profitable_accuracy': profitable_accuracy
        }
    
    def _calculate_feature_importance(self):
        """计算特征重要性"""
        if self.xgb_model is not None:
            importance = self.xgb_model.feature_importances_
            self.feature_importance_cache['xgb'] = importance
        
        self.logger.info("Feature importance calculated")
    
    def get_top_features(self, n_features: int = 10) -> Dict[str, np.ndarray]:
        """获取最重要的特征"""
        if 'xgb' in self.feature_importance_cache:
            indices = np.argsort(self.feature_importance_cache['xgb'])[::-1][:n_features]
            return {
                'indices': indices,
                'importance': self.feature_importance_cache['xgb'][indices]
            }
        return {}
    
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
        
        self.logger.info(f"Models saved to {path}")
    
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
        
        self.logger.info(f"Models loaded from {path}")
    
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