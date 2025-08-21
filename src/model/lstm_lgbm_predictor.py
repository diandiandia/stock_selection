from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Attention, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import shap
from src.utils.log_helper import LogHelper
import gc
import psutil

# 启用混合精度训练以减少内存使用
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class LSTMLGBMPredictor:
    """LSTM + LightGBM混合预测模型，专为T+1股票推荐设计"""
    
    TRADING_DAYS = 252  # 年交易日数
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = LogHelper.get_logger(__name__)
        self.config = config or self._get_default_config()
        self.lgbm_model = None
        self.lstm_model = None
        self.ensemble_weights = {'lgbm': 0.5, 'lstm': 0.5}
        self.feature_importance_cache = {}
        
    def _get_default_config(self) -> Dict:
        """获取默认模型配置 - 优化版"""
        return {
            'lookback_window': 20,
            'test_size': 0.2,
            # LightGBM配置（优化：降低复杂度）
            'lgbm_params': {
                'n_estimators': 1000,  # 减少树数量
                'learning_rate': 0.05,  # 提高学习率以加速收敛
                'max_depth': 8,  # 降低深度
                'num_leaves': 64,  # 减少叶子节点
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.3,  # 增强正则化
                'reg_lambda': 0.3,
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1
            },
            # LSTM配置（优化：减少单位数，降低内存需求）
            'lstm_units': [64, 32],  # 减少层数
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.1,
            'batch_size': 512,  # 增加batch size以加速
            'epochs': 50,  # 减少epochs
            'learning_rate': 0.001,
            'patience': 20,
            'attention_dropout': 0.1,
            'ensemble_method': 'weighted',
            'weight_optimization': True,
            'validation_split': 0.2,  # 增加验证集比例
            'early_stopping_patience': 20,
            'reduce_lr_patience': 10,
            'min_lr': 1e-7
        }

    def _build_lgbm_model(self) -> LGBMRegressor:
        """构建LightGBM模型"""
        return LGBMRegressor(**self.config['lgbm_params'])
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """构建优化后的LSTM模型（减少层数）"""
        inputs = Input(shape=input_shape, name='sequence_input', dtype='float16')
        
        # 第一层LSTM
        x = LSTM(
            self.config['lstm_units'][0],
            return_sequences=True,
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_regularizer=tf.keras.regularizers.l2(5e-4)
        )(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        
        # 第二层LSTM（减少到2层）
        x = LSTM(
            self.config['lstm_units'][1],
            return_sequences=True,
            recurrent_dropout=self.config['recurrent_dropout'],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
        x = BatchNormalization()(x)
        
        # 注意力机制
        x = Attention(use_scale=True, dropout=self.config['attention_dropout'])([x, x])
        x = GlobalAveragePooling1D()(x)
        
        # 全连接层
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        
        outputs = Dense(1, activation='linear', name='output', dtype='float32')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.config['learning_rate'],
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model

    def prepare_lstm_data(self, X_sequence: np.ndarray) -> np.ndarray:
        """标准化时序数据"""
        mean = np.mean(X_sequence, axis=1, keepdims=True)
        std = np.std(X_sequence, axis=1, keepdims=True) + 1e-8
        X_normalized = (X_sequence - mean) / std
        return X_normalized.astype(np.float16)  # 使用float16减少内存
    
    def fit(self, X_sequence: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练混合模型"""
        self.logger.info(f"开始混合模型训练，样本数量: {len(y)}")
        self.logger.info(f"输入序列形状: {X_sequence.shape}")
        
        # 数据标准化
        X_sequence = self.prepare_lstm_data(X_sequence)
        
        # 数据验证
        if len(X_sequence) != len(y):
            raise ValueError("X_sequence and y must have the same length")
        if len(X_sequence.shape) != 3:
            raise ValueError(f"Expected 3D input for X_sequence, got {len(X_sequence.shape)}D")
        
        # 训练LightGBM
        self.logger.info("训练LightGBM模型...")
        self.lgbm_model = self._build_lgbm_model()
        X_static_last = X_sequence[:, -1, :]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_static_last, y, 
            test_size=self.config['validation_split'],
            random_state=42
        )
        
        self.lgbm_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(self.config['early_stopping_patience'], verbose=True)]
        )
        
        # 内存监控和GPU检查
        self.logger.info(f"可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            self.logger.info(f"使用GPU: {len(gpus)} 个")
        else:
            self.logger.info("无GPU，使用CPU")
        
        # 训练LSTM
        self.logger.info("训练LSTM模型...")
        self.lstm_model = self._build_lstm_model(X_sequence.shape[1:])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.config['early_stopping_patience'], 
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.config['reduce_lr_patience'], 
                             min_lr=self.config['min_lr'], verbose=1),
            ModelCheckpoint('models/best_lstm_model.keras', monitor='val_loss', 
                           save_best_only=True, save_weights_only=False, verbose=1)
        ]
        
        history = self.lstm_model.fit(
            X_sequence, y,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_split'],
            callbacks=callbacks,
            verbose=1  # 更详细日志
        )
        
        # 优化集成权重
        if self.config['weight_optimization']:
            self._optimize_ensemble_weights(X_static_last, X_sequence, y)
        
        # 计算特征重要性
        self._calculate_feature_importance(X_sequence)
        
        # 清理内存
        gc.collect()
        
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
    
    def _optimize_ensemble_weights(self, X_static: np.ndarray, X_sequence: np.ndarray, y: np.ndarray):
        """优化集成权重"""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV
        
        lgbm_pred = self.lgbm_model.predict(X_static)
        lstm_pred = self.lstm_model.predict(X_sequence, verbose=0).flatten()
        
        predictions = np.column_stack([lgbm_pred, lstm_pred])
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        ridge = Ridge(fit_intercept=False)
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(predictions, y)
        
        best_ridge = grid_search.best_estimator_
        best_ridge.fit(predictions, y)
        
        weights = best_ridge.coef_
        weights_sum = np.sum(np.abs(weights))
        if weights_sum < 1e-6:
            weights = np.array([0.5, 0.5])
        else:
            weights = weights / weights_sum
        
        self.ensemble_weights = {'lgbm': weights[0], 'lstm': weights[1]}
        self.logger.info(f"优化后的集成权重: {self.ensemble_weights}")
    
    def _validate_input_data(func):
        """数据验证装饰器"""
        def wrapper(self, X_sequence, *args, **kwargs):
            if X_sequence is None or len(X_sequence) == 0:
                raise ValueError("Input data cannot be empty")
            if len(X_sequence.shape) != 3:
                raise ValueError(f"Expected 3D input (samples, timesteps, features), got {len(X_sequence.shape)}D")
            if np.any(np.isnan(X_sequence)) or np.any(np.isinf(X_sequence)):
                raise ValueError("Input data contains NaN or infinite values")
            return func(self, X_sequence, *args, **kwargs)
        return wrapper
    
    @_validate_input_data
    def predict(self, X_sequence: np.ndarray) -> np.ndarray:
        """使用混合模型进行预测"""
        if self.lgbm_model is None or self.lstm_model is None:
            raise ValueError("Models not trained. Call fit() first.")
        
        X_sequence = self.prepare_lstm_data(X_sequence)
        X_static_last = X_sequence[:, -1, :]
        lgbm_pred = self.lgbm_model.predict(X_static_last)
        lstm_pred = self.lstm_model.predict(X_sequence, verbose=0).flatten()
        
        final_pred = (
            self.ensemble_weights['lgbm'] * lgbm_pred +
            self.ensemble_weights['lstm'] * lstm_pred
        )
        
        return final_pred

    @_validate_input_data  
    def evaluate(self, X_sequence: np.ndarray, y: np.ndarray, dates: Optional[np.ndarray] = None) -> Dict[str, float]:
        """评估模型性能"""
        if len(X_sequence) != len(y):
            raise ValueError("X_sequence and y must have the same length")
        
        predictions = self.predict(X_sequence)
        
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mse)
        
        actual_direction = np.sign(y)
        predicted_direction = np.sign(predictions)
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        profitable_mask = y > 0
        profitable_accuracy = np.mean(predictions[profitable_mask] > 0) if np.sum(profitable_mask) > 0 else 0.0
        
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        actual_mean = np.mean(y)
        actual_std = np.std(y)
        
        correlation = np.corrcoef(y, predictions)[0, 1] if len(y) > 1 else 0.0
        
        financial_metrics = self._calculate_financial_metrics(predictions, y)
        
        evaluation_results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2_score,
            'direction_accuracy': direction_accuracy,
            'profitable_accuracy': profitable_accuracy,
            'sharpe_ratio': financial_metrics['sharpe_ratio'],
            'max_drawdown': financial_metrics['max_drawdown'],
            'information_ratio': financial_metrics['information_ratio'],
            'prediction_stability': financial_metrics['prediction_stability'],
            'correlation': correlation,
            'annual_return': financial_metrics['annual_return'],
            'annual_volatility': financial_metrics['annual_volatility'],
            'calmar_ratio': financial_metrics['calmar_ratio'],
            'sortino_ratio': financial_metrics['sortino_ratio'],
            'prediction_mean': pred_mean,
            'prediction_std': pred_std,
            'actual_mean': actual_mean,
            'actual_std': actual_std
        }
        
        if dates is not None and len(dates) == len(y):
            pred_autocorr = np.corrcoef(predictions[:-1], predictions[1:])[0, 1] if len(predictions) > 1 else 0.0
            evaluation_results['prediction_autocorrelation'] = pred_autocorr
            
            window_size = min(30, len(predictions) // 4)
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
    
    def _calculate_financial_metrics(self, predictions: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """计算金融评估指标"""
        returns = predictions
        
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        tracking_error = np.std(predictions - y)
        information_ratio = np.mean(predictions - y) / tracking_error if tracking_error > 0 else 0.0
        
        actual_std = np.std(y)
        pred_std = np.std(predictions)
        prediction_stability = 1 - abs(np.mean(predictions) - np.mean(y)) / (actual_std + 1e-8)
        
        annual_return = np.mean(returns) * self.TRADING_DAYS
        annual_volatility = np.std(returns) * np.sqrt(self.TRADING_DAYS)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        negative_returns = returns[returns < 0]
        downside_deviation = np.sqrt(np.mean(negative_returns**2)) if len(negative_returns) > 0 else 0.0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else np.inf
        
        correlation = np.corrcoef(y, predictions)[0, 1] if len(y) > 1 else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'prediction_stability': prediction_stability,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'correlation': correlation
        }
    
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
            if self.lgbm_model is not None:
                self.feature_importance_cache['lgbm'] = self.lgbm_model.feature_importances_
            
            if self.lstm_model is not None and len(X_sequence) > 0:
                explainer = shap.GradientExplainer(self.lstm_model, X_sequence[:100])
                shap_values = explainer.shap_values(X_sequence[:100])
                self.feature_importance_cache['lstm'] = np.mean(np.abs(shap_values[0]), axis=(0,1))
                
            self.logger.info("特征重要性计算完成")
        except Exception as e:
            self.logger.warning(f"计算特征重要性时出错: {str(e)}")
    
    def get_top_features(self, top_n: int = 20) -> Dict[str, List[Tuple[int, float]]]:
        """获取最重要的特征"""
        top_features = {}
        for model_name, importance in self.feature_importance_cache.items():
            sorted_idx = np.argsort(importance)[::-1][:top_n]
            top_features[model_name] = [(idx, importance[idx]) for idx in sorted_idx]
        return top_features
    
    def save_models(self, model_dir: str = "models"):
        """保存模型和配置"""
        os.makedirs(model_dir, exist_ok=True)
        if self.lgbm_model is not None:
            joblib.dump(self.lgbm_model, os.path.join(model_dir, "lgbm_model.pkl"))
        if self.lstm_model is not None:
            self.lstm_model.save(os.path.join(model_dir, "lstm_model.keras"))
        
        model_state = {
            'config': self.config,
            'ensemble_weights': self.ensemble_weights,
            'feature_importance': self.feature_importance_cache
        }
        joblib.dump(model_state, os.path.join(model_dir, "model_config.pkl"))
        self.logger.info(f"模型已保存到 {model_dir}")
    
    def load_models(self, model_dir: str = "models"):
        """加载模型和配置"""
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        
        lgbm_path = os.path.join(model_dir, "lgbm_model.pkl")
        if os.path.exists(lgbm_path):
            self.lgbm_model = joblib.load(lgbm_path)
        
        lstm_path = os.path.join(model_dir, "lstm_model.keras")
        if os.path.exists(lstm_path):
            self.lstm_model = tf.keras.models.load_model(lstm_path)
        
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
        """生成股票推荐"""
        if self.lgbm_model is None or self.lstm_model is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        predictions = self.predict(X_sequence)
        
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
        
        recommendation_df = recommendation_df.sort_values('prediction', ascending=False)
        return recommendation_df.head(top_n)