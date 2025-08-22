from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Attention
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import losses
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
        self.ensemble_weights = {'lgbm': 0.5, 'lstm': 0.5}  # 初始化为均等权重
        self.feature_importance_cache = {}
        
    def _get_default_config(self) -> Dict:
        """获取默认模型配置 - 优化版"""
        return {
            'lookback_window': 10,  # 与 main.py 保持一致
            'test_size': 0.2,
            'lgbm_params': {
                'n_estimators': 1500,  # 增加迭代次数以恢复RMSE ~0.496111
                'learning_rate': 0.02,  # 略降低以提高精度
                'max_depth': 6,
                'num_leaves': 32,
                'min_child_samples': 30,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1
            },
            'lstm_units': [64, 32],  # 保持两层以提升容量
            'dropout_rate': 0.3,
            'recurrent_dropout': 0.2,
            'batch_size': 128,  # 减小以适应1.14 GB内存
            'epochs': 100,
            'learning_rate': 0.0002,  # 进一步降低以稳定训练
            'patience': 15,
            'attention_dropout': 0.2,
            'ensemble_method': 'weighted',
            'weight_optimization': True,
            'validation_split': 0.2,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 7,
            'min_lr': 1e-6
        }

    def _build_lgbm_model(self) -> LGBMRegressor:
        """构建LightGBM模型"""
        return LGBMRegressor(**self.config['lgbm_params'])
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """构建优化后的LSTM模型"""
        if input_shape is None:
            self.logger.error("Input shape is None")
            return None
        
        inputs = Input(shape=input_shape, dtype=tf.float16)
        x = inputs
        
        # 确保所有LSTM层（除最后一层外）返回序列
        for i, units in enumerate(self.config['lstm_units']):
            return_sequences = True  # 保持序列输出以兼容Attention
            x = LSTM(units=units, 
                     return_sequences=return_sequences,
                     recurrent_dropout=self.config['recurrent_dropout'])(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config['dropout_rate'])(x)
        
        # 应用Attention机制
        attention = Attention(use_scale=True, dropout=self.config['attention_dropout'])([x, x])
        x = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        x = Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        outputs = Dense(1, dtype=tf.float32)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=self.config['learning_rate'], clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=self.custom_loss,
            metrics=['mae']
        )
        
        self.logger.info(f"LSTM model built with input shape: {input_shape}")
        return model
    
    def custom_loss(self, y_true, y_pred):
        """自定义损失函数：MSE + 方向准确性惩罚"""
        mse_loss = losses.MeanSquaredError()(y_true, y_pred)
        direction_penalty = tf.reduce_mean(tf.square(tf.sign(y_true) - tf.sign(y_pred)))
        return mse_loss + 0.1 * direction_penalty
    
    def fit(self, X_sequence: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练混合模型"""
        self.logger.info(f"开始混合模型训练，样本数量: {len(X_sequence)}")
        self.logger.info(f"输入序列形状: {X_sequence.shape}")
        
        # 检查输入数据
        if np.any(np.isnan(X_sequence)) or np.any(np.isnan(y)):
            self.logger.error("输入数据中存在NaN值")
            raise ValueError("输入数据中存在NaN值")
        
        # 训练LightGBM
        self.logger.info("训练LightGBM模型...")
        X_lgbm = X_sequence.reshape(X_sequence.shape[0], -1)
        X_train_lgbm, X_val_lgbm, y_train, y_val = train_test_split(
            X_lgbm, y, test_size=self.config['test_size'], shuffle=False)
        self.lgbm_model = self._build_lgbm_model()
        self.lgbm_model.fit(
            X_train_lgbm, y_train,
            eval_set=[(X_val_lgbm, y_val)],
            callbacks=[early_stopping(stopping_rounds=self.config['patience'], verbose=10)]
        )
        
        # 检查可用内存
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
        self.logger.info(f"可用内存: {available_memory:.2f} GB")
        
        # 训练LSTM
        self.logger.info("训练LSTM模型...")
        input_shape = (self.config['lookback_window'], X_sequence.shape[2])
        self.lstm_model = self._build_lstm_model(input_shape)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.config['early_stopping_patience'], 
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                            patience=self.config['reduce_lr_patience'], 
                            min_lr=self.config['min_lr'], verbose=1),
            ModelCheckpoint('models/lstm_best.keras', save_best_only=True, 
                           monitor='val_loss', verbose=1)
        ]
        
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
            self._optimize_ensemble_weights(X_sequence, y)
        
        # 缓存特征重要性
        if self.lgbm_model is not None:
            self.feature_importance_cache['lgbm'] = self.lgbm_model.feature_importances_
        
        return {
            'lstm_loss': history.history['loss'][-1],
            'lstm_val_loss': history.history['val_loss'][-1],
            'lgbm_rmse': self.lgbm_model.best_score_.get('valid_0', {}).get('rmse', np.nan)
        }
    
    def _optimize_ensemble_weights(self, X: np.ndarray, y: np.ndarray):
        """优化集成权重"""
        self.logger.info("优化集成权重...")
        weights = np.linspace(0, 1, 11)
        scores = []
        
        for w_lgbm in weights:
            w_lstm = 1 - w_lgbm
            preds = self.predict(X, weights={'lgbm': w_lgbm, 'lstm': w_lstm})
            score = mean_squared_error(y, preds)
            scores.append(score)
        
        best_idx = np.argmin(scores)
        self.ensemble_weights = {
            'lgbm': weights[best_idx],
            'lstm': 1 - weights[best_idx]
        }
        self.logger.info(f"最佳权重: LGBM {self.ensemble_weights['lgbm']:.2f}, LSTM {self.ensemble_weights['lstm']:.2f}")
    
    def predict(self, X_sequence: np.ndarray, weights: Optional[Dict] = None) -> np.ndarray:
        """生成预测"""
        if self.lgbm_model is None or self.lstm_model is None:
            raise ValueError("模型未训练")
        
        weights = weights or self.ensemble_weights
        
        X_lgbm = X_sequence.reshape(X_sequence.shape[0], -1)
        lgbm_preds = self.lgbm_model.predict(X_lgbm)
        lstm_preds = self.lstm_model.predict(X_sequence, verbose=0).flatten()
        
        return weights['lgbm'] * lgbm_preds + weights['lstm'] * lstm_preds
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, dates: Optional[np.ndarray] = None) -> Dict[str, float]:
        """评估模型性能"""
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        result = {'mse': mse, 'mae': mae}
        
        if dates is not None:
            df_eval = pd.DataFrame({
                'date': dates,
                'true': y_test,
                'pred': predictions
            }).sort_values('date')
            annual_return = (1 + df_eval['pred'].mean()) ** self.TRADING_DAYS - 1
            result['annual_return'] = annual_return
        
        return result
    
    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """计算SHAP值（仅LightGBM部分）"""
        self.logger.info("计算SHAP值...")
        if self.lgbm_model is None:
            raise ValueError("LightGBM模型未训练")
        
        X_lgbm = X.reshape(X.shape[0], -1)
        explainer = shap.TreeExplainer(self.lgbm_model)
        shap_values = explainer.shap_values(X_lgbm)
        # 调整SHAP值形状以匹配LSTM输入
        shap_values = shap_values.reshape(X.shape[0], X.shape[1], -1)
        return shap_values
    
    def get_top_features(self, top_n: int = 30) -> Dict[str, List[Tuple[int, float]]]:
        """获取顶部特征"""
        if not self.feature_importance_cache:
            self.logger.warning("特征重要性缓存为空")
            return {}
        
        importance = self.feature_importance_cache.get('lgbm', np.zeros(9))  # 调整为9个特征
        sorted_idx = np.argsort(importance)[::-1][:top_n]
        return {'lgbm': [(idx, importance[idx]) for idx in sorted_idx]}
    
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
            self.lstm_model = tf.keras.models.load_model(
                lstm_path, custom_objects={'custom_loss': self.custom_loss})
        
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