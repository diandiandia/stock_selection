from typing import List, Tuple, Optional, Dict
from unicodedata import bidirectional
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
from keras.layers import Bidirectional

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
        self.feature_columns = []  # Add this line

    def _get_default_config(self) -> Dict:
        return {
                'lookback_window': 10,
                'test_size': 0.2,
                'lgbm_params': {
                    'n_estimators': 2000,
                    'learning_rate': 0.01,
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
                'lstm_units': [64, 32],
                'dropout_rate': 0.3,  # 明确设置
                'recurrent_dropout': 0.0,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 50,
                'patience': 5  # 明确设置
        }

    def _build_lgbm_model(self) -> LGBMRegressor:
        """构建LightGBM模型"""
        return LGBMRegressor(**self.config['lgbm_params'])

    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(units=64, return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        x = Bidirectional(LSTM(units=32))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1, dtype='float32')(x)
        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss=self.custom_loss, metrics=['mae'])
        return model

    def custom_loss(self, y_true, y_pred):
        return tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)

    def fit(self, X_sequence: np.ndarray, y: np.ndarray, feature_columns: Optional[List[str]] = None) -> Dict[str, float]:
        """
        训练模型
        
        Args:
            X_sequence: 输入序列数据
            y: 目标变量
            feature_columns: 特征列名列表
        
        Returns:
            Dict[str, float]: 包含训练指标的字典
        """
        # 保存特征列名
        if feature_columns is not None:
            self.feature_columns = feature_columns
            self.logger.info(f"设置特征列: {len(self.feature_columns)}个特征")
        
        if self.lgbm_model is None:
            self.lgbm_model = self._build_lgbm_model()
        if self.lstm_model is None:
            self.lstm_model = self._build_lstm_model(
                input_shape=(self.config['lookback_window'], X_sequence.shape[2]))

        # LGBM训练
        X_lgbm = X_sequence.reshape(X_sequence.shape[0], -1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_lgbm, y, test_size=self.config['test_size'], shuffle=False)
        
        lgbm_eval_set = [(X_val, y_val)]
        self.lgbm_model.fit(
            X_train, y_train,
            eval_set=lgbm_eval_set,
            eval_metric='rmse',
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
        )
        lgbm_rmse = np.sqrt(mean_squared_error(y_val, self.lgbm_model.predict(X_val)))

        # LSTM训练
        early_stopping_callback = EarlyStopping(
            monitor='val_loss', patience=self.config['patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
        
        X_train_lstm = X_sequence[:X_train.shape[0]]
        X_val_lstm = X_sequence[X_train.shape[0]:]
        y_train_lstm = y[:X_train.shape[0]]
        y_val_lstm = y[X_train.shape[0]:]
        
        history = self.lstm_model.fit(
            X_train_lstm, y_train_lstm,
            validation_data=(X_val_lstm, y_val_lstm),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stopping_callback, reduce_lr],
            verbose=0
        )

        # 动态集成权重
        lgbm_pred = self.lgbm_model.predict(X_val)
        lstm_pred = self.lstm_model.predict(X_val_lstm, verbose=0).flatten()
        lgbm_mse = mean_squared_error(y_val, lgbm_pred)
        lstm_mse = mean_squared_error(y_val, lstm_pred)
        total_mse = lgbm_mse + lstm_mse
        self.ensemble_weights = {
            'lgbm': lstm_mse / total_mse if total_mse > 0 else 0.5,
            'lstm': lgbm_mse / total_mse if total_mse > 0 else 0.5
        }
        
        self.logger.info(f"动态集成权重: LGBM={self.ensemble_weights['lgbm']:.3f}, LSTM={self.ensemble_weights['lstm']:.3f}")
        
        ensemble_pred = self.ensemble_weights['lgbm'] * lgbm_pred + self.ensemble_weights['lstm'] * lstm_pred
        ensemble_mse = mean_squared_error(y_val, ensemble_pred)
        self.logger.info(f"验证集 MSE: LGBM={lgbm_mse:.4f}, LSTM={lstm_mse:.4f}, Ensemble={ensemble_mse:.4f}")

        # 返回训练指标
        return {
            'lstm_loss': history.history['loss'][-1],
            'lstm_val_loss': history.history['val_loss'][-1],
            'lgbm_rmse': lgbm_rmse,
            'ensemble_mse': ensemble_mse
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
        self.logger.info(
            f"最佳权重: LGBM {self.ensemble_weights['lgbm']:.2f}, LSTM {self.ensemble_weights['lstm']:.2f}")

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
            annual_return = (
                1 + df_eval['pred'].mean()) ** self.TRADING_DAYS - 1
            result['annual_return'] = annual_return

        return result

    def compute_shap_values(self, X_sequence: np.ndarray) -> np.ndarray:
        shap_subset_size = min(int(psutil.virtual_memory().available / (1024 ** 3) * 2000), len(X_sequence), 20000)
        subset_idx = np.random.choice(len(X_sequence), size=shap_subset_size, replace=False)
        X_subset = X_sequence[subset_idx]
        X_lgbm = X_subset.reshape(X_subset.shape[0], -1)
        
        explainer = shap.TreeExplainer(self.lgbm_model)
        shap_values_lgbm = explainer.shap_values(X_lgbm)
        
        X_lstm = X_subset
        explainer = shap.DeepExplainer(self.lstm_model, X_lstm)
        shap_values_lstm = explainer.shap_values(X_lstm, check_additivity=False)
        
        combined_shap = self.ensemble_weights['lgbm'] * shap_values_lgbm + self.ensemble_weights['lstm'] * shap_values_lstm
        self.feature_importance_cache = {
            'lgbm': shap_values_lgbm.mean(axis=0),
            'lstm': shap_values_lstm.mean(axis=0),
            'combined': combined_shap.mean(axis=0)
        }
        return combined_shap
    
    def get_top_features(self, top_n: int = 30) -> Dict[str, List[Tuple[int, float]]]:
        """获取顶部特征"""
        if not self.feature_importance_cache:
            self.logger.warning("特征重要性缓存为空")
            return {}

        importance = self.feature_importance_cache.get(
            'lgbm', np.zeros(9))  # 调整为9个特征
        sorted_idx = np.argsort(importance)[::-1][:top_n]
        return {'lgbm': [(idx, importance[idx]) for idx in sorted_idx]}

    def save_models(self, model_dir: str = "models"):
        """保存模型和配置"""
        os.makedirs(model_dir, exist_ok=True)
        if self.lgbm_model is not None:
            joblib.dump(self.lgbm_model, os.path.join(
                model_dir, "lgbm_model.pkl"))
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
            self.ensemble_weights = model_state.get(
                'ensemble_weights', self.ensemble_weights)
            self.feature_importance_cache = model_state.get(
                'feature_importance', {})

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

        recommendation_df = recommendation_df.sort_values(
            'prediction', ascending=False)
        return recommendation_df.head(top_n)

    def get_feature_importance(self):
        """
        获取特征重要性分数
        返回基于LGBM模型的特征重要性
        """
        if hasattr(self, 'lgbm_model') and self.lgbm_model is not None:
            # 获取LGBM模型的特征重要性
            importance = self.lgbm_model.feature_importances_
            
            # 检查feature_columns是否为空
            if not self.feature_columns:
                self.logger.warning("feature_columns为空，返回原始特征重要性scores")
                return importance
                
            # 重塑特征重要性数组
            try:
                n_features = len(self.feature_columns)
                if n_features == 0:
                    self.logger.error("特征列数为0，无法计算特征重要性")
                    return None
                    
                n_timesteps = len(importance) // n_features
                if n_timesteps == 0:
                    self.logger.error("时间步长为0，无法计算特征重要性")
                    return None
                    
                importance = importance.reshape(n_timesteps, n_features).mean(axis=0)
                
                # 创建特征名称和重要性的映射
                feature_importance_dict = dict(zip(self.feature_columns, importance))
                self.logger.info(f"特征重要性计算完成，特征数量: {len(feature_importance_dict)}")
                return importance
                
            except Exception as e:
                self.logger.error(f"计算特征重要性时发生错误: {str(e)}")
                return None
        else:
            self.logger.warning("LGBM模型未训练，无法获取特征重要性")
            return None