import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.log_helper import LogHelper



class HybridPredictor:
    def __init__(self, lstm_lookback=10):
        self.logger = LogHelper.get_logger(__name__)
        self.xgb_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.lstm_lookback = lstm_lookback
        self.lstm_model = None
        self.scaler = StandardScaler()
        
    def prepare_lstm_data(self, X):
        """准备LSTM所需的时序数据"""
        if len(X) < self.lstm_lookback:
            # 如果数据长度小于lookback，通过重复最后一行来补齐
            pad_rows = self.lstm_lookback - len(X)
            X = np.vstack([X, np.tile(X[-1], (pad_rows, 1))])
        
        X_lstm = []
        for i in range(len(X) - self.lstm_lookback + 1):
            X_lstm.append(X[i:(i + self.lstm_lookback)])
        return np.array(X_lstm)
    
    def build_lstm(self, input_shape):
        """构建LSTM模型 - 使用Keras函数式API"""
        inputs = Input(shape=input_shape)
        x = LSTM(50, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(50)(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def fit(self, X, y):
        """训练混合模型"""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data")
            
        # 数据清洗
        valid_mask = ~np.isnan(y) & ~np.isinf(y) & (np.abs(y) < 1.0)  # 过滤掉异常收益率
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid data after cleaning")
            
        # 标准化数据
        X_scaled = self.scaler.fit_transform(X)
        
        self.logger.info(f"Training with {len(X)} samples after cleaning")
        self.logger.info(f"Label statistics: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
        
        # 训练XGBoost
        self.xgb_model.fit(X_scaled, y)
        
        # 准备并训练LSTM
        X_lstm = self.prepare_lstm_data(X_scaled)
        y_lstm = y[self.lstm_lookback-1:]
        
        if len(X_lstm) != len(y_lstm):
            raise ValueError(f"Length mismatch: X_lstm={len(X_lstm)}, y_lstm={len(y_lstm)}")
        
        # 添加回调函数
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint('./models/best_lstm_model.keras', save_best_only=True)  # 使用.keras扩展名
        ]
        
        self.lstm_model = self.build_lstm((self.lstm_lookback, X.shape[1]))
        self.lstm_model.fit(
            X_lstm, y_lstm,
            epochs=5,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
    def predict_single(self, X):
        """单只股票预测"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_scaled = self.scaler.transform(X)
        
        # XGBoost预测
        xgb_pred = self.xgb_model.predict(X_scaled)
        
        # LSTM预测
        X_lstm = self.prepare_lstm_data(X_scaled)
        lstm_pred = self.lstm_model.predict(X_lstm, verbose=0).flatten()
        
        # 混合预测
        return 0.5 * xgb_pred[-1] + 0.5 * lstm_pred[-1]

    def predict_batch(self, X):
        """批量股票预测"""
        X_scaled = self.scaler.transform(X)
        
        # XGBoost预测
        xgb_pred = self.xgb_model.predict(X_scaled)
        
        # LSTM预测
        X_lstm = self.prepare_lstm_data(X_scaled)
        lstm_pred = self.lstm_model.predict(X_lstm, verbose=0).flatten()
        
        # 调整长度
        xgb_pred = xgb_pred[self.lstm_lookback-1:][:len(lstm_pred)]
        
        # 混合预测
        return 0.5 * xgb_pred + 0.5 * lstm_pred

    def predict(self, X):
        """预测接口"""
        if len(X) == 0:
            raise ValueError("Empty prediction data")
        
        # 根据输入数据的形状选择预测方法
        if len(X.shape) == 1 or (len(X.shape) == 2 and X.shape[0] == 1):
            return self.predict_single(X)
        else:
            return self.predict_batch(X)
        
    def get_top_stocks(self, predictions: pd.Series, n: int = 10) -> pd.Series:
        """
        获取预测收益率最高的前N只股票
        
        Args:
            predictions: 包含股票代码和预测收益率的Series
            n: 需要返回的股票数量
            
        Returns:
            pd.Series: 按预测收益率排序的前N只股票
        """
        if len(predictions) == 0:
            self.logger.warning("Empty predictions, cannot get top stocks")
            return pd.Series()
            
        # 按预测收益率降序排序并返回前N只股票
        return predictions.nlargest(n)