import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridPredictor:
    def __init__(self, lstm_units=50, sequence_length=20, n_features=None, xgb_params=None):
        if n_features is None:
            raise ValueError("n_features must be specified for LSTM input shape")
        self.lstm_units = lstm_units
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.xgb_params = xgb_params if xgb_params else {
            'objective': 'reg:squarederror', 'n_estimators': 100, 'max_depth': 5
        }
        self.lstm_model = self.build_lstm_model()
        self.xgb_model = XGBRegressor(**self.xgb_params)
    
    def build_lstm_model(self):
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        model.add(LSTM(units=self.lstm_units, return_sequences=True, kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LSTM(units=self.lstm_units // 2, kernel_regularizer='l2'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_lstm, X_xgb, y, validation_split=0.2, epochs=50, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.lstm_model.fit(X_lstm, y, validation_split=validation_split, epochs=epochs,
                           batch_size=batch_size, verbose=1, callbacks=[early_stopping])
        
        lstm_pred = self.lstm_model.predict(X_lstm, verbose=0)
        X_xgb_combined = np.hstack([X_xgb, lstm_pred])
        self.xgb_model.fit(X_xgb_combined, y)
    
    def predict(self, X_lstm, X_xgb):
        lstm_pred = self.lstm_model.predict(X_lstm, verbose=0)
        X_xgb_combined = np.hstack([X_xgb, lstm_pred])
        final_pred = self.xgb_model.predict(X_xgb_combined)
        return final_pred
    
    def save_model(self, lstm_path='lstm_model.h5', xgb_path='xgb_model.pkl'):
        self.lstm_model.save(lstm_path)
        joblib.dump(self.xgb_model, xgb_path)
    
    def load_model(self, lstm_path='lstm_model.h5', xgb_path='xgb_model.pkl'):
        self.lstm_model = load_model(lstm_path)
        self.xgb_model = joblib.load(xgb_path)