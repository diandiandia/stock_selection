import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import talib
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TechnicalIndicators:
    def __init__(self):
        self.feature_columns = [
            'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude',
            'change', 'change_amount', 'turnover', 'SMA5', 'SMA10', 'SMA20',
            'EMA12', 'EMA26', 'RSI14', 'MACD', 'MACD_Signal', 'BB_Upper',
            'BB_Middle', 'BB_Lower', 'K', 'D', 'VWAP'
        ]
        self.scaler = StandardScaler()

    def calculate_indicators(self, df, keep_nans=False, min_data_points=252):
        df = df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])  # Ensure trade_date is datetime
        grouped = df.groupby('ts_code')
        excluded_stocks = []
        
        # Validate date continuity and data sufficiency
        valid_stocks = []
        for ts_code, group in grouped:
            group = group.sort_values('trade_date')
            if len(group) < min_data_points:
                logging.warning(f"Excluding {ts_code}: insufficient data ({len(group)} rows, minimum {min_data_points})")
                excluded_stocks.append((ts_code, "insufficient_data", len(group)))
                continue
            date_diff = group['trade_date'].diff().dt.days.fillna(1)
            if date_diff.max() > 5:
                logging.warning(f"Excluding {ts_code}: large gaps in trading days (max gap: {date_diff.max()} days)")
                excluded_stocks.append((ts_code, "large_gaps", date_diff.max()))
                continue
            valid_stocks.append(ts_code)
        
        if excluded_stocks:
            pd.DataFrame(excluded_stocks, columns=['ts_code', 'reason', 'value']).to_csv('excluded_stocks.csv', index=False)
        
        df = df[df['ts_code'].isin(valid_stocks)]
        if df.empty:
            raise ValueError("No stocks with sufficient continuous data after filtering")
        
        grouped = df.groupby('ts_code')
        for ts_code, group in grouped:
            group = group.sort_values('trade_date')
            group['SMA5'] = talib.SMA(group['close'], timeperiod=5)
            group['SMA10'] = talib.SMA(group['close'], timeperiod=10)
            group['SMA20'] = talib.SMA(group['close'], timeperiod=20)
            group['EMA12'] = talib.EMA(group['close'], timeperiod=12)
            group['EMA26'] = talib.EMA(group['close'], timeperiod=26)
            group['RSI14'] = talib.RSI(group['close'], timeperiod=14)
            macd, signal, _ = talib.MACD(group['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            group['MACD'] = macd
            group['MACD_Signal'] = signal
            group['BB_Upper'], group['BB_Middle'], group['BB_Lower'] = talib.BBANDS(group['close'], timeperiod=20)
            group['K'], group['D'] = talib.STOCH(group['high'], group['low'], group['close'], 
                                                 fastk_period=14, slowk_period=3, slowd_period=3)
            group['VWAP'] = (group['amount'] / group['volume']).where(group['volume'] != 0, np.nan)
            df.loc[group.index, self.feature_columns[10:]] = group[self.feature_columns[10:]]
        
        df['y'] = df.groupby('ts_code')['change'].shift(-1)
        if not keep_nans:
            df = df.dropna(subset=['y'] + [col for col in self.feature_columns if col != 'change_amount'])
        return df

    def preprocess_data(self, data, sequence_length, for_prediction=False):
        if data.empty:
            raise ValueError("Input data is empty. Cannot preprocess.")
        
        X_lstm, X_xgb, y, stock_codes = [], [], [], []
        
        for ts_code in data['ts_code'].unique():
            stock_data = data[data['ts_code'] == ts_code].sort_values('trade_date')
            if len(stock_data) < sequence_length + (1 if not for_prediction else 0):
                logging.info(f"Skipping {ts_code}: insufficient data ({len(stock_data)} rows)")
                continue
            features = stock_data[self.feature_columns].values
            # Log NaN counts for debugging
            nan_counts = np.isnan(features).sum(axis=0)
            logging.info(f"{ts_code}: NaN counts in features: {dict(zip(self.feature_columns, nan_counts))}")
            if not for_prediction:
                features = self.scaler.fit_transform(features)
                joblib.dump(self.scaler, 'scaler.pkl')  # Use a single scaler file
            else:
                try:
                    self.scaler = joblib.load('scaler.pkl')
                    features = self.scaler.transform(features)
                except FileNotFoundError:
                    logging.warning(f"Scaler file not found, fitting new scaler for {ts_code}")
                    features = self.scaler.fit_transform(features)  # Fallback
            target = stock_data['y'].values if not for_prediction else None
            
            valid_sequences = 0
            for i in range(len(stock_data) - sequence_length):
                sequence = features[i:i + sequence_length]
                if np.any(np.isnan(sequence)) or np.any(np.isnan(features[i + sequence_length - 1])):
                    continue
                X_lstm.append(sequence)
                X_xgb.append(features[i + sequence_length - 1])
                if not for_prediction and i + sequence_length < len(target):
                    y.append(target[i + sequence_length])
                stock_codes.append(ts_code)
                valid_sequences += 1
            logging.info(f"{ts_code}: {valid_sequences} valid sequences generated")
        
        if not X_lstm:
            raise ValueError("No valid sequences generated. Check data sufficiency.")
        
        X_lstm = np.array(X_lstm)
        X_xgb = np.array(X_xgb)
        y = np.array(y) if not for_prediction else None
        logging.info(f"X_lstm shape: {X_lstm.shape}, X_xgb shape: {X_xgb.shape}, y shape: {y.shape if y is not None else 'None'}")
        return X_lstm, X_xgb, y, stock_codes