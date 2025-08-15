import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from src.data_acquisition.tushare_data import TushareDataFetcher
from src.utils.helpers import get_latest_trade_date
from src.data_preprocessing.technical_indicators import TechnicalIndicators
from src.model.hybrid_predictor import HybridPredictor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tushare_fetch_data(start_date: str, end_date: str):
    try:
        fetcher = TushareDataFetcher()
        df = fetcher.get_all_historical_data_from_db('stock_daily', start_date=start_date, end_date=end_date)
        if df.empty:
            raise ValueError(f"No data fetched for period {start_date} to {end_date}")
        # Filter for liquid stocks (average volume over last 20 days > 1000)
        df = df[df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=20).mean()) > 1000]
        logging.info(f"Fetched {len(df)} rows for {len(df['ts_code'].unique())} stocks")
        return df
    except Exception as e:
        raise ValueError(f"Error fetching data: {str(e)}")

def main():
    try:
        # 1. Fetch data (5 years of data to accommodate newer stocks)
        end_date = get_latest_trade_date()  # Format: '20250814'
        start_date = (datetime.datetime.strptime(end_date, '%Y%m%d').date() - 
                     datetime.timedelta(days=365*5)).strftime('%Y%m%d')  # 5 years
        df = tushare_fetch_data(start_date, end_date)
        
        # Debug: Log stock data details
        logging.info(f"Initial stocks: {df['ts_code'].unique()}")
        logging.info(f"Missing values in df:\n{df.isna().sum()}")
        
        # 2. Calculate technical indicators
        indicator_calculator = TechnicalIndicators()
        df_with_indicators = indicator_calculator.calculate_indicators(df, keep_nans=True, min_data_points=252)
        if df_with_indicators.empty:
            raise ValueError("No valid data after calculating indicators")
        logging.info(f"Data with indicators: {len(df_with_indicators)} rows, {len(df_with_indicators['ts_code'].unique())} stocks")
        logging.info(f"Missing values in df_with_indicators:\n{df_with_indicators.isna().sum()}")
        
        # 3. Preprocess data
        sequence_length = 20
        X_lstm, X_xgb, y, stock_codes = indicator_calculator.preprocess_data(df_with_indicators, sequence_length)
        
        # 4. Split data into train and test sets
        X_lstm_train, X_lstm_test, X_xgb_train, X_xgb_test, y_train, y_test, stock_codes_train, stock_codes_test = train_test_split(
            X_lstm, X_xgb, y, stock_codes, test_size=0.2, random_state=42
        )
        logging.info(f"Training set: {X_lstm_train.shape[0]} samples, Test set: {X_lstm_test.shape[0]} samples")
        
        # 5. Initialize and train hybrid model
        n_features = len(indicator_calculator.feature_columns)
        predictor = HybridPredictor(lstm_units=50, sequence_length=sequence_length, n_features=n_features)
        predictor.train(X_lstm_train, X_xgb_train, y_train, validation_split=0.2, epochs=1, batch_size=32)
        
        # 6. Evaluate model on test set
        y_pred = predictor.predict(X_lstm_test, X_xgb_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test Directional Accuracy: {directional_accuracy:.4f}")
        
        # 7. Predict latest data
        latest_data = df_with_indicators.groupby('ts_code').filter(
            lambda x: len(x) >= sequence_length
        ).groupby('ts_code').tail(sequence_length)
        if latest_data.empty:
            raise ValueError("No valid latest data for prediction")
        logging.info(f"Latest data: {len(latest_data)} rows, {len(latest_data['ts_code'].unique())} stocks")
        logging.info(f"Missing values in latest_data:\n{latest_data.isna().sum()}")
        
        X_lstm_latest, X_xgb_latest, _, latest_stock_codes = indicator_calculator.preprocess_data(
            latest_data, sequence_length, for_prediction=True
        )
        
        if len(X_lstm_latest) == 0:
            logging.warning("No sufficient data to generate predictions")
            return
        
        predictions = predictor.predict(X_lstm_latest, X_xgb_latest)
        
        # 8. Generate recommendations
        recommendations = pd.DataFrame({
            'ts_code': latest_stock_codes,
            'predicted_change': predictions
        })
        recommendations = recommendations.groupby('ts_code').last().reset_index()
        recommendations = recommendations.sort_values('predicted_change', ascending=False)
        
        print("T+1 Stock Recommendations (Top 10):")
        print(recommendations.head(10))
        
        # 9. Save model with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        predictor.save_model(lstm_path=f'lstm_model_{timestamp}.h5', xgb_path=f'xgb_model_{timestamp}.pkl')
    
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()