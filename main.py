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
        return df
    except Exception as e:
        raise ValueError(f"Error fetching data: {str(e)}")

def main():
    try:
        # 1. 获取数据
        end_date = get_latest_trade_date()
        start_date = (datetime.datetime.strptime(end_date, '%Y%m%d').date() - 
                     datetime.timedelta(days=365*5)).strftime('%Y%m%d')
        df = tushare_fetch_data(start_date, end_date)
        # Add after fetching data:
        logging.info(f"Unique stocks count: {df['ts_code'].nunique()}")
        logging.info(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
        logging.info(f"Missing values in raw data:\n{df.isnull().sum()}")
        
        # Add logging for data shape
        logging.info(f"Initial data shape: {df.shape}")
        
        # 2. 计算技术指标
        ti = TechnicalIndicators(df)
        df_with_indicators = ti.calculate_all_indicators()
        logging.info(f"Data with indicators shape: {df_with_indicators.shape}")
        
        # 3. 准备训练数据
        X, y = ti.prepare_features()
        logging.info(f"Features shape: X={X.shape}, y={y.shape}")
        
        # 数据验证
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty feature data")
            
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logging.warning("Found invalid values in features before cleaning")
            
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            logging.warning("Found invalid values in labels before cleaning")
        
        # 分割训练集和测试集前先清理数据
        valid_mask = (~np.isnan(y) & ~np.isinf(y) & (np.abs(y) < 1.0) & 
                     ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        logging.info(f"Data shape after cleaning: X={X.shape}, y={y.shape}")
        
        if len(X) == 0:
            raise ValueError("No valid data after cleaning")
        
        # 4. 分割训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        logging.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        logging.info(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")
        
        # 5. 训练混合模型
        predictor = HybridPredictor()
        logging.info("Starting model training...")
        predictor.fit(X_train, y_train)
        logging.info("Model training completed")
        
        # 6. 对最新数据进行预测
        latest_data = df_with_indicators.groupby('ts_code').tail(1).reset_index(drop=True)
        logging.info(f"Latest data shape: {latest_data.shape}")
        
        # 7. 获取推荐股票
        predictions = {}
        valid_predictions = 0
        total_stocks = len(latest_data)
        
        for idx, row in latest_data.iterrows():
            code = row['ts_code']
            try:
                stock_features = ti.prepare_single_stock_features(row)
                if stock_features is not None:
                    pred = predictor.predict(stock_features)
                    if not np.isnan(pred) and not np.isinf(pred):
                        predictions[code] = float(pred)
                        valid_predictions += 1
                        
                if (idx + 1) % 100 == 0:
                    logging.info(f"Processed {idx + 1}/{total_stocks} stocks")
                    
            except Exception as e:
                logging.warning(f"Failed to predict for stock {code}: {str(e)}")
                continue

        logging.info(f"Successfully predicted {valid_predictions}/{total_stocks} stocks")
        
        

        # 转换预测结果为Series
        predictions_series = pd.Series(predictions)
        if len(predictions_series) == 0:
            logging.warning("No valid predictions generated")
        else:
            logging.info(f"Number of valid predictions: {len(predictions_series)}")
        
        # 获取推荐股票
        top_stocks = predictor.get_top_stocks(predictions_series, n=10)
        
        # 8. 输出推荐结果
        logging.info("\nTop 10 Recommended Stocks for Tomorrow:")
        for code, pred_return in top_stocks.items():
            logging.info(f"Stock: {code}, Predicted Return: {pred_return:.2%}")
        
        # 9. 简单回测（使用测试集）
        test_pred = predictor.predict(X_test)
        # 由于LSTM预测会损失前lstm_lookback个数据点，需要相应调整y_test
        y_test_adjusted = y_test[predictor.lstm_lookback-1:][:len(test_pred)]
        mse = mean_squared_error(y_test_adjusted, test_pred)
        mae = mean_absolute_error(y_test_adjusted, test_pred)
        logging.info(f"\nBacktest Results:")
        logging.info(f"Mean Squared Error: {mse:.6f}")
        logging.info(f"Mean Absolute Error: {mae:.6f}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise  # 添加这行以显示完整的错误堆栈

if __name__ == "__main__":
    main()