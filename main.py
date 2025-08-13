

from src.data_acquisition.akshare_data import AkshareDataFetcher
from src.data_acquisition.tushare_data import TushareDataFetcher
from src.data_preprocessing.technical_indicators import TechnicalIndicators
from src.model.gru_model import GRUTrainer



def akshare_test():
    fetcher = AkshareDataFetcher()
    # df = fetcher.get_all_stock_codes()
    df = fetcher.get_stock_codes_by_symbol()
    fetcher.batch_fetch_historical_data(df.head(2), '20250101', '20250808')
    # fetcher.get_lhb_data(start_date='20250811', end_date='20250812')
    df = fetcher.get_all_historical_data_from_db('stock_daily')
    # 基础使用
    ti = TechnicalIndicators(df)
    # ML训练使用
    ml_data = ti.prepare_ml_dataset(target_threshold=0.01)
    X, y, full_data = ti.get_feature_importance_candidates()


    # fetcher.get_latest_trade_date('stock_daily', '600006.SH')


def tushare_test():
    fetcher = TushareDataFetcher()
    # df = fetcher.get_all_stock_codes()
    # df = fetcher.get_stock_codes_by_symbol()
    # fetcher.batch_fetch_historical_data(df, '20250101', '20250810')
    df = fetcher.get_all_historical_data_from_db('stock_daily')

    ti = TechnicalIndicators(df)
    X_tensor, y_tensor, feature_names = ti.prepare_gru_dataset(seq_len=10, target_threshold=0.01, include_enhanced=True)
    
    # 3. 初始化训练器
    trainer = GRUTrainer(input_size=len(feature_names), seq_len=10, epochs=20, lr=0.001, batch_size=64)
    
    # 4. 训练模型
    trainer.train(X_tensor, y_tensor)
    
    # 5. 保存模型
    trainer.save()
    
    # 6. 加载并预测最新窗口
    trainer.load()
    latest_seq = X_tensor[-1].unsqueeze(0)  # 取最新的一个窗口
    prob = trainer.predict(latest_seq)[0][0]
    print(f"最新预测上涨概率: {prob:.4f}")


    # fetcher.get_lhb_data(start_date='20250811', end_date='20250812')
    # fetcher.get_latest_trade_date('stock_daily', '600006.SH')


if __name__ == '__main__':
    # akshare_test()
    tushare_test()
    # akshare_fetcher = AkshareDataFetcher()
    # tushare_fetcher = TushareDataFetcher()
    # df = tushare_fetcher.get_all_stock_codes()
    # df = df.head(2)
    # ak_historical_data = akshare_fetcher.batch_fetch_historical_data(df, '20250811', '20250812')
    # tushare_historical_data = tushare_fetcher.batch_fetch_historical_data(df, '20250811', '20250812')
    # print('ak_historical_data')
    # print(ak_historical_data)
    # print('tushare_historical_data')
    # print(tushare_historical_data)






