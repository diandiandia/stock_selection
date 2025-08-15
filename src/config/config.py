import os

class Config:
    """
    股票推荐系统配置
    """
    
    # 数据配置
    DATA_DIR = "data"
    MODEL_DIR = "models"
    LOG_DIR = "logs"
    
    # 股票池配置
    STOCK_UNIVERSE = '000906'  # 中证800
    MIN_MARKET_CAP = 50e8  # 最小市值（元）
    EXCLUDE_ST = True  # 排除ST股票
    EXCLUDE_GEM = True  # 排除创业板
    
    # 特征配置
    TECHNICAL_FEATURES = [
        'ma5', 'ma10', 'ma20', 'rsi', 'kdj_diff', 'macd', 'macd_hist',
        'volume_ratio', 'turnover_rate', 'atr', 'gap', 'intraday_range',
        'bb_position', 'cci', 'momentum_5', 'roc_5', 'willr'
    ]
    
    # 模型配置
    PREDICTION_THRESHOLD = 0.01  # 预测涨幅阈值
    MIN_PROBABILITY = 0.7  # 最小推荐概率
    TOP_K_STOCKS = 10  # 每日推荐股票数量
    
    # 训练配置
    TRAIN_START_DATE = '2022-01-01'
    TRAIN_END_DATE = '2024-01-01'
    TEST_START_DATE = '2024-01-02'
    TEST_END_DATE = '2024-12-01'
    
    # 模型参数
    ENSEMBLE_PARAMS = {
        'xgb': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'rf': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5
        },
        'gb': {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 5
        }
    }
    
    LSTM_PARAMS = {
        'seq_length': 20,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'epochs': 30,
        'batch_size': 64,
        'learning_rate': 0.001
    }
    
    # 风险控制
    MAX_POSITION_SIZE = 0.1  # 单只股票最大仓位
    STOP_LOSS = -0.05  # 止损比例
    TAKE_PROFIT = 0.05  # 止盈比例
    
    # 回测配置
    INITIAL_CAPITAL = 100000  # 初始资金
    TRANSACTION_COST = 0.001  # 交易成本
    SLIPPAGE = 0.002  # 滑点
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)

class ModelPaths:
    """
    模型文件路径
    """
    ENSEMBLE_MODEL = "models/ensemble_model.pkl"
    LSTM_MODEL = "models/lstm_model.pth"
    SCALER = "models/scaler.pkl"
    FEATURE_IMPORTANCE = "models/feature_importance.csv"

class DataPaths:
    """
    数据文件路径
    """
    TRAIN_DATA = "data/train_data.csv"
    TEST_DATA = "data/test_data.csv"
    RECOMMENDATIONS = "data/recommendations_{date}.csv"
    BACKTEST_RESULTS = "data/backtest_results.csv"
    STOCK_LIST = "data/stock_list.csv"