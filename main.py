import datetime
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from src.data_acquisition.akshare_data import AkshareDataFetcher
from src.utils.helpers import get_latest_trade_date
from src.data_preprocessing.technical_indicators import TechnicalIndicators
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from src.model.lstm_lgbm_predictor import LSTMLGBMPredictor
from src.utils.log_helper import LogHelper
import gc

# 配置日志
logger = LogHelper().get_logger(__name__)

def main():
    """
    主函数：完整的股票预测模型训练流程
    """
    try:
        # ====================== 1. 获取数据 ======================
        logger.info("====== 开始数据获取 ======")
        end_date = get_latest_trade_date()
        start_date = (datetime.datetime.strptime(end_date, '%Y%m%d').date() - datetime.timedelta(days=365*10)).strftime('%Y%m%d')
        logger.info("正常模式：使用10年数据")
        
        fetcher = AkshareDataFetcher()
        df = fetcher.get_all_historical_data_from_db('stock_daily', start_date=start_date, end_date=end_date)
        
        if df.empty:
            logger.error("未能获取到任何数据")
            return
        
        logger.info(f"获取数据完成: {df['trade_date'].min()}至{df['trade_date'].max()}, 共{len(df)}条记录, {df['ts_code'].nunique()}支股票")
        
        # ====================== 2. 计算技术指标和信号 ======================
        logger.info("====== 开始计算技术指标 ======")
        ti = TechnicalIndicators(df)
        df_with_indicators = ti.calculate_all_indicators()
        
        if df_with_indicators.empty:
            logger.error("技术指标计算失败")
            return
            
        original_cols = len(df.columns)
        new_cols = len(df_with_indicators.columns) - original_cols
        logger.info(f"计算技术指标完成: 新增{new_cols}个特征，总特征数={len(df_with_indicators.columns)}")
        logger.info(f"技术指标列表示例: {df_with_indicators.columns.tolist()[:10]}...")
        
        gc.collect()
        
        # ====================== 3. 数据预处理 ======================
        logger.info("====== 开始数据预处理 ======")
        preprocessor = DataPreprocessor(
            lookback_window=10,  # Reduced from 20
            prediction_horizon=1,
            feature_scaler_type='standard',
            target_scaler_type='standard',
            test_size=0.2,
            imputation_strategy='hybrid',
            outlier_threshold=5.0,
            shap_subset_size=10000
        )
        
        # 数据质量检查
        logger.info("数据质量检查...")
        null_counts = df_with_indicators.isnull().sum().sum()
        if null_counts > 0:
            logger.warning(f"发现{null_counts}个空值，将在预处理中处理")
        
        # 初始化模型
        logger.info("初始化LSTM+LightGBM模型...")
        model = LSTMLGBMPredictor()
        
        # 一站式数据准备和模型拟合，包含子集SHAP特征选择
        logger.info("====== 数据预处理和SHAP特征选择 ======")
        X_sequence_scaled, y_scaled, stock_codes, dates = preprocessor.fit(df_with_indicators, model=model)
        
        if len(X_sequence_scaled) == 0:
            logger.error("没有有效的训练数据，请检查数据质量和参数设置")
            return
            
        logger.info(f"数据预处理完成: 获得{len(X_sequence_scaled)}个样本，{len(np.unique(stock_codes))}支股票")
        logger.info(f"时序数据维度: X_sequence={X_sequence_scaled.shape}, y={y_scaled.shape}")
        
        # ====================== 4. 创建训练测试集 ======================
        logger.info("====== 创建训练测试集 ======")
        X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
            X_sequence_scaled, y_scaled, stock_codes=stock_codes, split_method='time_series'
        )
        
        if len(X_train) < 10:
            logger.error("训练数据不足，请增加数据量或减少测试比例")
            return
            
        if len(X_test) == 0:
            logger.error("测试数据为空，请检查数据质量和参数设置")
            return

        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {len(X_train)}个样本 ({len(X_train)/len(X_sequence_scaled)*100:.1f}%)")
        logger.info(f"  测试集: {len(X_test)}个样本 ({len(X_test)/len(X_sequence_scaled)*100:.1f}%)")

        test_stock_codes = stock_codes[len(X_train):]
        if len(test_stock_codes) != len(X_test):
            logger.error(f"测试集股票代码长度{len(test_stock_codes)}与X_test长度{len(X_test)}不匹配")
            return

        # ====================== 5. 训练LSTM+LightGBM混合模型 ======================
        logger.info("====== 开始训练LSTM+LightGBM混合模型 ======")
        training_results = model.fit(
            X_sequence=X_train, 
            y=y_train
        )
        
        logger.info("模型训练完成")
        logger.info(f"训练结果: {training_results}")

        # ====================== 6. 评估模型 ======================
        logger.info("====== 评估模型性能 ======")
        evaluation_results = model.evaluate(X_test, y_test, dates=dates[len(X_train):])
        
        logger.info("模型评估结果:")
        for metric, value in evaluation_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        # ====================== 7. 生成T+1股票推荐 ======================
        logger.info("====== 生成T+1股票推荐 ======")
        recommendations_df = model.get_recommendations(X_test, stock_codes=test_stock_codes, top_n=10)
        
        latest_date = max(dates)
        next_trading_day = (datetime.datetime.strptime(latest_date, '%Y%m%d') + datetime.timedelta(days=1)).strftime('%Y%m%d')
        
        logger.info(f"\n=== T+1 股票推荐 (交易日期: {next_trading_day}) ===")
        logger.info("前10增长股票:")
        for rank, (_, row) in enumerate(recommendations_df.iterrows(), 1):
            logger.info(f"{rank}. {row['ts_code']}: 预测收益率 {row['prediction']:.4f}")

        # ====================== 8. 保存预处理器和模型 ======================
        logger.info("====== 保存模型和预处理器 ======")
        os.makedirs('models', exist_ok=True)
        
        try:
            preprocessor.save_preprocessor('models/data_preprocessor.joblib')
            model.save_models('models/lstm_lgbm_model')
            logger.info("模型和预处理器保存成功")
        except Exception as e:
            logger.error(f"保存模型时出错: {str(e)}")
        
        # ====================== 9. 测试模式：交易策略模拟 ======================
        logger.info("====== 测试模式：生成交易建议 ======")
        from src.strategy.trading_strategy import TradingStrategy
        strategy = TradingStrategy()
        
        current_market_data = {
            code: df_with_indicators[df_with_indicators['ts_code'] == code]
            for code in recommendations_df['ts_code'].unique()
        }
        
        portfolio = {
            row['ts_code']: (df[df['ts_code'] == row['ts_code']]['close'].iloc[-2], 100) 
            for _, row in recommendations_df.iterrows()
        }
        
        sell_recommendations = strategy.generate_sell_recommendations(
            portfolio=portfolio,
            market_data=current_market_data,
            market_trend='neutral'
        )
        
        logger.info("\n=== T+1卖出建议 ===")
        if sell_recommendations:
            for i, rec in enumerate(sell_recommendations, 1):
                logger.info(f"{i}. {rec['stock_code']}: 当前价{rec['current_price']:.2f}, 收益率{rec['profit_rate']:.2%}, 理由: {rec['reason']}")
        else:
            logger.info("暂无卖出建议")

        logger.info(f"\n=== 测试模式总结 ===")
        logger.info(f"使用的股票: {list(np.unique(stock_codes))[:3]}...")
        logger.info(f"数据时间范围: {min(dates)} 到 {max(dates)}")
        logger.info(f"最终样本数: {len(X_sequence_scaled)}")
        logger.info("测试运行成功完成！")

    except Exception as e:
        logger.error(f"系统执行失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()