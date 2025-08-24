import gc
from src.utils.log_helper import LogHelper
from src.model.lstm_lgbm_predictor import LSTMLGBMPredictor
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from src.data_preprocessing.technical_indicators import TechnicalIndicators
from src.utils.helpers import get_latest_trade_date
from src.data_acquisition.akshare_data import AkshareDataFetcher
import numpy as np
import pandas as pd
import datetime
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
        start_date = (datetime.datetime.strptime(end_date, '%Y%m%d').date(
        ) - datetime.timedelta(days=365*10)).strftime('%Y%m%d')
        logger.info("正常模式：使用10年数据")

        fetcher = AkshareDataFetcher()
        df = fetcher.get_all_historical_data_from_db(
            'stock_daily', start_date=start_date, end_date=end_date)

        if df.empty:
            logger.error("未能获取到任何数据")
            return

        logger.info(
            f"获取数据完成: {df['trade_date'].min()}至{df['trade_date'].max()}, 共{len(df)}条记录, {df['ts_code'].nunique()}支股票")

        # 过滤数据不足的股票
        min_data_points = 20  # 与technical_indicators.py一致
        stock_counts = df.groupby('ts_code').size()
        valid_stocks = stock_counts[stock_counts >= min_data_points].index
        df = df[df['ts_code'].isin(valid_stocks)]
        logger.info(f"过滤后剩余股票: {len(valid_stocks)}")

        # ====================== 2. 计算技术指标和信号 ======================
        logger.info("====== 开始计算技术指标 ======")
        ti = TechnicalIndicators(df)
        df_with_indicators = ti.calculate_all_indicators(chunksize=50)

        if df_with_indicators.empty:
            logger.error("技术指标计算失败")
            return

        # 检查NaN值
        logger.info("检查技术指标中的NaN值...")
        nan_counts = df_with_indicators.isnull().sum()
        logger.info(f"NaN统计: {nan_counts[nan_counts > 0].to_dict()}")

        # ====================== 3. 数据预处理和特征选择 ======================
        logger.info("====== 开始数据预处理 ======")
        logger.info("数据质量检查...")

        preprocessor = DataPreprocessor(
            lookback_window=10,  # 与lstm_lgbm_predictor.py一致
            prediction_horizon=1,
            shap_subset_size=3000,  # 减小以适应1.14 GB内存
            max_workers=2,  # 减小以降低内存使用
            outlier_threshold=3.0  # 更严格的异常值处理
        )

        logger.info("初始化LSTM+LightGBM模型...")
        model = LSTMLGBMPredictor()

        logger.info("====== 数据预处理和SHAP特征选择 ======")
        X_sequence_scaled, y_scaled, stock_codes, dates = preprocessor.fit(
            df_with_indicators, model=model)

        logger.info(
            f"数据预处理完成: 获得{len(X_sequence_scaled)}个样本，{len(np.unique(stock_codes))}支股票")

        # ====================== 4. 模型训练 ======================
        logger.info("====== 开始模型训练 ======")
        metrics = model.fit(X_sequence_scaled, y_scaled)
        logger.info(
            f"训练完成: LSTM Loss={metrics['lstm_loss']:.4f}, LSTM Val Loss={metrics['lstm_val_loss']:.4f}, LGBM RMSE={metrics['lgbm_rmse']:.4f}")

        # ====================== 5. 模型评估 ======================
        logger.info("====== 开始模型评估 ======")
        eval_metrics = model.evaluate(X_sequence_scaled, y_scaled, dates=dates)
        logger.info(
            f"模型评估结果: MSE={eval_metrics['mse']:.4f}, MAE={eval_metrics['mae']:.4f}, Annual Return={eval_metrics.get('annual_return', np.nan):.4f}")

        # ====================== 6. 生成推荐 ======================
        logger.info("====== 生成股票推荐 ======")
        recommendations_df = model.get_recommendations(
            X_sequence_scaled, stock_codes=stock_codes, top_n=50)
        logger.info(f"生成推荐: 推荐{len(recommendations_df)}支股票")
        logger.info(
            f"推荐股票示例: {recommendations_df['ts_code'].iloc[:3].tolist()}...")

        # ====================== 7. 保存推荐结果 ======================
        logger.info("====== 保存推荐结果 ======")
        os.makedirs('outputs', exist_ok=True)
        recommendations_df.to_csv('outputs/recommendations.csv', index=False)
        logger.info("推荐结果已保存到 outputs/recommendations.csv")

        # ====================== 8. 保存模型和预处理器 ======================
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
            row['ts_code']: (df[df['ts_code'] == row['ts_code']]
                             ['close'].iloc[-2], 100)
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
                logger.info(
                    f"{i}. {rec['stock_code']}: 当前价{rec['current_price']:.2f}, 收益率{rec['profit_rate']:.2%}, 理由: {rec['reason']}")
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
