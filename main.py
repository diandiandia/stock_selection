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
        logger.info("正常模式：使用{} 到 {}年数据".format(start_date, end_date))


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

        logger.info("====== 开始数据质量检查 ======")
        preprocessor = DataPreprocessor(
            lookback_window=20,
            prediction_horizon=5,
            shap_subset_size=5000,
            max_workers=4,
            outlier_threshold=5.0,
            vif_threshold=10.0,
            corr_threshold=0.95,
            feature_selection='combined',
            feature_scaler_type='standard',
            target_scaler_type='standard',
            imputation_strategy='hybrid',
            outlier_detection='zscore'  # Add this parameter
        )

        quality_report = preprocessor.check_data_quality(df)
        logger.info(f"数据质量报告:\n"
                    f"缺失值统计: {quality_report['missing_values']}\n"
                    f"重复记录数: {quality_report['duplicates']}\n"
                    f"股票覆盖数: {quality_report['stock_coverage']}\n"
                    f"交易日期范围: {quality_report['date_range']}\n"
                    f"价格异常数: {quality_report['price_anomalies']}\n"
                    f"成交量异常数: {quality_report['volume_anomalies']}")

        if quality_report['duplicates'] > 0 or quality_report['price_anomalies'] > 0:
            logger.warning("数据质量存在问题，请检查")

        # 添加计算技术指标的步骤
        logger.info("====== 开始计算技术指标 ======")
        technical_indicator = TechnicalIndicators(df)
        df_with_indicators = technical_indicator.calculate_all_indicators()
        
        # 检查技术指标计算结果
        nan_cols = df_with_indicators.isna().sum()
        if nan_cols.any():
            logger.warning("技术指标中存在NaN值:")
            for col, nan_count in nan_cols[nan_cols > 0].items():
                logger.warning(f"{col}: {nan_count}个NaN值")
        else:
            logger.info("技术指标计算完成，无NaN值")

        # 保存中间结果
        gc.collect()  # 清理内存

        logger.info("====== 初始化LSTM+LightGBM模型 ======")
        try:
            model = LSTMLGBMPredictor()
            logger.info("模型初始化成功")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise

        logger.info("====== 数据预处理和SHAP特征选择 ======")
        X_sequence_scaled, y_scaled, stock_codes, dates = preprocessor.fit(
            df_with_indicators, model=model)

        logger.info(
            f"数据预处理完成: 获得{len(X_sequence_scaled)}个样本，{len(np.unique(stock_codes))}支股票")

        # ====================== 4. 模型训练 ======================
        logger.info("====== 开始模型训练 ======")
        metrics = model.fit(
            X_sequence_scaled, 
            y_scaled,
            feature_columns=preprocessor.feature_columns,
            validation_split=0.2
        )

        logger.info(
            f"训练完成: LSTM Loss={metrics['lstm_loss']:.4f}, "
            f"LSTM Val Loss={metrics['lstm_val_loss']:.4f}, "
            f"LSTM Direction Acc={metrics['lstm_direction_acc']:.4f}, "
            f"LGBM MSE={metrics['lgbm_mse']:.4f}, "
            f"集成权重: LSTM={metrics['ensemble_weights']['lstm']:.3f}, "
            f"LGBM={metrics['ensemble_weights']['lgbm']:.3f}"
        )

        # ====================== 5. 模型评估 ======================
        logger.info("====== 开始模型评估 ======")
        eval_metrics = model.evaluate(X_sequence_scaled, y_scaled, dates=dates)

        # 记录详细的评估指标
        logger.info("\n=== 预测准确性指标 ===")
        logger.info(f"MSE: {eval_metrics['mse']:.4f}")
        logger.info(f"RMSE: {eval_metrics['rmse']:.4f}")
        logger.info(f"MAE: {eval_metrics['mae']:.4f}")
        logger.info(f"MAPE: {eval_metrics['mape']:.2f}%")
        logger.info(f"R²: {eval_metrics['r2']:.4f}")

        logger.info("\n=== 方向准确性指标 ===")
        logger.info(f"方向准确率: {eval_metrics['direction_accuracy']:.2%}")
        logger.info(f"精确率: {eval_metrics['precision']:.2%}")
        logger.info(f"召回率: {eval_metrics['recall']:.2%}")
        logger.info(f"F1分数: {eval_metrics['f1_score']:.2%}")

        logger.info("\n=== 收益相关指标 ===")
        logger.info(f"年化收益率: {eval_metrics['annual_return']:.2%}")
        logger.info(f"最大回撤: {eval_metrics['max_drawdown']:.2%}")
        logger.info(f"Calmar比率: {eval_metrics['calmar_ratio']:.2f}")

        logger.info("\n=== 风险调整指标 ===")
        logger.info(f"年化波动率: {eval_metrics['annual_volatility']:.2%}")
        logger.info(f"夏普比率: {eval_metrics['sharpe_ratio']:.2f}")
        logger.info(f"索提诺比率: {eval_metrics['sortino_ratio']:.2f}")
        logger.info(f"信息比率: {eval_metrics['information_ratio']:.2f}")

        # 保存评估结果
        eval_results = pd.DataFrame([eval_metrics])
        eval_results.to_csv('outputs/model_evaluation.csv', index=False)
        logger.info("\n评估结果已保存到 outputs/model_evaluation.csv")

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

        # ====================== 10. 风险管理 ======================
        logger.info("====== 应用风险管理 ======")
        from src.risk_management.risk_manager import RiskConfig
        from src.strategy.risk_strategy import RiskStrategy

        risk_config = RiskConfig(
            max_position_size=0.1,
            max_sector_exposure=0.3,
            stop_loss=0.05,
            take_profit=0.15,
            max_volatility=0.3,
            max_drawdown=0.2,
            min_liquidity=1e6
        )

        risk_strategy = RiskStrategy(risk_config)

        # 应用风险管理
        risk_adjusted_recommendations = risk_strategy.apply_position_limits(
            recommendations_df, 
            df_with_indicators
        )

        logger.info(f"风险调整后推荐数量: {len(risk_adjusted_recommendations)}")
        logger.info(f"平均建议仓位: {risk_adjusted_recommendations['position_size'].mean():.2%}")

        # 生成止盈止损建议
        stop_orders = risk_strategy.generate_stop_orders(portfolio, df_with_indicators)

        if stop_orders:
            logger.info("\n=== 止盈止损建议 ===")
            for order in stop_orders:
                logger.info(
                    f"{order['stock']}: {order['action'].upper()}, "
                    f"原因: {order['reason']}, "
                    f"收益率: {order['profit_rate']:.2%}"
                )

        # 保存风险调整后的推荐结果
        risk_adjusted_recommendations.to_csv(
            'outputs/risk_adjusted_recommendations.csv', 
            index=False
        )
        logger.info("风险调整后的推荐结果已保存到 outputs/risk_adjusted_recommendations.csv")

    except Exception as e:
        logger.error(f"系统执行失败: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
