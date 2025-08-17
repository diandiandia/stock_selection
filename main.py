
import logging
import datetime
import os
import pandas as pd
import sys
from src.data_acquisition.tushare_data import TushareDataFetcher
from src.utils.helpers import get_latest_trade_date
from src.data_preprocessing.technical_indicators import TechnicalIndicators
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from src.model.hybrid_predictor import HybridPredictor
from src.utils.log_helper import LogHelper

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
        
        # 正常模式：使用3年数据
        start_date = (datetime.datetime.strptime(end_date, '%Y%m%d').date() - datetime.timedelta(days=365*3)).strftime('%Y%m%d')
        logger.info("正常模式：使用3年数据")
        
        fetcher = TushareDataFetcher()
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
        logger.info(f"技术指标列表示例: {df_with_indicators.columns.tolist()[-10:]}")  # 显示最后10个列名
        
        # ====================== 3. 数据预处理 ======================   
        logger.info("====== 开始数据预处理 ======")
        preprocessor = DataPreprocessor(
            lookback_window=20,
            prediction_horizon=1,
            feature_scaler_type='standard',
            target_scaler_type='standard',
            test_size=0.2
        )
        
        # 数据质量检查
        logger.info("数据质量检查...")
        total_rows = len(df_with_indicators)
        null_counts = df_with_indicators.isnull().sum().sum()
        if null_counts > 0:
            logger.warning(f"发现{null_counts}个空值，将在预处理中处理")
        
        # 准备所有股票的训练数据
        X_sequence, y, stock_codes, dates = preprocessor.prepare_all_stocks(df_with_indicators)
        
        if len(X_sequence) == 0:
            logger.error("没有有效的训练数据，请检查数据质量和参数设置")
            return
            
        logger.info(f"数据预处理完成: 获得{len(X_sequence)}个样本，{len(stock_codes)}支股票")
        logger.info(f"时序数据维度: X_sequence={X_sequence.shape}, y={y.shape}")

        # ====================== 4. 拟合缩放器 ======================
        logger.info("====== 拟合数据缩放器 ======")
        preprocessor.fit_scalers(X_sequence, y)
        logger.info("数据缩放器拟合完成")

        # ====================== 5. 转换数据 ======================
        logger.info("====== 转换数据 ======")
        X_sequence_scaled = preprocessor.transform_features(X_sequence)
        y_scaled = preprocessor.transform_target(y)
        logger.info(f"数据转换完成: X_scaled={X_sequence_scaled.shape}, y_scaled={y_scaled.shape}")

        # ====================== 6. 创建训练测试集 ======================
        logger.info("====== 创建训练测试集 ======")
        X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
            X_sequence_scaled, y_scaled, split_method='time_series'
        )
        
        if len(X_train) < 10:
            logger.error("训练数据不足，请增加数据量或减少测试比例")
            return

        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {len(X_train)}个样本 ({len(X_train)/len(X_sequence)*100:.1f}%)")
        logger.info(f"  测试集: {len(X_test)}个样本 ({len(X_test)/len(X_sequence)*100:.1f}%)")

        # ====================== 7. 训练混合模型 ======================
        logger.info("====== 开始训练混合模型 ======")
        model = HybridPredictor()
        

        training_results = model.fit(
            X_static=None, 
            X_sequence=X_train, 
            y=y_train
        )
        
        logger.info("模型训练完成")

        # ====================== 8. 评估模型 ======================
        logger.info("====== 评估模型性能 ======")
        evaluation_results = model.evaluate(None, X_test, y_test)
        
        logger.info("模型评估结果:")
        for metric, value in evaluation_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        # 生成T+1股票推荐
        logger.info("生成T+1股票推荐...")
        test_predictions = model.predict(None, X_test)
        
        # 准备测试集股票代码
        test_stock_codes = stock_codes[len(X_train):]
        
        # 按股票代码分组，取最新预测
        predictions_df = pd.DataFrame({
            'stock_code': test_stock_codes,
            'prediction': test_predictions
        })
        latest_predictions = predictions_df.groupby('stock_code').tail(1)
        
        # 生成推荐
        pred_dict = dict(zip(latest_predictions['stock_code'], latest_predictions['prediction']))
        recommendations = model.get_recommendations(pred_dict, top_n=10)
        
        # 计算下一个交易日
        latest_date = max(dates)
        next_trading_day = (datetime.datetime.strptime(latest_date, '%Y%m%d') + datetime.timedelta(days=1)).strftime('%Y%m%d')
        
        # 显示推荐结果
        logger.info(f"\n=== T+1 股票推荐 (交易日期: {next_trading_day}) ===")
        logger.info("前10增长股票:")
        for rank, (code, score) in enumerate(recommendations, 1):
            logger.info(f"{rank}. {code}: 预测收益率 {score:.4f}")

        # ====================== 9. 保存预处理器和模型 ======================
        logger.info("====== 保存模型和预处理器 ======")
        os.makedirs('models', exist_ok=True)
        
        try:
            preprocessor.save_preprocessor('models/data_preprocessor.joblib')
            model.save_models('models/hybrid_model')
            logger.info("模型和预处理器保存成功")
        except Exception as e:
            logger.error(f"保存模型时出错: {str(e)}")
        
        logger.info("====== 训练流程完成 ======")
        
        # 测试模式下打印关键信息
        if test_mode:
            logger.info("=== 测试模式总结 ===")
            logger.info(f"使用的股票: {stock_codes[:3]}...")
            logger.info(f"数据时间范围: {min(dates)} 到 {max(dates)}")
            logger.info(f"最终样本数: {len(X_sequence)}")
            logger.info("测试运行成功完成！")

    except Exception as e:
        logger.error(f"系统执行失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()