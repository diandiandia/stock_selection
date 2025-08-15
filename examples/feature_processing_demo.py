#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票特征处理快速上手示例
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 示例1：处理已有数据
print("=== 示例1：处理已有股票数据 ===")

# 假设你已经有了股票数据DataFrame
# 数据格式应该包含：trade_date, ts_code, open, close, high, low, volume, amount, amplitude, change, turnover

# 创建示例数据（你可以用自己的数据替换）
example_data = pd.DataFrame({
    'trade_date': pd.date_range('2024-01-01', periods=100, freq='D').strftime('%Y%m%d'),
    'ts_code': '000001.SZ',
    'open': np.random.uniform(10, 15, 100),
    'close': np.random.uniform(10, 15, 100),
    'high': np.random.uniform(10.5, 15.5, 100),
    'low': np.random.uniform(9.5, 14.5, 100),
    'volume': np.random.uniform(1000000, 5000000, 100),
    'amount': np.random.uniform(10000000, 50000000, 100),
    'amplitude': np.random.uniform(1, 5, 100),
    'change': np.random.uniform(-5, 5, 100),
    'turnover': np.random.uniform(1, 10, 100)
})

# 确保收盘价是合理的
example_data['close'] = example_data['close'].sort_values().values
example_data['high'] = example_data[['high', 'close']].max(axis=1)
example_data['low'] = example_data[['low', 'close']].min(axis=1)

print("原始数据示例:")
print(example_data.head())

# 导入特征处理模块
from src.data_preprocessing.technical_indicators import TechnicalIndicators

# 创建特征处理器
processor = TechnicalIndicators(example_data)

# 计算所有特征
features = processor.calculate_all_features()

print(f"\n生成的特征数量: {len(features.columns)}")
print("特征列名:")
for col in features.columns:
    print(f"  - {col}")

print("\n特征数据示例:")
print(features[['trade_date', 'ts_code', 'close', 'next_day_return', 'next_day_up', 'rsi_6', 'macd']].tail(5))

# 特征选择
print("\n=== 特征选择 ===")
selected_features = processor.select_features(threshold=0.01)
print(f"选择后的特征数量: {len(selected_features.columns)}")

# 特征重要性
print("\n=== 特征重要性 ===")
importance = processor.get_feature_importance()
print("前10个重要特征:")
print(importance.head(10))

# 标准化特征
print("\n=== 特征标准化 ===")
normalized_features = processor.normalize_features(method='standard')
print("标准化后的特征统计:")
print(normalized_features.describe())

# 获取最新特征用于预测
print("\n=== 最新预测特征 ===")
latest_features = processor.get_latest_features(days=1)
print("最新一天特征:")
print(latest_features.drop(columns=['next_day_return', 'next_day_up', 'next_day_return_class'], errors='ignore'))

print("\n=== 使用提示 ===")
print("1. 替换example_data为你自己的股票数据")
print("2. 确保数据包含所有必要列：trade_date, ts_code, open, close, high, low, volume, amount, amplitude, change, turnover")
print("3. 使用processor.select_features()进行特征选择")
print("4. 使用processor.normalize_features()进行标准化")
print("5. 使用processor.get_latest_features()获取最新预测特征")