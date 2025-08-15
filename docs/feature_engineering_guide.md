# 股票特征工程指南

## 概述

本系统专为A股T+1交易策略设计，提供了完整的股票技术特征计算和处理功能。基于用户已获取的股票历史数据（包含开高低收、成交量、成交额、振幅、涨跌幅、换手率等），系统能够生成100+个技术特征用于机器学习模型训练。

## 数据结构要求

输入数据必须包含以下列：
- `trade_date`: 交易日期 (格式: YYYYMMDD)
- `ts_code`: 股票代码 (格式: 000001.SZ)
- `open`: 开盘价
- `close`: 收盘价
- `high`: 最高价
- `low`: 最低价
- `volume`: 成交量
- `amount`: 成交额
- `amplitude`: 振幅
- `change`: 涨跌幅
- `turnover`: 换手率

## 特征类别

### 1. 基础价格特征
- **价格相对位置**: 收盘价在当日价格区间中的位置
- **价格变化率**: 1日、3日、5日、10日价格变化率
- **开盘缺口**: 当日开盘价相对前日收盘价的跳空幅度
- **日内波动**: 当日价格波动幅度
- **收盘位置**: 收盘价相对当日开盘价的强弱

### 2. 技术指标特征
- **移动平均线**: SMA(5,10,20,30)、EMA(5,10,20,30)
- **相对强弱指标**: RSI(6,12,24)
- **MACD指标**: MACD线、信号线、柱状图
- **布林带**: 上轨、下轨、带宽、相对位置
- **KDJ指标**: K值、D值、J值
- **ATR指标**: 真实波幅、相对波幅
- **OBV指标**: 能量潮指标

### 3. 量价特征
- **成交量移动平均**: 5日、10日、20日成交量均值
- **成交量比率**: 相对移动平均的成交量倍数
- **成交额移动平均**: 5日、10日、20日成交额均值
- **价格成交量相关性**: 10日窗口内价格与成交量的相关性
- **换手率特征**: 换手率移动平均、相对比率
- **量价配合度**: 价格变动与成交量的配合程度

### 4. 波动率特征
- **历史波动率**: 5日、10日、20日历史波动率
- **GARCH波动率**: 基于平方收益率的波动率估计
- **高低价差波动**: 高低价差的标准差
- **振幅特征**: 5日、10日平均振幅

### 5. 动量特征
- **ROC指标**: 5日、10日、20日价格变化率
- **动量指标**: 1日、3日、5日价格动量
- **威廉指标**: 10日、20日威廉指标
- **价格通道**: 10日价格通道位置

### 6. 统计特征
- **偏度峰度**: 10日、20日价格分布的偏度和峰度
- **价格分位数**: 10日、20日价格分位数
- **成交量分位数**: 10日、20日成交量分位数
- **均值回归**: 相对移动平均的Z-Score

### 7. 市场情绪特征
- **上涨比率**: 5日、10日上涨天数占比
- **连续涨跌**: 连续上涨/下跌天数
- **价格强度**: 收盘价相对当日价格区间的强度

### 8. 标签特征
- **次日收益率**: 次日收盘相对今日收盘的收益率
- **次日涨跌**: 次日是否上涨的二分类标签
- **收益分档**: 次日收益率的6档分类标签

## 使用方法

### 快速开始

```python
from src.data_preprocessing.technical_indicators import TechnicalIndicators

# 假设df是你的股票数据DataFrame
processor = TechnicalIndicators(df)
features = processor.calculate_all_features()

# 特征选择
selected_features = processor.select_features(threshold=0.05)

# 特征标准化
normalized_features = processor.normalize_features(method='standard')

# 获取最新预测特征
latest_features = processor.get_latest_features(days=1)
```

### 批量处理

```python
from src.data_preprocessing.technical_indicators import batch_process_stocks

# stock_data_list是多个股票DataFrame的列表
all_features = batch_process_stocks(stock_data_list)
```

### 使用特征处理器

```python
from src.data_preprocessing.feature_processor import FeatureProcessor

# 创建特征处理器
processor = FeatureProcessor(data_source='tushare')

# 处理所有中证800股票
features = processor.process_all_stocks(
    start_date='20240101',
    end_date='20241231',
    save_path='data/processed/stock_features.csv'
)

# 处理单只股票
single_features = processor.process_single_stock_features('000001.SZ')

# 获取预测特征
pred_features = processor.get_latest_prediction_features('000001.SZ', days=3)
```

## 特征选择建议

### 相关性筛选
使用`select_features(threshold=0.05)`可以筛选出与次日收益相关性大于0.05的特征，通常能保留50-70个有效特征。

### 特征重要性排序
使用`get_feature_importance()`可以查看特征与次日收益的相关性排名，帮助理解哪些特征对预测更重要。

### 常用有效特征
基于历史经验，以下特征通常对次日预测较为有效：
- RSI系列指标
- MACD相关指标
- 成交量比率
- 价格相对移动平均的位置
- 波动率指标
- 价格通道位置

## 注意事项

1. **数据质量**: 确保输入数据没有缺失值和异常值
2. **时间序列**: 数据必须按时间升序排列
3. **标准化**: 训练模型前务必进行特征标准化
4. **标签泄露**: 确保预测时使用的特征不包含未来信息
5. **计算效率**: 大量股票计算时建议使用批量处理

## 示例数据

运行`examples/feature_processing_demo.py`可以查看完整的使用示例和特征效果。

## 扩展建议

如需添加更多特征，可以在`TechnicalIndicators`类中添加新的计算方法，并确保：
1. 特征计算只使用历史数据
2. 处理缺失值和异常值
3. 添加适当的特征描述
4. 测试新特征的有效性

## 技术支持

如遇到问题，请检查：
1. 数据格式是否符合要求
2. 依赖包是否安装完整（pandas, numpy, talib, sklearn）
3. 数据时间跨度是否足够（建议至少60个交易日）
4. 特征计算是否出现内存不足（可分批处理）