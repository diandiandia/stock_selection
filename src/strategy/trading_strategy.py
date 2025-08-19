import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from src.utils.log_helper import LogHelper

class TradingStrategy:
    """T+1交易策略模块，提供卖出条件判断和交易建议"""
    def __init__(self, config: Dict = None):
        self.logger = LogHelper.get_logger(__name__)
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """获取默认策略配置"""
        return {
            # 止盈止损配置
            'take_profit': 0.05,        # 5%止盈
            'stop_loss': -0.03,         # 3%止损
            'trailing_stop': 0.02,      # 2%追踪止损
            
            # 技术指标配置
            'rsi_overbought': 70,       # RSI超买阈值
            'rsi_oversold': 30,         # RSI超卖阈值
            'macd_cross_signal': True,  # MACD交叉信号
            'bollinger_band_signal': True, # 布林带突破信号
            
            # 时间窗口配置
            'morning_window': '09:30-10:30', # 早盘窗口
            'afternoon_window': '14:00-15:00' # 尾盘窗口
        }
    
    def evaluate_sell_conditions(self, 
                                stock_code: str, 
                                buy_price: float, 
                                current_data: pd.DataFrame, 
                                market_trend: str = 'neutral') -> Tuple[bool, str, float]:
        """
        评估卖出条件
        
        Args:
            stock_code: 股票代码
            buy_price: 买入价格
            current_data: 当前交易日数据（包含实时行情和指标）
            market_trend: 大盘趋势（'bull'/'bear'/'neutral'）
        
        Returns:
            Tuple[是否卖出, 卖出理由, 建议卖出价格]
        """
        if current_data.empty:
            self.logger.warning(f"股票{stock_code}无当前数据，无法评估卖出条件")
            return False, "无数据", 0.0
        
        # 获取当前价格
        current_price = current_data['close'].iloc[-1] if 'close' in current_data.columns else current_data['price'].iloc[-1]
        price_change = (current_price - buy_price) / buy_price
        
        # 1. 止盈止损条件
        if price_change >= self.config['take_profit']:
            return True, f"达到止盈({price_change:.2%})", current_price
        if price_change <= self.config['stop_loss']:
            return True, f"达到止损({price_change:.2%})", current_price
        
        # 2. 技术指标条件
        sell_reason = []
        
        # RSI超买
        if 'rsi' in current_data.columns and not pd.isna(current_data['rsi'].iloc[-1]):
            if current_data['rsi'].iloc[-1] >= self.config['rsi_overbought']:
                sell_reason.append(f"RSI超买({current_data['rsi'].iloc[-1]:.1f})")
        
        # MACD死叉
        if self.config['macd_cross_signal'] and all(col in current_data.columns for col in ['macd', 'macd_signal']):
            if current_data['macd'].iloc[-2] > current_data['macd_signal'].iloc[-2] and current_data['macd'].iloc[-1] <= current_data['macd_signal'].iloc[-1]:
                sell_reason.append("MACD死叉")
        
        # 布林带上轨突破
        if self.config['bollinger_band_signal'] and all(col in current_data.columns for col in ['boll_ub', 'close']):
            if current_data['close'].iloc[-1] >= current_data['boll_ub'].iloc[-1]:
                sell_reason.append("突破布林带上轨")
        
        # 综合技术指标判断
        if len(sell_reason) >= 2:
            return True, ",".join(sell_reason), current_price
        
        # 3. 市场趋势配合
        if market_trend == 'bear' and price_change < 0:
            return True, f"熊市下跌({price_change:.2%})", current_price
        
        # 4. 尾盘无盈利
        current_time = pd.Timestamp.now().strftime('%H:%M')
        if self._is_in_time_window(current_time, self.config['afternoon_window']) and price_change <= 0:
            return True, f"尾盘无盈利({price_change:.2%})", current_price
        
        # 未满足卖出条件
        return False, "未满足卖出条件", 0.0
    
    def _is_in_time_window(self, current_time: str, time_window: str) -> bool:
        """判断当前时间是否在目标窗口内"""
        start, end = time_window.split('-')
        start_h, start_m = map(int, start.split(':'))
        end_h, end_m = map(int, end.split(':'))
        
        current_h, current_m = map(int, current_time.split(':'))
        current_total = current_h * 60 + current_m
        start_total = start_h * 60 + start_m
        end_total = end_h * 60 + end_m
        
        return start_total <= current_total <= end_total
    
    def generate_sell_recommendations(self, 
                                     portfolio: Dict[str, Tuple[float, int]], 
                                     market_data: Dict[str, pd.DataFrame], 
                                     market_trend: str = 'neutral') -> List[Dict]:
        """
        为当前持仓生成卖出建议
        
        Args:
            portfolio: 持仓字典 {股票代码: (买入价格, 数量)}
            market_data: 市场数据 {股票代码: 数据DataFrame}
            market_trend: 大盘趋势
        
        Returns:
            卖出建议列表
        """
        recommendations = []
        
        for stock_code, (buy_price, quantity) in portfolio.items():
            if stock_code not in market_data:
                self.logger.warning(f"股票{stock_code}无市场数据")
                continue
            
            sell_flag, reason, price = self.evaluate_sell_conditions(
                stock_code, buy_price, market_data[stock_code], market_trend
            )
            
            if sell_flag:
                recommendations.append({
                    'stock_code': stock_code,
                    'buy_price': buy_price,
                    'current_price': price,
                    'profit_rate': (price - buy_price) / buy_price,
                    'reason': reason,
                    'suggested_action': 'SELL',
                    'quantity': quantity,
                    'estimated_profit': (price - buy_price) * quantity
                })
        
        # 按建议优先级排序（亏损优先，盈利次之）
        recommendations.sort(key=lambda x: (x['profit_rate'] < 0, -abs(x['profit_rate'])))
        return recommendations