from typing import Dict, List
import numpy as np
import pandas as pd
from src.risk_management.risk_manager import RiskManager, RiskConfig


class RiskStrategy:
    def __init__(self, risk_config: RiskConfig = None):
        self.risk_manager = RiskManager(risk_config)
        
    def apply_position_limits(self, 
                            recommendations: pd.DataFrame, 
                            market_data: pd.DataFrame) -> pd.DataFrame:
        """应用仓位限制"""
        risk_adjusted_recs = recommendations.copy()
        
        # 计算波动率和流动性指标
        for idx, row in risk_adjusted_recs.iterrows():
            stock_data = market_data[market_data['ts_code'] == row['ts_code']]
            
            volatility = stock_data['close'].pct_change().std() * np.sqrt(252)
            avg_volume = stock_data['amount'].mean()
            
            # 应用风险限制
            if volatility > self.risk_manager.config.max_volatility:
                risk_adjusted_recs.drop(idx, inplace=True)
                continue
                
            if avg_volume < self.risk_manager.config.min_liquidity:
                risk_adjusted_recs.drop(idx, inplace=True)
                continue
            
            # 调整建议仓位
            confidence = risk_adjusted_recs.loc[idx, 'confidence']
            pred = risk_adjusted_recs.loc[idx, 'prediction']
            
            position_size = self.risk_manager.calculate_position_size(
                pred, confidence, volatility
            )
            
            risk_adjusted_recs.loc[idx, 'position_size'] = position_size
            
        return risk_adjusted_recs

    def generate_stop_orders(self, 
                           portfolio: Dict, 
                           market_data: pd.DataFrame) -> List[Dict]:
        """生成止盈止损订单"""
        stop_orders = []
        
        for stock, position in portfolio.items():
            entry_price = position['price']
            current_price = market_data[market_data['ts_code'] == stock]['close'].iloc[-1]
            
            profit_rate = (current_price - entry_price) / entry_price
            
            # 检查止损
            if profit_rate < -self.risk_manager.config.stop_loss:
                stop_orders.append({
                    'stock': stock,
                    'action': 'sell',
                    'reason': 'stop_loss',
                    'profit_rate': profit_rate
                })
                
            # 检查止盈
            elif profit_rate > self.risk_manager.config.take_profit:
                stop_orders.append({
                    'stock': stock,
                    'action': 'sell',
                    'reason': 'take_profit',
                    'profit_rate': profit_rate
                })
                
        return stop_orders