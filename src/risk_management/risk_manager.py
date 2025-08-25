import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RiskConfig:
    max_position_size: float = 0.1      # 单个股票最大仓位
    max_sector_exposure: float = 0.3    # 单个行业最大敞口
    stop_loss: float = 0.05            # 止损线
    take_profit: float = 0.15          # 止盈线
    max_volatility: float = 0.3        # 最大波动率
    max_drawdown: float = 0.2          # 最大回撤限制
    min_liquidity: float = 1e6         # 最小日均成交额


class RiskManager:
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        
    def calculate_position_size(self, 
                              prediction: float, 
                              confidence: float, 
                              volatility: float) -> float:
        """计算建议仓位"""
        # 基础仓位
        base_position = abs(prediction) * confidence
        
        # 波动率调整
        vol_factor = max(0, 1 - volatility/self.config.max_volatility)
        
        # 最终仓位
        position = min(
            base_position * vol_factor,
            self.config.max_position_size
        )
        
        return position

    def check_portfolio_risk(self, 
                           portfolio: Dict, 
                           market_data: pd.DataFrame) -> Dict:
        """检查组合风险"""
        risk_metrics = {
            'position_violations': [],
            'sector_violations': [],
            'volatility_violations': [],
            'liquidity_violations': [],
            'total_risk_score': 0.0
        }
        
        # 检查仓位集中度
        total_value = sum(pos['value'] for pos in portfolio.values())
        for stock, pos in portfolio.items():
            exposure = pos['value'] / total_value
            if exposure > self.config.max_position_size:
                risk_metrics['position_violations'].append({
                    'stock': stock,
                    'exposure': exposure,
                    'limit': self.config.max_position_size
                })

        # 计算总体风险分数
        risk_metrics['total_risk_score'] = len(risk_metrics['position_violations']) * 0.3 + \
                                         len(risk_metrics['sector_violations']) * 0.3 + \
                                         len(risk_metrics['volatility_violations']) * 0.2 + \
                                         len(risk_metrics['liquidity_violations']) * 0.2
                                         
        return risk_metrics

    def generate_risk_adjusted_signals(self, 
                                     predictions: Dict, 
                                     market_data: pd.DataFrame) -> Dict:
        """生成风险调整后的交易信号"""
        risk_adjusted_signals = {}
        
        for stock, pred in predictions.items():
            # 计算波动率
            stock_data = market_data[market_data['ts_code'] == stock]
            volatility = stock_data['close'].pct_change().std() * np.sqrt(252)
            
            # 检查流动性
            avg_volume = stock_data['amount'].mean()
            
            # 风险检查
            if volatility > self.config.max_volatility:
                continue
            if avg_volume < self.config.min_liquidity:
                continue
                
            # 调整信号强度
            confidence = 1 - (volatility / self.config.max_volatility)
            position_size = self.calculate_position_size(pred, confidence, volatility)
            
            if position_size > 0:
                risk_adjusted_signals[stock] = {
                    'original_prediction': pred,
                    'adjusted_position': position_size,
                    'confidence': confidence,
                    'volatility': volatility,
                    'liquidity': avg_volume
                }
                
        return risk_adjusted_signals