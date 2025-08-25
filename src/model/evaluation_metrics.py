import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats


class ModelEvaluator:
    def __init__(self):
        self.risk_free_rate = 0.03  # 年化无风险利率
        
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         dates: np.ndarray,
                         prices: np.ndarray) -> Dict:
        """计算完整的评估指标集"""
        # 基础预测指标
        base_metrics = self._calculate_prediction_metrics(y_true, y_pred)
        
        # 方向准确性指标
        direction_metrics = self._calculate_direction_metrics(y_true, y_pred)
        
        # 收益相关指标
        returns_metrics = self._calculate_returns_metrics(
            y_true, y_pred, dates, prices
        )
        
        # 风险调整指标
        risk_metrics = self._calculate_risk_metrics(
            returns_metrics['daily_returns']
        )
        
        return {**base_metrics, **direction_metrics, 
                **returns_metrics, **risk_metrics}
    
    def _calculate_prediction_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict:
        """计算基础预测指标"""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    def _calculate_direction_metrics(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict:
        """计算方向预测准确性指标"""
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        
        direction_accuracy = np.mean(true_direction == pred_direction)
        
        # 计算混淆矩阵
        tp = np.sum((true_direction == 1) & (pred_direction == 1))
        tn = np.sum((true_direction == -1) & (pred_direction == -1))
        fp = np.sum((true_direction == -1) & (pred_direction == 1))
        fn = np.sum((true_direction == 1) & (pred_direction == -1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'direction_accuracy': direction_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _calculate_returns_metrics(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                dates: np.ndarray,
                                prices: np.ndarray) -> Dict:
        """计算收益相关指标"""
        # 计算每日收益率
        daily_returns = pd.Series(index=dates)
        for i in range(len(y_pred)):
            if y_pred[i] > 0:
                daily_returns[dates[i]] = y_true[i]
                
        # 计算累积收益
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        # 年化收益率
        trading_days = len(dates)
        annual_return = (1 + cumulative_returns.iloc[-1]) ** (252/trading_days) - 1
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        return {
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'calmar_ratio': abs(annual_return / max_drawdown) if max_drawdown != 0 else np.inf
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """计算风险调整指标"""
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        excess_returns = returns - self.risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() \
            if returns.std() != 0 else 0
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() \
            if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
        
        # 信息比率
        information_ratio = np.sqrt(252) * returns.mean() / returns.std() \
            if returns.std() != 0 else 0
        
        return {
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio
        }
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return drawdowns.min()