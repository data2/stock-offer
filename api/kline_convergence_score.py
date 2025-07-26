import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from colorama import Fore, Style, init  # 控制台颜色输出
import numpy as np
from prettytable import PrettyTable  # 美观的表格输出
import time
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
import traceback
import math

"""
0-5	均线结构差	避免参与
5-8	结构一般	需结合其他指标验证
8-12	结构良好	重点观察候选
12-15	完美多头结构	优先考虑
"""

def calculate_convergence_score(self, ma_values: np.ndarray, klines: List[Dict], lookback_days: int = 5) -> float:
    """
    终极版均线综合评分系统（0-15分）
    评分维度：
    1. 聚合度评分（0-7分）：基于变异系数和均线距离
    2. 方向评分（0-5分）：基于均线角度和趋势强度
    3. 形态评分（0-3分）：基于均线排列和多头形态

    参数:
        ma_values: [MA5, MA10, MA20]的当前值数组
        klines: 包含足够历史数据的K线列表
        lookback_days: 计算趋势用的回溯天数（默认5天）

    返回:
        综合评分（0-15分），保留2位小数
    """

    # =================================================================
    # 第一部分：聚合度评分（0-7分）
    # =================================================================
    def calculate_convergence():
        """计算均线聚合度得分"""
        # 动态调整完美聚合阈值（根据市场波动率）
        market_volatility = np.std([k['close'] for k in klines[-20:]]) / np.mean([k['close'] for k in klines[-20:]])
        cv_perfect = max(0.003, 0.01 - market_volatility * 0.2)  # 波动越大容忍度越高

        cv = np.std(ma_values) / np.mean(ma_values)

        # 非线性评分曲线（更强调完美聚合）
        if cv <= cv_perfect:
            return 7.0
        elif cv <= cv_perfect * 2:
            return 6 - (cv - cv_perfect) / cv_perfect * 2
        elif cv <= cv_perfect * 4:
            return 4 - (cv - cv_perfect * 2) / cv_perfect
        else:
            return max(0, 2 - (cv - cv_perfect * 4) / cv_perfect)

    convergence_score = calculate_convergence()

    # =================================================================
    # 第二部分：方向评分（0-5分）
    # =================================================================
    def calculate_angle_score():
        """计算均线趋势角度得分"""
        if len(klines) < lookback_days + 1:
            return 0

        # 计算三条均线的角度和R²值（趋势可靠性）
        angles = []
        r_squared = []

        for ma in ['ma5', 'ma10', 'ma20']:
            y = np.array([k[ma] for k in klines[-lookback_days:] if ma in k])
            if len(y) < 3: continue

            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            angle = math.degrees(math.atan(slope / (y[-1] - y[0] + 1e-6)))  # 防止除零

            # 计算R²值（趋势强度）
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-6))

            angles.append(angle)
            r_squared.append(r2)

        if not angles: return 0

        # 加权平均角度（R²值作为权重）
        avg_angle = np.average(angles, weights=r_squared)
        avg_r2 = np.mean(r_squared)

        # 动态角度评分（考虑趋势可靠性）
        if avg_r2 < 0.6:
            return min(2, avg_angle / 15)  # 弱趋势最高2分
        elif avg_angle <= 5:
            return 0
        elif avg_angle <= 15:
            return 3 * (avg_angle - 5) / 10  # 5-15°线性评分
        elif avg_angle <= 25:
            return 3 + 2 * (avg_angle - 15) / 10  # 15-25°加分
        else:
            return max(0, 5 - (avg_angle - 25) / 10)  # >25°递减

    angle_score = calculate_angle_score()

    # =================================================================
    # 第三部分：形态评分（0-3分）
    # =================================================================
    def calculate_pattern_score():
        """计算均线排列形态得分"""
        ma5, ma10, ma20 = ma_values

        # 1. 多头排列基础分（MA5>MA10>MA20）
        if ma5 > ma10 > ma20:
            base_score = 1.5
        elif ma5 > ma10 and ma10 > ma20:
            base_score = 1.0
        else:
            base_score = 0

        # 2. 均线间距合理性（防止过度发散）
        spacing_ratio = (ma5 - ma20) / ma20
        if 0.02 < spacing_ratio < 0.1:  # 2%-10%为理想间距
            spacing_score = 1.0
        else:
            spacing_score = max(0, 1 - abs(spacing_ratio - 0.06) / 0.1)

        # 3. 近期金叉/死叉判断
        if len(klines) >= 3:
            prev_ma5 = klines[-2]['ma5']
            prev_ma10 = klines[-2]['ma10']
            golden_cross = (prev_ma5 < prev_ma10) and (ma5 > ma10)
            death_cross = (prev_ma5 > prev_ma10) and (ma5 < ma10)

            if golden_cross:
                cross_score = 1.0
            elif death_cross:
                cross_score = -0.5
            else:
                cross_score = 0
        else:
            cross_score = 0

        return max(0, min(3, base_score + spacing_score + cross_score))

    pattern_score = calculate_pattern_score()

    # =================================================================
    # 综合评分（动态权重调整）
    # =================================================================
    # 根据市场阶段调整权重（示例：通过20日涨幅判断）
    market_trend = (klines[-1]['close'] - klines[-20]['close']) / klines[-20]['close']

    if market_trend > 0.05:  # 强势市场
        weights = [0.5, 0.3, 0.2]  # 更看重聚合度
    elif market_trend < -0.05:  # 弱势市场
        weights = [0.3, 0.5, 0.2]  # 更看重趋势
    else:  # 震荡市场
        weights = [0.4, 0.4, 0.2]

    total_score = (
            convergence_score * weights[0] +
            angle_score * weights[1] * (7 / 5) +  # 归一化
            pattern_score * weights[2] * (7 / 3)  # 归一化
    )

    return min(15, max(0, round(total_score, 2)))