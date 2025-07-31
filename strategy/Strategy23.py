"""
策略二分析器：识别放量大涨后缩量调整至极致的股票

该模块用于识别符合以下特征的股票：
1. 出现显著放量大涨（涨幅>7%且成交量显著放大）
2. 随后成交量萎缩至高峰期的35%以下
3. 价格回调幅度不超过18%
4. 股价回调至关键均线（MA5/MA10/MA20）附近
"""

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

# 初始化colorama（自动重置颜色）
init(autoreset=True)


class StrategyTwoAnalyzer:
    """
    策略类型2分析器：放量大涨后缩量调整至极致

    核心逻辑：
    1. 识别放量大涨日（涨幅>7%且成交量显著放大）
    2. 检测后续缩量调整过程（成交量萎缩至高峰期的35%以下）
    3. 验证价格回调幅度（不超过18%）
    4. 确认股价是否回调至关键均线附近
    """

    def __init__(self):
        """初始化分析器参数"""
        # 策略参数
        self.min_change_rate = 7  # 最小涨幅阈值（%）
        self.volume_threshold = 0.4  # 成交量萎缩阈值（相对于高峰期的比例）
        self.price_drop_threshold = 0.18  # 最大允许价格回调幅度（18%）
        self.ABSOLUTE_SHRINK_THRESHOLD = 0.4  # 绝对成交量萎缩阈值
        self.TEMPORARY_INCREASE_ALLOWANCE = 0.2  # 临时成交量放大容忍度
        self.MIN_CONSECUTIVE_DAYS = 8  # 最小缩量调整天数（非连续）
        self.MIN_MA_SCORE = 5  # 均线聚合最小得分

        # HTTP请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Referer": "https://emrnweb.eastmoney.com/"
        }

        # 创建输出目录
        if not os.path.exists("strategy_two_plots"):
            os.makedirs("strategy_two_plots")

    def parse_kline(self, kline_str: str) -> Dict:
        """
        解析K线数据字符串为字典

        参数:
            kline_str: 逗号分隔的K线数据字符串

        返回:
            包含解析后K线数据的字典
        """
        fields = kline_str.split(",")
        return {
            "date": fields[0],  # 交易日期（YYYY-MM-DD）
            "open": float(fields[1]),  # 开盘价
            "close": float(fields[2]),  # 收盘价
            "high": float(fields[3]),  # 最高价
            "low": float(fields[4]),  # 最低价
            "volume": float(fields[5]),  # 成交量
            "amount": float(fields[6]),  # 成交额
            "amplitude": float(fields[7]),  # 振幅
            "change_rate": float(fields[8]),  # 涨跌幅（%）
            "change_amount": float(fields[9]),  # 涨跌额
            "turnover": float(fields[10]) if len(fields) > 10 else 0,  # 换手率
            "ma5": 0,  # 5日均线（初始化为0）
            "ma10": 0,  # 10日均线
            "ma20": 0  # 20日均线
        }

    def get_daily_kline(self, stock_code: str, years: float = 0.5) -> Optional[List[Dict]]:
        """
        获取股票日K线数据（带重试机制）

        参数:
            stock_code: 6位股票代码
            years: 获取数据的时间跨度（年，默认0.5年）

        返回:
            K线数据字典列表，失败返回None
        """
        max_retries = 3  # 最大重试次数
        retry_delay = 2  # 重试间隔（秒）

        for attempt in range(max_retries):
            try:
                # 计算日期范围
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=int(365 * years))).strftime('%Y%m%d')

                # API请求参数
                params = {
                    "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                    "beg": start_date,
                    "end": end_date,
                    "ut": "fa5fd1943c7b386f172d6893dbfba10b",
                    "rtntype": "6",
                    "secid": f"1.{stock_code}" if stock_code.startswith('6') else f"0.{stock_code}",
                    "klt": "101",  # 日K线
                    "fqt": "1"  # 前复权
                }

                url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                response.raise_for_status()  # 检查HTTP错误

                # 处理JSON响应
                json_str = response.text.strip()
                if json_str.startswith("jsonp") and json_str.endswith(")"):
                    json_str = json_str[json_str.index("(") + 1:-1]  # 去除JSONP包装

                data = json.loads(json_str)
                if data.get("rc") == 0 and data.get("data", {}).get("klines"):
                    klines = [self.parse_kline(k) for k in data["data"]["klines"]]

                    # 计算并填充均线值
                    for i in range(len(klines)):
                        if i >= 4:  # MA5需要至少5天数据
                            klines[i]["ma5"] = self.calculate_ma(klines[:i + 1], 5)
                        if i >= 9:  # MA10
                            klines[i]["ma10"] = self.calculate_ma(klines[:i + 1], 10)
                        if i >= 19:  # MA20
                            klines[i]["ma20"] = self.calculate_ma(klines[:i + 1], 20)
                    return klines

            except Exception as e:
                time.sleep(1)
                if attempt < max_retries - 1:  # 非最后一次重试
                    print(f"{Fore.YELLOW}获取日K线失败({stock_code})，"
                          f"第 {attempt + 1}/{max_retries} 次重试...{Style.RESET_ALL}")
                    time.sleep(retry_delay)
                    continue
                print(f"{Fore.RED}获取日K线失败({stock_code}): {e}{Style.RESET_ALL}")
                return None

    def get_stock_name(self, stock_code: str) -> str:
        """
        根据股票代码获取股票名称

        参数:
            stock_code: 6位股票代码

        返回:
            股票名称，失败返回"未知"
        """
        url = "https://emrnweb.eastmoney.com/api/security/quote"
        params = {
            "secids": f"1.{stock_code}" if stock_code.startswith('6') else f"0.{stock_code}"
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            data = response.json()
            return data["data"][0]["f14"]
        except Exception:
            return "未知"

    def get_hot_stocks(self, top_n: int = 500) -> Optional[List[Dict]]:
        """
        获取东方财富热股榜数据

        参数:
            top_n: 获取的热股数量（默认100）

        返回:
            热股字典列表，失败返回None
        """
        url = "https://datacenter.eastmoney.com/stock/selection/api/data/get/"

        # 准备multipart表单数据
        multipart_data = MultipartEncoder(
            fields={
                "type": "RPTA_SECURITY_STOCKSELECT",
                "sty": "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,NEW_PRICE,CHANGE_RATE,TOTAL_MARKET_CAP,POPULARITY_RANK",
                "filter": "(@LISTING_DATE=\"OVER1Y\")(TOTAL_MARKET_CAP<15000000000)(POPULARITY_RANK>0)"
                          "(POPULARITY_RANK<=1000)(HOLDER_NEWEST>0)(HOLDER_NEWEST<=40000)",
                "p": "1",
                "ps": str(top_n),
                "sr": "-1",
                "st": "POPULARITY_RANK",  # 按热度排序
                "source": "SELECT_SECURITIES",
                "client": "APP"
            },
            boundary='----WebKitFormBoundaryrNTxuNLy4HmrZleF'
        )

        headers = self.headers.copy()
        headers.update({
            "Content-Type": multipart_data.content_type,
            "Content-Length": str(multipart_data.len)
        })

        try:
            response = requests.post(url, headers=headers, data=multipart_data, timeout=15)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                return result["result"]["data"][::-1]  # 返回反转后的列表

        except Exception as e:
            print(f"{Fore.RED}获取热股榜失败: {e}{Style.RESET_ALL}")

        return None

    def is_big_rise_with_volume(self, klines: List[Dict], lookback_days: int = 20,
                                min_retrace_ratio: float = 0.9) -> Tuple[bool, Optional[Dict]]:
        """
        检查是否存在放量大涨日

        判断标准：
        1. 涨幅 > min_change_rate（默认7%）
        2. 成交量显著放大（满足任一）：
           - 是前一日2倍以上
           - 是5日均量1.5倍以上
           - 是20日均量1.2倍以上
        3. 收盘涨幅/最高涨幅 ≥ min_retrace_ratio（默认90%）

        参数:
            klines: K线数据列表
            lookback_days: 回溯天数（默认20）
            min_retrace_ratio: 收盘涨幅保留比例阈值（默认0.9）

        返回:
            (是否满足条件, 满足条件的K线数据)
        """
        # 参数校验
        if not klines or len(klines) < lookback_days + 1 or min_retrace_ratio <= 0:
            return False, None

        recent_klines = klines[-lookback_days - 1:]  # 获取最近N+1天的数据

        for i in range(1, len(recent_klines)):
            try:
                current = recent_klines[i]
                prev = recent_klines[i - 1]

                # 必需字段检查
                required_fields = ['open', 'close', 'high', 'volume', 'change_rate']
                if any(field not in current for field in required_fields):
                    continue

                # 类型转换
                open_price = float(current['open'])
                close_price = float(current['close'])
                high_price = float(current['high'])
                volume = float(current['volume'])
                prev_volume = float(prev['volume'])
                change_rate = float(current['change_rate'])

                # 条件1：基础涨幅检查
                if change_rate <= self.min_change_rate:
                    continue

                # 条件3：涨幅保留率检查
                max_rise = (high_price - open_price) / open_price  # 最高涨幅
                close_rise = (close_price - open_price) / open_price  # 收盘涨幅

                # 处理除零问题（如开盘=最高价）
                if max_rise <= 1e-6:  # 浮点精度处理
                    continue

                if (close_rise / max_rise) < min_retrace_ratio:
                    continue

                # 条件2：成交量放大检查
                lookback_start = max(0, i - 19)  # 确保有20天数据
                volumes = [float(k['volume']) for k in recent_klines[lookback_start:i + 1] if 'volume' in k]

                ma5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0  # 5日均量
                ma20 = np.mean(volumes) if len(volumes) >= 20 else 0  # 20日均量

                volume_expanded = (
                        volume > prev_volume * 2 or  # 是前一日2倍以上
                        (ma5 > 0 and volume > ma5 * 1.5) or  # 是5日均量1.5倍以上
                        (ma20 > 0 and volume > ma20 * 1.2)  # 是20日均量1.2倍以上
                )

                if volume_expanded:
                    return True, current

            except (TypeError, ValueError) as e:
                continue  # 跳过数据异常的日子

        return False, None  # 未找到符合条件的交易日

    def is_shrink_adjustment(self, klines: List[Dict], peak_day: Dict) -> Tuple[bool, int, float, float]:
        """
        改进版缩量调整判断

        参数:
            klines: K线数据列表
            peak_day: 放量大涨日的K线数据

        返回:
            (是否满足缩量条件, 有效缩量天数, 最终量比, 当前价格回调幅度)
        """
        if len(klines) < 5:  # 至少需要5天数据
            return False, 0, 0, 0

        peak_volume = peak_day["volume"]
        peak_price = peak_day["close"]
        peak_date = peak_day["date"]

        # 找到大涨日在K线中的位置
        peak_index = next((i for i, k in enumerate(klines) if k["date"] == peak_date), -1)

        # 检查大涨日位置是否有效
        if peak_index == -1 or peak_index >= len(klines) - 3:
            return False, 0, 0, 0

        effective_days = 0  # 有效缩量天数
        volume_ma5 = []  # 存储5日均量数据
        end_index = peak_index  # 初始化结束索引

        # 从大涨日后开始检查
        for i in range(peak_index + 1, len(klines)):
            current = klines[i]
            prev = klines[i - 1]
            current_ratio = current["volume"] / peak_volume  # 当前成交量比例
            price_drop = (peak_price - current["close"]) / peak_price  # 价格回调幅度

            # 计算5日均量（需要有足够数据）
            if i >= peak_index + 5:
                ma5 = np.mean([k["volume"] for k in klines[i - 4:i + 1]])
                volume_ma5.append(ma5)

            # 终止条件：价格回调超过阈值
            if price_drop > self.price_drop_threshold:
                end_index = i - 1  # 记录结束位置
                break

            # 检查成交量是否满足缩量条件
            if current["volume"] > peak_volume * self.volume_threshold:
                continue  # 跳过不满足缩量条件的交易日
            else:
                effective_days += 1
                end_index = i  # 更新结束位置

        # 获取最后一天的成交量数据
        if end_index >= len(klines):
            end_index = len(klines) - 1

        last_day = klines[-1]
        final_ratio = last_day["volume"] / peak_volume  # 最终量比

        # 计算当前价格回调幅度（即使未超过阈值）
        current_price_drop = (peak_price - last_day["close"]) / peak_price

        # 综合判断是否满足缩量条件
        qualified = (
                effective_days >= self.MIN_CONSECUTIVE_DAYS and  # 满足最小缩量天数
                final_ratio < self.ABSOLUTE_SHRINK_THRESHOLD  # 量比低于绝对阈值
        )

        return qualified, effective_days, final_ratio, current_price_drop

    def calculate_ma(self, klines: List[Dict], days: int) -> float:
        """
        计算指定天数的移动平均线

        参数:
            klines: K线数据列表
            days: 均线天数（如5、10、20）

        返回:
            移动平均值
        """
        if len(klines) < days:  # 数据不足
            return 0.0

        # 如果最新数据已计算过ma值，直接返回
        if f"ma{days}" in klines[-1] and klines[-1][f"ma{days}"] > 0:
            return klines[-1][f"ma{days}"]

        # 计算收盘价的移动平均
        closes = [k["close"] for k in klines[-days:]]
        return sum(closes) / days

    def calculate_convergence_score(self, ma_values: np.ndarray, klines: List[Dict], lookback_days: int = 5) -> float:

        """
        终极版均线综合评分系统（0-15分）
        评分维度：
        1. 聚合度评分（0-7分）：基于变异系数和均线距离
        2. 方向评分（0-5分）：基于均线角度和趋势强度
        3. 形态评分（0-3分）：基于均线排列和多头形态

        0-5	    均线结构差	避免参与
        5-8	    结构一般	 需结合其他指标验证
        8-12	结构良好	 重点观察候选
        12-15	完美多头结构	 优先考虑

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

    def is_near_key_ma(self, klines: List[Dict], adjustment_days: int = 0) -> Tuple[bool, str, float]:
        """
        严格判断股价是否回调至关键均线附近

        参数:
            klines: K线数据列表
            adjustment_days: 数据调整天数

        返回:
            (是否在关键均线附近, 触及的均线名称, 聚合分数0-10)
        """
        # 数据校验
        if len(klines) < 20 + adjustment_days:
            return False, "", 0.0

        # 计算三条关键均线
        ma5 = self.calculate_ma(klines, 5)
        ma10 = self.calculate_ma(klines, 10)
        ma20 = self.calculate_ma(klines, 20)

        # 检查数据有效性
        if ma5 <= 0 or ma10 <= 0 or ma20 <= 0:
            return False, "", 0.0

        # 计算均线聚合分数
        ma_values = np.array([ma5, ma10, ma20])
        convergence_score = self.calculate_convergence_score(ma_values, klines)

        # 动态聚合阈值（通过历史数据70分位校准）
        cv_threshold = 0.02  # 对应分数约7分
        is_converged = (np.std(ma_values) / np.mean(ma_values)) <= cv_threshold

        # 检查股价接近均线（3%阈值）
        close = klines[-1]["close"]
        near_threshold = 0.03
        is_near = {
            "MA5": abs(close - ma5) / ma5 <= near_threshold,
            "MA10": abs(close - ma10) / ma10 <= near_threshold,
            "MA20": abs(close - ma20) / ma20 <= near_threshold
        }

        # 优先返回聚合度最好的均线
        if is_converged:
            for ma_name in ["MA5", "MA10", "MA20"]:
                if is_near[ma_name]:
                    return True, ma_name, convergence_score

        return False, "", convergence_score

    def analyze_strategy_two(self, klines: List[Dict]) -> Dict:
        """
        综合分析股票是否符合策略类型2的条件

        参数:
            klines: K线数据列表

        返回:
            包含分析结果的字典
        """
        if not klines or len(klines) < 10:
            return {
                "qualified": False,
                "reason": "数据不足(需要至少10天数据)"
            }

        # 条件1: 检查放量大涨
        has_big_rise, peak_day = self.is_big_rise_with_volume(klines)
        if not has_big_rise:
            return {
                "qualified": False,
                "reason": "无放量大涨日(涨幅>7%且成交量显著放大且不回落)"
            }

        # 条件2: 检查缩量调整
        is_shrink, adjust_days, volume_ratio, current_price_drop = self.is_shrink_adjustment(klines, peak_day)

        # 条件3: 检查回调至均线附近
        is_near_ma, ma_type, ma_score = self.is_near_key_ma(klines, adjust_days)

        # 综合判断
        qualified = (
                has_big_rise and
                adjust_days >= self.MIN_CONSECUTIVE_DAYS and
                is_near_ma and
                ma_score >= self.MIN_MA_SCORE
        )

        if qualified:
            return {
                "qualified": True,
                "peak_day": peak_day["date"],
                "peak_price": peak_day["close"],
                "peak_volume": peak_day["volume"],
                "is_shrink": is_shrink,
                "adjust_days": adjust_days,
                "current_volume_ratio": volume_ratio,
                "current_price_drop": current_price_drop,
                "is_near_ma": is_near_ma,
                "near_ma": ma_type,
                "ma_score": ma_score,
                "reason": "符合所有条件"
            }
        else:
            # 构建详细的不符合原因
            reasons = []
            if not has_big_rise:
                reasons.append("无放量大涨日")
            if adjust_days < self.MIN_CONSECUTIVE_DAYS:
                reasons.append(f"调整天数不足(当前:{adjust_days}天,需要≥{self.MIN_CONSECUTIVE_DAYS}天)")
            if not is_near_ma:
                reasons.append(f"未回调至{ma_type}均线附近")
            if ma_score <= self.MIN_MA_SCORE:
                reasons.append(f"均线聚合度不足(当前:{ma_score:.1f},需要>=8)")

            return {
                "qualified": False,
                "peak_day": peak_day["date"],
                "peak_price": peak_day["close"],
                "peak_volume": peak_day["volume"],
                "is_shrink": is_shrink,
                "adjust_days": adjust_days,
                "current_volume_ratio": volume_ratio,
                "current_price_drop": current_price_drop,
                "is_near_ma": is_near_ma,
                "near_ma": ma_type,
                "ma_score": ma_score,
                "reason": " | ".join(reasons) if reasons else "未知原因"
            }

    def analyze_single_stock(self, stock_code: str) -> Optional[Dict]:
        """分析单只股票"""
        print(f"\n{Fore.YELLOW}🌟 开始分析股票 {stock_code}...{Style.RESET_ALL}")

        # 获取股票名称和K线数据
        stock_name = self.get_stock_name(stock_code)
        klines = self.get_daily_kline(stock_code)

        if not klines:
            print(f"{Fore.RED}无法获取股票 {stock_code} 的K线数据{Style.RESET_ALL}")
            return None

        # 执行策略分析
        analysis = self.analyze_strategy_two(klines)

        # 打印分析结果
        print(f"\n{Fore.CYAN}=== 分析结果 ==={Style.RESET_ALL}")
        result_line = [
            f"股票: {Fore.YELLOW}{stock_code} {stock_name}{Style.RESET_ALL}",
            f"结论: {Fore.GREEN if analysis['qualified'] else Fore.RED}"
            f"{'符合' if analysis['qualified'] else '不符合'}条件",
            f"{'' if analysis['qualified'] else analysis['reason']}{Style.RESET_ALL}"
        ]

        if "peak_day" in analysis:
            result_line.extend([
                f"放量大涨日: {analysis['peak_day']}",
                f"价格: {analysis['peak_price']:.2f}",
                f"成交量: {analysis['peak_volume']:.0f}",
                f"调整天数: {analysis['adjust_days']}天",
                f"量比: {analysis['current_volume_ratio']:.2f}",
                f"均线位置: {analysis.get('near_ma', '无')}",
                f"拟合得分: {analysis.get('ma_score', 0)}"
            ])

        print(" | ".join(result_line))

        return {
            "code": stock_code,
            "name": stock_name,
            "qualified": analysis["qualified"],
            "analysis": analysis
        }

    def analyze_hot_stocks(self, top_n: int = 20) -> Tuple[List[Dict], str]:
        """分析热股榜股票并返回结果HTML"""
        print(f"\n{Fore.CYAN}=== 开始分析热股榜前{top_n}只股票 ==={Style.RESET_ALL}")

        # 获取热股榜数据
        hot_stocks = self.get_hot_stocks(top_n)
        if not hot_stocks:
            print(f"{Fore.RED}无法获取热股榜数据{Style.RESET_ALL}")
            return [], ""

        # 数据清洗和转换
        processed_stocks = []
        for stock in hot_stocks:
            try:
                # 处理涨跌幅字段
                change_rate = stock['CHANGE_RATE']
                if isinstance(change_rate, str):
                    if change_rate == '-':  # 停牌股票
                        change_rate = 0.0
                    else:
                        change_rate = float(change_rate.replace('%', ''))
                stock['CHANGE_RATE'] = change_rate

                # 处理价格字段
                new_price = stock['NEW_PRICE']
                if isinstance(new_price, str):
                    if new_price == '-':  # 停牌股票
                        new_price = 0.0
                stock['NEW_PRICE'] = new_price

                processed_stocks.append(stock)
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ 股票{stock.get('SECURITY_CODE', '未知')}数据异常: {e}{Style.RESET_ALL}")
                continue

        # 统计分析
        up_stocks = [s for s in processed_stocks if s['CHANGE_RATE'] > 0]
        if up_stocks:
            avg_up = np.mean([s['CHANGE_RATE'] for s in up_stocks])
            strong_up = len([s for s in up_stocks if s['CHANGE_RATE'] > 5]) / len(up_stocks)

            print(f"上涨股票占比: {len(up_stocks) / len(processed_stocks):.1%}")
            print(f"平均涨幅: {avg_up:.2f}%")
            print(f"大涨(>5%)比例: {strong_up:.1%}")
        else:
            print(f"{Fore.YELLOW}⚠️ 当前无上涨股票{Style.RESET_ALL}")

        qualified_stocks = []  # 符合条件的股票列表
        table = PrettyTable()  # 创建美观的表格
        table.field_names = [
            "排名", "代码", "名称", "当前价", "涨跌", "涨幅",
            "大涨日", "调整天数", "量比", "近均线", "拟合得分", "结果"
        ]
        table.align = "r"  # 右对齐数字列
        table.align["名称"] = "l"  # 左对齐名称列

        # 创建HTML表格
        html_table = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>热股榜分析结果 (前{top_n}只股票)</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 100%;
                    overflow-x: auto;
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #4a6fa5;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .up {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .down {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .qualified {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .unqualified {{
                    color: #e74c3c;
                }}
                .highlight {{
                    font-weight: bold;
                }}
                @media screen and (max-width: 600px) {{
                    table {{
                        font-size: 12px;
                    }}
                    th, td {{
                        padding: 5px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>热股榜分析结果 (前{top_n}只股票)</h1>
                <table>
                    <thead>
                        <tr>
                            <th>排名</th>
                            <th>代码</th>
                            <th>名称</th>
                            <th>当前价</th>
                            <th>涨跌</th>
                            <th>涨幅</th>
                            <th>大涨日</th>
                            <th>调整天数</th>
                            <th>量比</th>
                            <th>近均线</th>
                            <th>拟合得分</th>
                            <th>结果</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # 统计变量
        up_count = 0  # 上涨股票数
        down_count = 0  # 下跌股票数
        no_change_count = 0  # 平盘股票数

        for i, stock in enumerate(hot_stocks, 1):
            stock_code = stock["SECURITY_CODE"]
            stock_name = stock["SECURITY_NAME_ABBR"]
            current_price = stock["NEW_PRICE"]
            change_rate = stock["CHANGE_RATE"]

            # 计算价格变化
            try:
                price_change = current_price - (current_price / (1 + change_rate / 100)) if isinstance(change_rate,
                                                                                                       (int,
                                                                                                        float)) else 0
            except:
                price_change = 0

            # 统计涨跌情况
            if isinstance(change_rate, str) and change_rate == '-':
                no_change_count += 1
            elif isinstance(change_rate, (int, float)):
                if change_rate > 0:
                    up_count += 1
                else:
                    down_count += 1

            print(f"\n{Fore.YELLOW}[{i}/{top_n}] 分析 {stock_code} {stock_name}...{Style.RESET_ALL}")
            print(f"  当前价: {current_price:.2f}")
            print(f"  涨跌幅: {change_rate:.2f}%" if isinstance(change_rate, (int, float)) else "  涨跌幅: -")

            # 获取K线数据
            klines = self.get_daily_kline(stock_code)
            if not klines:
                print(f"{Fore.RED}  无法获取K线数据{Style.RESET_ALL}")
                # 添加表格行
                table.add_row([
                    i, stock_code, stock_name, current_price,
                    f"{price_change:.2f}" if isinstance(change_rate, (int, float)) else '-',
                    f"{change_rate:.2f}%" if isinstance(change_rate, (int, float)) else '-',
                    "-", "-", "-", "-",
                    f"{Fore.RED}无数据{Style.RESET_ALL}"
                ])

                # 添加HTML行
                html_table += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{stock_code}</td>
                        <td>{stock_name}</td>
                        <td>{current_price:.2f}</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td style="color: #e74c3c;">无数据</td>
                        <td>-</td>
                    </tr>
                """
                continue

            try:
                # 执行策略分析
                analysis = self.analyze_strategy_two(klines)

                # 打印分析结果
                if analysis["qualified"]:
                    print(f"  结论: {Fore.GREEN}✅ 符合条件{Style.RESET_ALL}")
                    qualified_stocks.append({
                        "rank": i,
                        "code": stock_code,
                        "name": stock_name,
                        "price": current_price,
                        "change": price_change,
                        "change_rate": change_rate,
                        "analysis": analysis
                    })
                else:
                    print(f"  结论: {Fore.RED}❌ 不符合条件 - {analysis['reason']}{Style.RESET_ALL}")

                # 添加表格行
                change_class = "up" if isinstance(change_rate,
                                                  (int, float)) and change_rate > 0 else "down" if isinstance(
                    change_rate, (int, float)) and change_rate < 0 else ""
                result_class = "qualified" if analysis["qualified"] else "unqualified"

                peak_day = analysis.get('peak_day', '无')
                adjust_days = analysis.get('adjust_days', 0)
                volume_ratio = analysis.get('current_volume_ratio', 0)
                near_ma = analysis.get('near_ma', '无')
                ma_score = analysis.get('ma_score', 0)

                # 设置分数颜色
                ma_score_class = ""
                if isinstance(ma_score, (int, float)):
                    if ma_score > 8:
                        ma_score_class = "highlight up"
                    elif ma_score > 5:
                        ma_score_class = "highlight"

                # 格式化显示
                change_rate_display = f"{change_rate:.2f}%" if isinstance(change_rate, (int, float)) else "-"
                price_change_display = f"{price_change:.2f}" if isinstance(change_rate, (int, float)) else "-"
                ma_score_display = f"{ma_score:.2f}" if isinstance(ma_score, (int, float)) else "无"

                table.add_row([
                    i,
                    stock_code,
                    stock_name,
                    current_price,
                    f"{Fore.RED if price_change > 0 else Fore.GREEN}{price_change:.2f}{Style.RESET_ALL}",
                    f"{Fore.RED if (isinstance(change_rate, (int, float)) and change_rate > 0) else Fore.GREEN}{change_rate_display}{Style.RESET_ALL}",
                    f"{Style.BRIGHT if 'peak_day' in analysis and analysis['peak_day'] else ''}{Fore.RED if 'peak_day' in analysis and analysis['peak_day'] else Fore.GREEN}{peak_day}{Style.RESET_ALL}",
                    f"{Style.BRIGHT if (analysis.get('adjust_days', 0) >= self.MIN_CONSECUTIVE_DAYS) else ''}{Fore.RED if (analysis.get('adjust_days', 0) >= self.MIN_CONSECUTIVE_DAYS) else Fore.GREEN}{adjust_days}{Style.RESET_ALL}",
                    f"{Style.BRIGHT if (analysis.get('current_volume_ratio', 0) < self.volume_threshold and analysis.get('current_volume_ratio', 0) > 0) else ''}{Fore.RED if (analysis.get('current_volume_ratio', 0) < self.volume_threshold and analysis.get('current_volume_ratio', 0) > 0) else Fore.YELLOW if analysis.get('current_volume_ratio', 0) < 0.5 else Fore.GREEN}{volume_ratio:.2f}{Style.RESET_ALL}",
                    f"{Style.BRIGHT if analysis.get('is_near_ma', False) else ''}{Fore.RED if analysis.get('is_near_ma', False) else Fore.GREEN}{near_ma}{Style.RESET_ALL}",
                    f"{Style.BRIGHT if (analysis.get('ma_score', 0) > self.MIN_MA_SCORE) else ''}{Fore.RED if (analysis.get('ma_score', 0) > self.MIN_MA_SCORE) else Fore.GREEN}{ma_score_display}{Style.RESET_ALL}",
                    f"{Style.BRIGHT if analysis['qualified'] else ''}{Fore.RED if analysis['qualified'] else Fore.GREEN}{'符合' if analysis['qualified'] else '不符合'}{Style.RESET_ALL}",
                ])

                # 添加HTML行
                html_table += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{stock_code}</td>
                        <td>{stock_name}</td>
                        <td>{current_price:.2f}</td>
                        <td class="{change_class}">{price_change_display}</td>
                        <td class="{change_class}">{change_rate_display}</td>
                        <td>{peak_day}</td>
                        <td>{adjust_days}</td>
                        <td>{volume_ratio:.2f}</td>
                        <td>{near_ma}</td>
                        <td class="{ma_score_class}">{ma_score_display}</td>
                        <td class="{result_class}">{"符合" if analysis["qualified"] else "不符合"}</td>
                    </tr>
                """

            except Exception as e:
                print(f"{Fore.RED}分析股票 {stock_code} 时出错: {e}{Style.RESET_ALL}")
                # 添加表格行
                table.add_row([
                    i, stock_code, stock_name, current_price,
                    f"{price_change:.2f}" if isinstance(change_rate, (int, float)) else '-',
                    f"{change_rate:.2f}%" if isinstance(change_rate, (int, float)) else '-',
                    "-", "-", "-", "-",
                    f"{Fore.RED}分析错误{Style.RESET_ALL}"
                ])

                # 添加HTML行
                html_table += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{stock_code}</td>
                        <td>{stock_name}</td>
                        <td>{current_price:.2f}</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td style="color: #e74c3c;">分析错误</td>
                        <td>-</td>
                    </tr>
                """
                continue

        # 完成HTML表格
        html_table += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """

        # 打印汇总信息
        print(f"\n{Fore.CYAN}=== 热股榜分析结果 ==={Style.RESET_ALL}")
        print(table)

        print(f"\n{Fore.MAGENTA}=== 涨跌统计 ==={Style.RESET_ALL}")
        print(f"上涨股票数: {Fore.RED}{up_count}{Style.RESET_ALL}")
        print(f"下跌股票数: {Fore.GREEN}{down_count}{Style.RESET_ALL}")
        if no_change_count > 0:
            print(f"平盘股票数: {Fore.YELLOW}{no_change_count}{Style.RESET_ALL}")

        if hot_stocks:
            print(f"上涨比例: {Fore.RED}{(up_count / len(hot_stocks)) * 100:.1f}%{Style.RESET_ALL}")
            print(f"下跌比例: {Fore.GREEN}{(down_count / len(hot_stocks)) * 100:.1f}%{Style.RESET_ALL}")

        print(f"\n找到 {len(qualified_stocks)} 只符合放量大涨后缩量调整条件的股票")

        # 打印符合条件的股票详情
        if qualified_stocks:
            print(f"\n{Fore.CYAN}=== 符合条件的股票详情 ==={Style.RESET_ALL}")
            for stock in qualified_stocks:
                analysis = stock["analysis"]
                print(f"\n{Fore.GREEN}✅ {stock['code']} {stock['name']}{Style.RESET_ALL}")
                print(f"  排名: {stock['rank']}")
                print(f"  当前价: {stock['price']}")
                print(f"  涨跌: {Fore.RED if stock['change'] > 0 else Fore.GREEN}"
                      f"{stock['change']:.2f}({stock['change_rate']:.2f}%){Style.RESET_ALL}")
                print(f"  放量大涨日: {analysis['peak_day']} (价格: {analysis['peak_price']:.2f})")
                print(f"  调整天数: {analysis['adjust_days']}天, 量比: {analysis['current_volume_ratio']:.2f}")
                print(f"  接近均线: {analysis['near_ma']}")

        # 保存结果到文件
        if qualified_stocks:
            filename = f"strategy_two_qualified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(qualified_stocks, f, ensure_ascii=False, indent=4)
            print(f"\n{Fore.CYAN}💾 分析结果已保存到: {filename}{Style.RESET_ALL}")

        return qualified_stocks, html_table

    def continuous_monitoring(self):
        """
        持续监控热股榜（增强日志版）
        采用两轮筛选机制：
        1. 第一轮：快速筛选至少符合2项条件的股票
        2. 第二轮：对候选股票进行严格全条件检查
        """
        print(f"\n{Fore.CYAN}=== 启动热股榜智能监控 ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}按Ctrl+C停止 | 每5秒自动刷新{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}筛选条件：放量大涨+缩量调整+均线聚合{Style.RESET_ALL}")

        # 初始化统计数据
        stats = {
            'total_cycles': 0,  # 总循环次数
            'total_candidates': 0,  # 总候选股数
            'total_qualified': 0,  # 总合格股数
            'best_stock': {'code': None, 'score': 0}  # 最佳股票记录
        }

        while True:
            stats['total_cycles'] += 1
            current_time = datetime.now().strftime('%H:%M:%S')

            try:
                print(f"\n{Fore.BLUE}➤ 第{stats['total_cycles']}轮检测 [{current_time}]{Style.RESET_ALL}")

                # 获取热股榜数据
                print(f"{Fore.WHITE}⌛ 获取热股榜数据...{Style.RESET_ALL}")
                hot_stocks = self.get_hot_stocks(500)
                if not hot_stocks:
                    print(f"{Fore.YELLOW}⚠️ 获取热股榜失败，5秒后重试{Style.RESET_ALL}")
                    time.sleep(5)
                    continue
                print(f"✅ 获取到{len(hot_stocks)}只股票 | 最新：{hot_stocks[0]['SECURITY_NAME_ABBR']}")

                # 第一轮：快速筛选（至少符合2项条件）
                print(f"{Fore.WHITE}🔍 第一轮快速筛选（至少2项条件）...{Style.RESET_ALL}")
                candidates = []
                for stock in hot_stocks[:100]:  # 测试时只检查前100只
                    stock_code = stock["SECURITY_CODE"]

                    try:
                        klines = self.get_daily_kline(stock_code)
                        if not klines or len(klines) < 20:
                            continue

                        # 条件检查
                        conditions = []

                        # 条件1: 放量大涨
                        has_big_rise, peak_day = self.is_big_rise_with_volume(klines)
                        conditions.append(1 if has_big_rise else 0)

                        # 条件2: 缩量调整
                        if has_big_rise:
                            is_shrink, adjust_days, volume_ratio, _ = self.is_shrink_adjustment(klines, peak_day)
                            conditions.append(1 if (is_shrink and adjust_days >= self.MIN_CONSECUTIVE_DAYS) else 0)
                        else:
                            conditions.append(0)

                        # 条件3: 均线聚合
                        _, _, ma_score = self.is_near_key_ma(klines)
                        conditions.append(1 if ma_score > self.MIN_MA_SCORE else 0)

                        # 记录候选股
                        if sum(conditions) >= 2:
                            candidates.append({
                                "stock": stock,
                                "klines": klines,
                                "conditions": conditions,
                                "log": f"放量:{conditions[0]} | 缩量:{conditions[1]} | 均线:{conditions[2]}"
                            })
                            print(f"  🟢 候选 {stock_code} {stock['SECURITY_NAME_ABBR']} | {conditions}")

                    except Exception as e:
                        print(f"{Fore.RED}  ❗ {stock_code}分析异常: {str(e)[:30]}...{Style.RESET_ALL}")

                # 第二轮：精确验证
                print(f"\n{Fore.WHITE}🔎 第二轮精确验证（{len(candidates)}只候选）...{Style.RESET_ALL}")
                qualified = []
                for cand in candidates:
                    stock = cand["stock"]
                    try:
                        # 严格验证所有条件
                        valid = all([
                            self.is_big_rise_with_volume(cand["klines"])[0],
                            (shr := self.is_shrink_adjustment(cand["klines"], cand["klines"][-1]))[
                                1] >= self.MIN_CONSECUTIVE_DAYS,
                            shr[2] < 0.4,
                            self.is_near_key_ma(cand["klines"])[2] > self.MIN_MA_SCORE
                        ])

                        if valid:
                            qualified.append(stock["SECURITY_CODE"])
                            score = sum(cand["conditions"])
                            print(f"  🎯 合格 {stock['SECURITY_CODE']} {stock['SECURITY_NAME_ABBR']} | 得分:{score}")

                            # 更新最佳股票记录
                            if score > stats['best_stock']['score']:
                                stats['best_stock'] = {
                                    'code': stock["SECURITY_CODE"],
                                    'name': stock["SECURITY_NAME_ABBR"],
                                    'score': score
                                }
                    except Exception as e:
                        print(f"{Fore.RED}  ❗ {stock['SECURITY_CODE']}验证异常: {e}{Style.RESET_ALL}")

                # 更新统计数据
                stats['total_candidates'] += len(candidates)
                stats['total_qualified'] += len(qualified)

                # 结果展示
                print(f"\n{Fore.CYAN}📊 本轮结果{Style.RESET_ALL}")
                print(f"┌{'─' * 30}┐")
                print(f"│ 初选候选股: {Fore.YELLOW}{len(candidates)}只{Style.RESET_ALL}")
                print(f"│ 终选合格股: {Fore.GREEN if qualified else Fore.RED}{len(qualified)}只{Style.RESET_ALL}")
                print(f"└{'─' * 30}┘")

                if candidates:
                    # 打印候选股条件分布
                    cond_counts = {
                        '放量': sum(1 for c in candidates if c['conditions'][0]),
                        '缩量': sum(1 for c in candidates if c['conditions'][1]),
                        '均线': sum(1 for c in candidates if c['conditions'][2])
                    }
                    print(f"\n{Fore.MAGENTA}📈 候选股条件分布:{Style.RESET_ALL}")
                    for k, v in cond_counts.items():
                        print(f"  {k}: {v}/{len(candidates)} ({v / len(candidates):.0%})")

                if qualified:
                    print(f"\n{Fore.GREEN}🏆 合格股列表:{Style.RESET_ALL}")
                    for code in qualified:
                        stock = next(s for s in hot_stocks if s["SECURITY_CODE"] == code)
                        print(f"  • {code} {stock['SECURITY_NAME_ABBR']}")
                else:
                    print(f"\n{Fore.YELLOW}⚠️ 本轮无合格股票{Style.RESET_ALL}")

                # 全局统计
                print(f"\n{Fore.BLUE}🌐 累计统计（{stats['total_cycles']}轮）:{Style.RESET_ALL}")
                print(f"├ 总候选股: {stats['total_candidates']}")
                print(f"├ 总合格股: {stats['total_qualified']}")
                print(f"└ 最佳股票: {stats['best_stock']['code']} (得分:{stats['best_stock']['score']})")

                print(f"\n{Fore.YELLOW}⏳ 5秒后重新扫描...{Style.RESET_ALL}")
                time.sleep(300)

            except KeyboardInterrupt:
                print(f"\n{Fore.CYAN}🛑 监控终止 | 共运行{stats['total_cycles']}轮{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}💥 全局异常: {e}{Style.RESET_ALL}")
                time.sleep(300)

    def execute_followup_action(self, stock_code: str, klines: List[Dict]):
        """
        执行后续操作（由您填充具体逻辑）

        参数:
            stock_code: 股票代码
            klines: K线数据
        """
        # 这里放置您需要实现的后续逻辑
        pass


if __name__ == "__main__":
    analyzer = StrategyTwoAnalyzer()

    # 打印工具介绍
    print(f"{Fore.CYAN}=== 放量大涨后缩量调整策略分析工具 ===")
    print("特点:")
    print("1. 寻找近期有放量大涨(>7%)的股票")
    print("2. 随后成交量萎缩至高峰期的35%以下")
    print("3. 价格回调幅度不超过18%")
    print("4. 股价回调至10日或20日均线附近")
    print(f"============================={Style.RESET_ALL}\n")

    # 主循环菜单
    while True:
        print("\n选择操作:")
        print("1. 分析单只股票")
        print("2. 扫描热股榜")
        print("3. 启动持续监控(500只热股)")
        print("4. 查看帮助")
        print("q. 退出")

        choice = input("请输入选择(1/2/3/4/q): ").strip().lower()

        if choice == 'q':
            break

        if choice == '1':
            stock_code = input("请输入股票代码(如600000): ").strip()
            if not stock_code.isdigit() or len(stock_code) != 6:
                print(f"{Fore.RED}股票代码应为6位数字{Style.RESET_ALL}")
                continue
            analyzer.analyze_single_stock(stock_code)
        elif choice == '2':
            top_n = input("请输入要分析的热股数量(默认20): ").strip()
            try:
                top_n = int(top_n) if top_n else 20
                analyzer.analyze_hot_stocks(top_n)
            except ValueError:
                print(f"{Fore.RED}请输入有效的数字{Style.RESET_ALL}")
        elif choice == '3':
            print(f"\n{Fore.YELLOW}启动自动循环监控(每5秒刷新)...{Style.RESET_ALL}")
            analyzer.continuous_monitoring()
        elif choice == '4':
            print(f"\n{Fore.YELLOW}=== 使用帮助 ===")
            print("1. 分析单只股票: 输入6位股票代码")
            print("2. 扫描热股榜: 分析东财热股榜前N只股票")
            print("3. 持续监控: 每N分钟自动扫描500只热股")
            print("4. 结果会保存在当前目录下的json文件中")
            print(f"================{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}无效的选择，请重新输入{Style.RESET_ALL}")
