import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from colorama import Fore, Style, init
import numpy as np
from prettytable import PrettyTable
import time
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
import traceback
import math
import numpy as np
from typing import List, Dict, Tuple
# from wxauto import WeChat
# 初始化colorama
init(autoreset=True)


class StrategyTwoAnalyzer:
    """策略类型2分析器：放量大涨后缩量调整至极致"""

    def __init__(self):
        # 参数配置
        self.min_change_rate = 7  # 最小涨幅阈值
        self.volume_threshold = 0.35  # 成交量萎缩至高峰期的35%以下
        self.price_drop_threshold = 0.18  # 价格回调幅度不超过18%
        self.ABSOLUTE_SHRINK_THRESHOLD = 0.4 # 绝对萎缩幅度不超过40%
        self.TEMPORARY_INCREASE_ALLOWANCE = 0.2
        self.MIN_CONSECUTIVE_DAYS = 8 # 最少缩量天数（非连续）
        self.MIN_MA_SCORE = 8  # 最少拟合分数
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Referer": "https://emrnweb.eastmoney.com/"
        }

        # 创建输出目录
        if not os.path.exists("strategy_two_plots"):
            os.makedirs("strategy_two_plots")

    def parse_kline(self, kline_str: str) -> Dict:
        """解析K线数据字符串为字典"""
        fields = kline_str.split(",")
        return {
            "date": fields[0],#日期
            "open": float(fields[1]),#开盘价
            "close": float(fields[2]),#收盘价
            "high": float(fields[3]),#最高价
            "low": float(fields[4]),#最低价
            "volume": float(fields[5]),#成交量
            "amount": float(fields[6]),#成交额
            "amplitude": float(fields[7]),#振幅
            "change_rate": float(fields[8]),# 涨幅
            "change_amount": float(fields[9]), #成交额
            "turnover": float(fields[10]) if len(fields) > 10 else 0, #换手率
            "ma5":  0,
            "ma10":  0,
            "ma20":  0
        }

    def get_daily_kline(self, stock_code: str, years: float = 0.5) -> Optional[List[Dict]]:
        """获取日K线数据（带重试和间隔控制）"""
        max_retries = 3  # 最大重试次数
        retry_delay = 2  # 重试间隔（秒）

        for attempt in range(max_retries):
            try:
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=int(365 * years))).strftime('%Y%m%d')

                params = {
                    "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                    "beg": start_date,
                    "end": end_date,
                    "ut": "fa5fd1943c7b386f172d6893dbfba10b",
                    "rtntype": "6",
                    "secid": f"1.{stock_code}" if stock_code.startswith('6') else f"0.{stock_code}",
                    "klt": "101",  # 日线
                    "fqt": "1"
                }

                url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                response.raise_for_status()

                json_str = response.text.strip()
                if json_str.startswith("jsonp") and json_str.endswith(")"):
                    json_str = json_str[json_str.index("(") + 1:-1]

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
                if attempt < max_retries - 1:  # 不是最后一次重试
                    print(
                        f"{Fore.YELLOW}获取日K线失败({stock_code})，第 {attempt + 1}/{max_retries} 次重试...{Style.RESET_ALL}")
                    time.sleep(retry_delay)  # 等待间隔
                    continue
                print(f"{Fore.RED}获取日K线失败({stock_code}): {e}{Style.RESET_ALL}")
                return None

    def get_stock_name(self, stock_code: str) -> str:
        """获取股票名称"""
        url = "https://emrnweb.eastmoney.com/api/security/quote"
        params = {
            "secids": f"1.{stock_code}" if stock_code.startswith('6') else f"0.{stock_code}"
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            data = response.json()
            return data["data"][0]["f14"]
        except:
            return "未知"

    def get_hot_stocks(self, top_n: int = 100) -> Optional[List[Dict]]:
        """获取热股榜数据"""
        url = "https://datacenter.eastmoney.com/stock/selection/api/data/get/"

        multipart_data = MultipartEncoder(
            fields={
                "type": "RPTA_SECURITY_STOCKSELECT",
                "sty": "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,NEW_PRICE,CHANGE_RATE,TOTAL_MARKET_CAP,POPULARITY_RANK",
                "filter": "(@LISTING_DATE=\"OVER1Y\")(TOTAL_MARKET_CAP<15000000000)(POPULARITY_RANK>0)(POPULARITY_RANK<=1000)(HOLDER_NEWEST>0)(HOLDER_NEWEST<=40000)",
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
                return result["result"]["data"][::-1]

        except Exception as e:
            print(f"{Fore.RED}获取热股榜失败: {e}{Style.RESET_ALL}")

        return None

    def is_big_rise_with_volume(self, klines: List[Dict], lookback_days: int = 20, min_retrace_ratio: float = 0.9) -> \
    Tuple[bool, Optional[Dict]]:
        """
        检查放量大涨日（需同时满足）：
        1. 涨幅 > min_change_rate（默认7%）
        2. 成交量显著放大（满足任一）：
           - 是前一日2倍以上
           - 是5日均量1.5倍以上
           - 是20日均量1.2倍以上
        3. 收盘涨幅/最高涨幅 ≥ min_retrace_ratio（默认90%）
           - 最高涨幅a = (最高价 - 开盘价) / 开盘价
           - 收盘涨幅b = (收盘价 - 开盘价) / 开盘价
           - 要求 b/a ≥ min_retrace_ratio

        参数：
        min_retrace_ratio: 收盘涨幅相对最高涨幅的最小保留比例（默认0.9即90%）
        """
        # 参数校验
        if not klines or len(klines) < lookback_days + 1 or min_retrace_ratio <= 0:
            return False, None

        recent_klines = klines[-lookback_days - 1:]

        for i in range(1, len(recent_klines)):
            try:
                current = recent_klines[i]
                prev = recent_klines[i - 1]

                # 必须存在的字段检查
                required_fields = ['open', 'close', 'high', 'volume', 'change_rate']
                if any(field not in current for field in required_fields):
                    continue

                # 数据类型转换
                open_price = float(current['open'])
                close_price = float(current['close'])
                high_price = float(current['high'])
                volume = float(current['volume'])
                prev_volume = float(prev['volume'])
                change_rate = float(current['change_rate'])

                # 条件1：基础涨幅检查
                if change_rate <= self.min_change_rate:
                    continue

                # 条件3：涨幅保留率检查（新核心逻辑）
                max_rise = (high_price - open_price) / open_price  # 最高涨幅a
                close_rise = (close_price - open_price) / open_price  # 收盘涨幅b

                # 处理零除问题（如开盘=最高价）
                if max_rise <= 1e-6:  # 浮点数精度处理
                    continue

                if (close_rise / max_rise) < min_retrace_ratio:
                    continue

                # 条件2：成交量放大检查
                lookback_start = max(0, i - 19)
                volumes = [float(k['volume']) for k in recent_klines[lookback_start:i + 1] if 'volume' in k]

                ma5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
                ma20 = np.mean(volumes) if len(volumes) >= 20 else 0

                volume_expanded = (
                        volume > prev_volume * 2 or
                        (ma5 > 0 and volume > ma5 * 1.5) or
                        (ma20 > 0 and volume > ma20 * 1.2)
                )

                if volume_expanded:
                    return True, current

            except (TypeError, ValueError) as e:
                continue

        return False, None

    def is_shrink_adjustment(self, klines: List[Dict], peak_day: Dict) -> Tuple[bool, int, float]:
        """
        改进版缩量调整判断
        :param klines: K线数据列表
        :param peak_day: 峰值日数据
        :return: (是否满足缩量调整条件, 有效天数, 最终成交量比例)
        """
        if len(klines) < 5:
            return False, 0, 0, 0


        peak_volume = peak_day["volume"]
        peak_price = peak_day["close"]
        peak_date = peak_day["date"]

        # 找到大涨日位置
        peak_index = next((i for i, k in enumerate(klines) if k["date"] == peak_date), -1)

        if peak_index == -1 or peak_index >= len(klines) - 3:
            return False, 0, 0, 0

        effective_days = 0
        volume_ma5 = []
        end_index = peak_index  # 初始化结束索引

        for i in range(peak_index + 1, len(klines)):
            current = klines[i]
            prev = klines[i - 1]
            current_ratio = current["volume"] / peak_volume
            price_drop = (peak_price - current["close"]) / peak_price

            # 计算5日均量
            if i >= peak_index + 5:
                ma5 = np.mean([k["volume"] for k in klines[i - 4:i + 1]])
                volume_ma5.append(ma5)

            # 终止条件
            if price_drop > self.price_drop_threshold:
                end_index = i - 1  # 记录结束位置
                break

            if current["volume"] > peak_volume * self.volume_threshold:
                continue  # 跳过不符合条件的交易日
            else:
                effective_days += 1
                end_index = i  # 更新结束位置

        # 获取最后一天的成交量
        if end_index >= len(klines):
            end_index = len(klines) - 1

        last_day = klines[-1]
        final_ratio = last_day["volume"] / peak_volume

        # 计算当前价格降幅（即使没有超过阈值也计算）
        current_price_drop = (peak_price - last_day["close"]) / peak_price

        # 计算5日均量是否下降
        ma5_decreasing = len(volume_ma5) < 2 or volume_ma5[-1] < volume_ma5[0] * 0.8

        qualified = (effective_days >= self.MIN_CONSECUTIVE_DAYS and
                     final_ratio < self.ABSOLUTE_SHRINK_THRESHOLD
                     # and ma5_decreasing  # 可选条件
                     )


        return qualified, effective_days, final_ratio, current_price_drop

    def calculate_ma(self, klines: List[Dict], days: int) -> float:
        """计算指定天数的均线值（带缓存检查）"""
        if len(klines) < days:
            return 0.0

        # 如果最新数据已计算过ma值，直接返回
        if f"ma{days}" in klines[-1] and klines[-1][f"ma{days}"] > 0:
            return klines[-1][f"ma{days}"]

        closes = [k["close"] for k in klines[-days:]]
        return sum(closes) / days

    def calculate_convergence_score(self, ma_values: np.ndarray) -> float:
        """
        基于变异系数(CV)计算均线聚合分数 (0-10)
        分数越高表示均线聚合度越好
        """
        cv = np.std(ma_values) / np.mean(ma_values)  # 计算变异系数

        # 科学分段评分标准（需根据实际数据分布校准）
        cv_perfect = 0.005  # MA5/10/20差异<0.5% (视觉上几乎重合)
        cv_good = 0.015  # MA5/10/20差异<1.5% (视觉上轻度分散)
        cv_poor = 0.03  # MA5/10/20差异>3% (视觉上明显分散)

        if cv <= cv_perfect:
            return 10.0
        elif cv <= cv_good:
            # 线性映射：cv_perfect→10分, cv_good→7分
            return 10 - 3 * (cv - cv_perfect) / (cv_good - cv_perfect)
        elif cv <= cv_poor:
            # 线性映射：cv_good→7分, cv_poor→3分
            return 7 - 4 * (cv - cv_good) / (cv_poor - cv_good)
        else:
            # cv > cv_poor → 0-3分
            return max(0, 3 - 3 * (cv - cv_poor) / cv_poor)

    def is_near_key_ma(self, klines: List[Dict], adjustment_days: int = 0) -> Tuple[bool, str, float]:
        """
        严格判断股价是否回调至聚合均线附近
        :param klines: K线数据，需包含close,open,high,low,volume
        :param adjustment_days: 数据调整天数
        :return: (是否在关键均线附近, 触及的均线名称, 聚合分数0-10)
        """
        # 数据校验
        if len(klines) < 20 + adjustment_days:
            return False, "", 0.0

        # 计算三条关键均线
        ma5 = self.calculate_ma(klines, 5)  # 使用self调用类方法
        ma10 = self.calculate_ma(klines, 10)
        ma20 = self.calculate_ma(klines, 20)

        # 检查数据有效性
        if ma5 <= 0 or ma10 <= 0 or ma20 <= 0:
            return False, "", 0.0

        # 计算标准化聚合分数
        ma_values = np.array([ma5, ma10, ma20])
        convergence_score = self.calculate_convergence_score(ma_values)  # 使用self调用类方法

        # 动态聚合阈值（建议通过历史数据70分位校准）
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
        分析股票是否符合策略类型2的条件
        """
        if not klines or len(klines) < 10:
            return {
                "qualified": False,
                "reason": "数据不足(需要至少10天数据)"
            }

        # 条件1: 放量大涨
        has_big_rise, peak_day = self.is_big_rise_with_volume(klines)
        if not has_big_rise:
            return {
                "qualified": False,
                "reason": "无放量大涨日(涨幅>7%且成交量显著放大且不回落)"
            }

        # 条件2: 缩量调整
        is_shrink, adjust_days, volume_ratio, current_price_drop = self.is_shrink_adjustment(klines, peak_day)

        # 条件3: 回调至均线附近
        is_near_ma, ma_type, ma_score = self.is_near_key_ma(klines, adjust_days)

        # 综合判断条件
        qualified = has_big_rise and adjust_days >= self.MIN_CONSECUTIVE_DAYS and is_near_ma and ma_score >= self.MIN_MA_SCORE

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

        stock_name = self.get_stock_name(stock_code)
        klines = self.get_daily_kline(stock_code)

        if not klines:
            print(f"{Fore.RED}无法获取股票 {stock_code} 的K线数据{Style.RESET_ALL}")
            return None

        analysis = self.analyze_strategy_two(klines)

        # 打印在一行的分析结果
        print(f"\n{Fore.CYAN}=== 分析结果 ==={Style.RESET_ALL}")
        result_line = [
            f"股票: {Fore.YELLOW}{stock_code} {stock_name}{Style.RESET_ALL}",
            f"结论: {Fore.GREEN if analysis['qualified'] else Fore.RED}{'符合' if analysis['qualified'] else '不符合'}条件",
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

    def analyze_hot_stocks(self, top_n: int = 20) -> List[Dict]:
        """分析热股榜股票"""
        print(f"\n{Fore.CYAN}=== 开始分析热股榜前{top_n}只股票 ==={Style.RESET_ALL}")

        hot_stocks = self.get_hot_stocks(top_n)
        if not hot_stocks:
            print(f"{Fore.RED}无法获取热股榜数据{Style.RESET_ALL}")
            return []

        qualified_stocks = []
        table = PrettyTable()
        table.field_names = [
            "排名", "代码", "名称", "当前价", "涨跌", "涨幅",
            "大涨日", "调整天数", "量比", "近均线", "拟合得分", "结果"
        ]
        table.align = "r"
        table.align["名称"] = "l"

        # 新增统计变量
        up_count = 0
        down_count = 0
        no_change_count = 0

        for i, stock in enumerate(hot_stocks, 1):
            stock_code = stock["SECURITY_CODE"]
            stock_name = stock["SECURITY_NAME_ABBR"]
            current_price = stock["NEW_PRICE"]
            change_rate = stock["CHANGE_RATE"]
            price_change = current_price - (current_price / (1 + change_rate / 100)) if change_rate != '-' else 0

            # 统计涨跌情况
            if change_rate == '-':
                no_change_count += 1
            elif change_rate > 0:
                up_count += 1
            else:
                down_count += 1

            print(f"\n{Fore.YELLOW}[{i}/{top_n}] 分析 {stock_code} {stock_name}...{Style.RESET_ALL}")

            klines = self.get_daily_kline(stock_code)
            if not klines:
                print(f"{Fore.RED}无法获取K线数据{Style.RESET_ALL}")
                table.add_row([
                    i, stock_code, stock_name, current_price,
                    f"{price_change:.2f}" if change_rate != '-' else '-',
                    f"{change_rate:.2f}%" if change_rate != '-' else '-',
                    "-", "-", "-", "-",
                    f"{Fore.RED}无数据{Style.RESET_ALL}"
                ])
                continue

            try:
                analysis = self.analyze_strategy_two(klines)

                if analysis["qualified"]:
                    print(f"  结论: {Fore.RED}✅ 符合条件{Style.RESET_ALL}")
                else:
                    print(f"  结论: {Fore.GREEN}❌ 不符合条件 - {analysis['reason']}{Style.RESET_ALL}")

                result_parts = []
                if "peak_day" in analysis:
                    result_parts.append(f"{Fore.RED}大涨日:{analysis['peak_day']}{Style.RESET_ALL}")

                if "adjust_days" in analysis:
                    adjust_color = Fore.RED if (analysis.get("adjust_days", 0) >= self.MIN_CONSECUTIVE_DAYS) else Fore.GREEN
                    result_parts.append(f"{adjust_color}调整:{analysis['adjust_days']}天{Style.RESET_ALL}")

                if "current_volume_ratio" in analysis:
                    ratio_color = Fore.RED if (
                                analysis["current_volume_ratio"] < self.ABSOLUTE_SHRINK_THRESHOLD and analysis[
                            "current_volume_ratio"] != 0) else Fore.GREEN
                    result_parts.append(f"{ratio_color}量比:{analysis['current_volume_ratio']:.2f}{Style.RESET_ALL}")

                if "near_ma" in analysis:
                    ma_color = Fore.RED if analysis.get("is_near_ma", False) else Fore.GREEN
                    result_parts.append(f"{ma_color}均线:{analysis['near_ma']}{Style.RESET_ALL}")

                if "ma_score" in analysis:
                    ma_color = Fore.RED if (analysis.get("ma_score", 0) > self.MIN_MA_SCORE) else Fore.GREEN
                    result_parts.append(f"{ma_color}拟合得分:{analysis['ma_score']:.2f}{Style.RESET_ALL}")

                if result_parts:
                    print("  " + " | ".join(result_parts))

                if analysis["qualified"]:
                    qualified_stocks.append({
                        "rank": i,
                        "code": stock_code,
                        "name": stock_name,
                        "price": current_price,
                        "change": price_change,
                        "change_rate": change_rate,
                        "analysis": analysis
                    })

                # 添加表格行
                table.add_row([
                    i,
                    stock_code,
                    stock_name,
                    current_price,
                    f"{Fore.RED if price_change > 0 else Fore.GREEN}{price_change:.2f}{Style.RESET_ALL}",
                    f"{Fore.RED if (change_rate != '-' and change_rate > 0) else Fore.GREEN}{change_rate}%{Style.RESET_ALL}",
                    f"{Fore.RED if 'peak_day' in analysis and analysis['peak_day'] else Fore.GREEN}{analysis.get('peak_day', '无')}{Style.RESET_ALL}",
                    f"{Fore.RED if (analysis.get('adjust_days', 0) >= self.MIN_CONSECUTIVE_DAYS) else Fore.GREEN}{analysis.get('adjust_days', 0)}{Style.RESET_ALL}",
                    f"{Fore.RED if (analysis.get('current_volume_ratio', 0) < self.volume_threshold and analysis.get('current_volume_ratio', 0) > 0) else Fore.YELLOW if analysis.get('current_volume_ratio', 0) < 0.5 else Fore.GREEN}{analysis.get('current_volume_ratio', 0):.2f}{Style.RESET_ALL}",
                    f"{Fore.RED if analysis.get('is_near_ma', False) else Fore.GREEN}{analysis.get('near_ma', '无')}{Style.RESET_ALL}",
                    f"{Fore.RED if (analysis.get('ma_score', 0) > self.MIN_MA_SCORE) else Fore.GREEN}{analysis.get('ma_score', 0):.2f}{Style.RESET_ALL}",
                    f"{Fore.RED if analysis['qualified'] else Fore.GREEN}{'符合' if analysis['qualified'] else '不符合'}{Style.RESET_ALL}",
                ])

            except Exception as e:
                print(f"{Fore.RED}分析股票 {stock_code} 时出错: {e}{Style.RESET_ALL}")
                table.add_row([
                    i, stock_code, stock_name, current_price,
                    f"{price_change:.2f}" if change_rate != '-' else '-',
                    f"{change_rate:.2f}%" if change_rate != '-' else '-',
                    "-", "-", "-", "-",
                    f"{Fore.RED}分析错误{Style.RESET_ALL}"
                ])
                continue

        # 打印汇总表格
        print(f"\n{Fore.CYAN}=== 热股榜分析结果 ==={Style.RESET_ALL}")
        print(table)

        # 新增：打印涨跌统计
        print(f"\n{Fore.MAGENTA}=== 涨跌统计 ==={Style.RESET_ALL}")
        print(f"上涨股票数: {Fore.RED}{up_count}{Style.RESET_ALL}")
        print(f"下跌股票数: {Fore.GREEN}{down_count}{Style.RESET_ALL}")
        if no_change_count > 0:
            print(f"平盘股票数: {Fore.YELLOW}{no_change_count}{Style.RESET_ALL}")
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
                print(
                    f"  涨跌: {Fore.RED if stock['change'] > 0 else Fore.GREEN}{stock['change']:.2f}({stock['change_rate']:.2f}%){Style.RESET_ALL}")
                print(f"  放量大涨日: {analysis['peak_day']} (价格: {analysis['peak_price']:.2f})")
                print(f"  调整天数: {analysis['adjust_days']}天, 量比: {analysis['current_volume_ratio']:.2f}")
                print(f"  接近均线: {analysis['near_ma']}")

        # 保存结果
        if qualified_stocks:
            filename = f"strategy_two_qualified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(qualified_stocks, f, ensure_ascii=False, indent=4)
            print(f"\n{Fore.CYAN}💾 分析结果已保存到: {filename}{Style.RESET_ALL}")

        return qualified_stocks

    def continuous_monitoring(self):
        """
        持续监控热股榜（增强日志版）
        第一轮：快速筛选至少符合2项条件的股票
        第二轮：对候选股票进行严格全条件检查
        """
        print(f"\n{Fore.CYAN}=== 启动热股榜智能监控 ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}按Ctrl+C停止 | 每5秒自动刷新{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}筛选条件：放量大涨+缩量调整+均线聚合{Style.RESET_ALL}")

        # 初始化统计数据
        stats = {
            'total_cycles': 0,
            'total_candidates': 0,
            'total_qualified': 0,
            'best_stock': {'code': None, 'score': 0}
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

                # 第一轮：快速筛选
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
                        # 严格验证
                        valid = all([
                            self.is_big_rise_with_volume(cand["klines"])[0],
                            (shr := self.is_shrink_adjustment(cand["klines"], cand["klines"][-1]))[1] >= self.MIN_CONSECUTIVE_DAYS,
                            shr[2] < 0.4,
                            self.is_near_key_ma(cand["klines"])[2] > self.MIN_MA_SCORE
                        ])

                        if valid:
                            qualified.append(stock["SECURITY_CODE"])
                            score = sum(cand["conditions"])
                            print(f"  🎯 合格 {stock['SECURITY_CODE']} {stock['SECURITY_NAME_ABBR']} | 得分:{score}")

                            # 更新最佳股票
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
        :param stock_code: 股票代码
        :param klines: K线数据
        """
        # 这里放置您需要实现的后续逻辑
        # wx.send_message(f" {stock_code} ","文件传输助手")

# wx = WeChat()

if __name__ == "__main__":
    analyzer = StrategyTwoAnalyzer()

    print(f"{Fore.CYAN}=== 放量大涨后缩量调整策略分析工具 ===")
    print("特点:")
    print("1. 寻找近期有放量大涨(>7%)的股票")
    print("2. 随后成交量萎缩至高峰期的35%以下")
    print("3. 价格回调幅度不超过18%")
    print("4. 股价回调至10日或20日均线附近")
    print(f"============================={Style.RESET_ALL}\n")

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