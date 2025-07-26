import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from colorama import Fore, Style, init
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import math
from requests_toolbelt.multipart.encoder import MultipartEncoder
from prettytable import PrettyTable
from functools import lru_cache
import time
import traceback
# 初始化colorama
init(autoreset=True)


class StableRiseStockAnalyzer:
    """稳定小碎步上涨股票分析器"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Referer": "https://emrnweb.eastmoney.com/"
        }

        # ========== 关键可调参数 ==========
        # 趋势相关参数
        self.min_slope = 0.0015  # 最小日斜率阈值
        self.min_r_squared = 0.75  # 最小R平方值 (趋势稳定性)
        self.min_days = 30  # 最小分析天数
        self.max_daily_change = 0.1  # 最大单日涨幅(8%)

        # 成交量相关参数
        self.volume_stability_threshold = 0.8  # 成交量稳定性阈值(越小越稳定)
        self.min_avg_volume = 5e6  # 最小平均成交量(手)

        # 风险控制参数
        self.max_drawdown_threshold = -0.2  # 最大回撤阈值(-15%)
        self.min_annual_return = 0.15  # 最小年化收益率要求(15%)
        self.max_pe_ratio = 100  # 最大市盈率

        # 其他参数
        self.blacklist = ['ST', '*ST', '退市']  # 黑名单关键词
        self.plot_enabled = True  # 是否保存趋势图
        # ================================

        # 创建输出目录
        if not os.path.exists("stable_rise_plots"):
            os.makedirs("stable_rise_plots")

    def parse_kline(self, kline_str: str) -> Dict:
        """解析K线数据字符串为字典"""
        fields = kline_str.split(",")
        return {
            "date": fields[0],
            "open": float(fields[1]) if fields[1] else 0.0,
            "close": float(fields[2]) if fields[2] else 0.0,
            "high": float(fields[3]) if fields[3] else 0.0,
            "low": float(fields[4]) if fields[4] else 0.0,
            "volume": float(fields[5]) if fields[5] else 0.0,
            "amount": float(fields[6]) if fields[6] else 0.0,
            "amplitude": float(fields[7]) if fields[7] else 0.0,
            "change_rate": float(fields[8]) if fields[8] else 0.0,
            "turnover": float(fields[9]) if len(fields) > 9 and fields[9] else 0.0
        }

    def get_daily_kline(self, stock_code: str, years: float = 1.0, max_retries: int = 3) -> Optional[List[Dict]]:
        """获取日K线数据（带重试机制）"""
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

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                response.raise_for_status()

                json_str = response.text.strip()
                if json_str.startswith("jsonp") and json_str.endswith(")"):
                    json_str = json_str[json_str.index("(") + 1:-1]

                data = json.loads(json_str)
                if data.get("rc") == 0 and data.get("data", {}).get("klines"):
                    return [self.parse_kline(k) for k in data["data"]["klines"]]
                return None

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                print(f"{Fore.RED}获取日K线失败({stock_code}): {e}{Style.RESET_ALL}")
                print()
                return None

    @lru_cache(maxsize=100)
    def get_stock_name(self, stock_code: str) -> str:
        """获取股票名称（带缓存）"""
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
                "filter": "(@LISTING_DATE=\"OVER1Y\")(TOTAL_MARKET_CAP<50000000000)(POPULARITY_RANK>0)(POPULARITY_RANK<=2000)(HOLDER_NEWEST>0)(HOLDER_NEWEST<=40000)",
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

    def calculate_annualized_return(self, start_price: float, end_price: float, days: int) -> float:
        """计算年化收益率"""
        if days <= 0 or start_price <= 0:
            return 0.0

        total_return = end_price / start_price
        years = days / 365.0
        annualized_return = math.pow(total_return, 1 / years) - 1
        return annualized_return

    def is_blacklisted(self, stock_name: str) -> bool:
        """检查是否在黑名单中"""
        return any(bad in stock_name for bad in self.blacklist)

    def analyze_trend_stability(self, klines: List[Dict]) -> Dict:
        """分析趋势稳定性"""
        if len(klines) < self.min_days:
            return {"error": f"数据不足，至少需要{self.min_days}天数据"}

        # 准备数据
        closes = np.array([k['close'] for k in klines])
        dates = np.arange(len(closes)).reshape(-1, 1)
        volumes = np.array([k['volume'] for k in klines])
        changes = np.array([abs(k['change_rate']) for k in klines])

        # 线性回归分析
        model = LinearRegression().fit(dates, closes)
        slope = model.coef_[0]
        r_squared = model.score(dates, closes)
        trend_line = model.predict(dates)

        # 计算最大回撤
        peak = np.maximum.accumulate(closes)
        drawdowns = (closes - peak) / peak
        max_drawdown = drawdowns.min()

        # 计算价格波动率和成交量稳定性
        price_volatility = np.std(closes) / np.mean(closes)
        volume_stability = np.std(volumes) / np.mean(volumes)
        max_daily_change = np.max(changes)
        avg_volume = np.mean(volumes)

        # 计算年化收益率
        start_date = datetime.strptime(klines[0]["date"], "%Y-%m-%d")
        end_date = datetime.strptime(klines[-1]["date"], "%Y-%m-%d")
        days = (end_date - start_date).days
        annualized_return = self.calculate_annualized_return(
            klines[0]["close"],
            klines[-1]["close"],
            days
        )

        return {
            "slope": slope,
            "slope_annual": slope * 250,  # 年化斜率
            "r_squared": r_squared,
            "trend_line": trend_line,
            "volatility": price_volatility,
            "volume_stability": volume_stability,
            "avg_volume": avg_volume,
            "max_daily_change": max_daily_change,
            "max_drawdown": max_drawdown,
            "start_date": klines[0]["date"],
            "end_date": klines[-1]["date"],
            "start_price": klines[0]["close"],
            "end_price": klines[-1]["close"],
            "total_change": (klines[-1]["close"] - klines[0]["close"]) / klines[0]["close"],
            "annualized_return": annualized_return,
            "analysis_days": days
        }

    def save_trend_plot(self, stock_code: str, stock_name: str, dates: np.ndarray,
                        closes: np.ndarray, trend_line: np.ndarray, volumes: np.ndarray,
                        annual_return: float):
        """保存趋势图"""
        if not self.plot_enabled:
            return

        plt.figure(figsize=(12, 8))

        # 主图：价格和趋势线
        ax1 = plt.subplot(211)
        ax1.plot(dates, closes, label="收盘价", color='blue', alpha=0.7)
        ax1.plot(dates, trend_line, linestyle='--', label="趋势线", color='red')

        # 添加5日和20日均线
        ma5 = np.convolve(closes, np.ones(5) / 5, mode='valid')
        ma20 = np.convolve(closes, np.ones(20) / 20, mode='valid')
        ax1.plot(dates[4:], ma5, label="5日均线", color='orange', alpha=0.7)
        ax1.plot(dates[19:], ma20, label="20日均线", color='green', alpha=0.7)

        # 添加标题和信息
        title = f"{stock_code} {stock_name}\n年化收益率: {annual_return:.2%}"
        ax1.set_title(title)
        ax1.set_ylabel("价格")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        # 副图：成交量
        ax2 = plt.subplot(212)
        ax2.bar(dates, volumes, alpha=0.3, label="成交量", color='gray')
        ax2.set_xlabel("交易日")
        ax2.set_ylabel("成交量")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()

        plot_path = os.path.join("stable_rise_plots", f"{stock_code}.png")
        plt.savefig(plot_path)
        plt.close()

    def is_stable_rising(self, klines: List[Dict], stock_code: str, stock_name: str) -> Tuple[bool, Dict]:
        """判断是否符合稳定小碎步上涨条件"""
        # 检查黑名单
        if self.is_blacklisted(stock_name):
            return False, {"reason": "股票在黑名单中"}

        analysis = self.analyze_trend_stability(klines)
        if "error" in analysis:
            return False, {"reason": analysis["error"]}

        # 检查各项指标
        checks = [
            (analysis["slope"] >= self.min_slope,
             f"斜率不足({analysis['slope']:.4f}<{self.min_slope})"),
            (analysis["r_squared"] >= self.min_r_squared,
             f"趋势不稳(R²={analysis['r_squared']:.2f}<{self.min_r_squared})"),
            (analysis["max_daily_change"] <= self.max_daily_change * 100,
             f"单日涨幅过大({analysis['max_daily_change']:.2f}%>{self.max_daily_change * 100:.1f}%)"),
            (analysis["volume_stability"] <= self.volume_stability_threshold,
             f"成交量不稳({analysis['volume_stability']:.2f}>{self.volume_stability_threshold})"),
            (analysis["avg_volume"] >= self.min_avg_volume,
             f"成交量不足({analysis['avg_volume']:.0f}<{self.min_avg_volume:.0f})"),
            (analysis["max_drawdown"] >= self.max_drawdown_threshold,
             f"最大回撤过大({analysis['max_drawdown']:.2%}<{self.max_drawdown_threshold:.2%})"),
            (analysis["annualized_return"] >= self.min_annual_return,
             f"年化收益不足({analysis['annualized_return']:.2%}<{self.min_annual_return:.2%})")
        ]

        failed_checks = [reason for passed, reason in checks if not passed]

        if failed_checks:
            return False, {
                "reason": " | ".join(failed_checks),
                **analysis
            }

        # 只有符合条件的股票才绘制趋势图
        dates = np.arange(len(klines))
        closes = np.array([k['close'] for k in klines])
        volumes = np.array([k['volume'] for k in klines])

        self.save_trend_plot(
            stock_code, stock_name, dates,
            closes, analysis["trend_line"], volumes,
            analysis["annualized_return"]
        )

        return True, {
            "reason": "符合所有条件",
            **analysis
        }

    def analyze_single_stock(self, stock_code: str) -> Optional[Dict]:
        """分析单只股票"""
        print(f"\n{Fore.YELLOW}🌟 开始分析股票 {stock_code}...{Style.RESET_ALL}")

        stock_name = self.get_stock_name(stock_code)
        klines = self.get_daily_kline(stock_code)

        if not klines:
            print(f"{Fore.RED}无法获取股票 {stock_code} 的K线数据{Style.RESET_ALL}")
            return None

        is_stable, analysis = self.is_stable_rising(klines, stock_code, stock_name)

        # 打印详细分析结果
        print(f"\n{Fore.CYAN}=== 分析结果 ==={Style.RESET_ALL}")
        print(f"股票: {Fore.YELLOW}{stock_code} {stock_name}{Style.RESET_ALL}")
        print(f"分析周期: {analysis['start_date']} 至 {analysis['end_date']} ({analysis['analysis_days']}天)")
        print(
            f"累计涨幅: {Fore.RED if analysis['total_change'] > 0 else Fore.GREEN}{analysis['total_change']:.2%}{Style.RESET_ALL}")
        print(
            f"年化收益: {Fore.GREEN if analysis['annualized_return'] >= self.min_annual_return else Fore.RED}{analysis['annualized_return']:.2%}{Style.RESET_ALL}")
        print(f"日斜率: {analysis['slope']:.6f} (年化斜率: {analysis['slope_annual']:.2f})")
        print(
            f"趋势稳定性(R²): {Fore.GREEN if analysis['r_squared'] >= self.min_r_squared else Fore.RED}{analysis['r_squared']:.3f}{Style.RESET_ALL}")
        print(f"平均成交量: {analysis['avg_volume']:.0f}手")
        print(
            f"成交量稳定性: {Fore.GREEN if analysis['volume_stability'] <= self.volume_stability_threshold else Fore.RED}{analysis['volume_stability']:.2f}{Style.RESET_ALL}")
        print(
            f"最大回撤: {Fore.RED if analysis['max_drawdown'] < self.max_drawdown_threshold else Fore.GREEN}{analysis['max_drawdown']:.2%}{Style.RESET_ALL}")
        print(
            f"最大单日涨幅: {Fore.RED if analysis['max_daily_change'] > self.max_daily_change * 100 else Fore.GREEN}{analysis['max_daily_change']:.2f}%{Style.RESET_ALL}")
        print(
            f"综合结论: {Fore.GREEN if is_stable else Fore.RED}{'符合' if is_stable else '不符合'}稳定小碎步上涨条件{Style.RESET_ALL}")
        print(f"失败原因: {analysis.get('reason', '无')}")
        if self.plot_enabled:
            print(f"趋势图已保存: stable_rise_plots/{stock_code}.png")

        result = {
            "code": stock_code,
            "name": stock_name,
            "qualified": is_stable,
            "analysis": analysis
        }

        return result

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
            "排名", "代码", "名称", "当前价", "涨幅",
            "年化收益", "斜率", "R平方", "成交量", "回撤", "结果"
        ]
        table.align = "r"
        table.align["名称"] = "l"

        for i, stock in enumerate(hot_stocks, 1):
            stock_code = stock["SECURITY_CODE"]
            stock_name = stock["SECURITY_NAME_ABBR"]

            # 确保数据类型正确
            try:
                current_price = float(stock["NEW_PRICE"]) if stock["NEW_PRICE"] not in [None, "", "-"] else 0.0
                change_rate = float(stock["CHANGE_RATE"]) if stock["CHANGE_RATE"] not in [None, "", "-"] else 0.0
            except Exception as e:
                print(f"{Fore.RED}转换股票数据失败({stock_code}): {e}{Style.RESET_ALL}")
                continue

            print(f"\n{Fore.YELLOW}[{i}/{top_n}] 分析 {stock_code} {stock_name}...{Style.RESET_ALL}")

            klines = self.get_daily_kline(stock_code)
            if not klines:
                print(f"{Fore.RED}无法获取K线数据{Style.RESET_ALL}")
                table.add_row([
                    i, stock_code, stock_name, current_price,
                    f"{change_rate:.2f}%", "-", "-", "-", "-",
                    f"{Fore.RED}无数据{Style.RESET_ALL}"
                ])
                continue

            try:
                is_stable, analysis = self.is_stable_rising(klines, stock_code, stock_name)

                # 提取分析结果
                annual_return = analysis.get('annualized_return', 0)
                slope = analysis.get('slope', 0)
                r_squared = analysis.get('r_squared', 0)
                avg_volume = analysis.get('avg_volume', 0)
                max_drawdown = analysis.get('max_drawdown', 0)

                # 打印简要分析结果
                print(
                    f"  年化收益: {Fore.GREEN if annual_return >= self.min_annual_return else Fore.RED}{annual_return:.2%}{Style.RESET_ALL}")
                print(f"  斜率: {slope:.4f} (R²: {r_squared:.2f})")
                print(f"  平均成交量: {avg_volume:.0f}手")
                print(f"  回撤: {max_drawdown:.2%}")
                print(
                    f"  结论: {Fore.GREEN if is_stable else Fore.RED}{'符合' if is_stable else '不符合'}{Style.RESET_ALL}")

                if is_stable:
                    qualified_stocks.append({
                        "rank": i,
                        "code": stock_code,
                        "name": stock_name,
                        "price": current_price,
                        "change": change_rate,
                        "analysis": analysis
                    })

                # 添加表格行 - 这里修复了change_rate的比较问题
                change_color = Fore.RED if change_rate > 0 else Fore.GREEN
                table.add_row([
                    i,
                    stock_code,
                    stock_name,
                    current_price,
                    f"{change_color}{change_rate:.2f}%{Style.RESET_ALL}",
                    f"{Fore.GREEN if annual_return >= self.min_annual_return else Fore.RED}{annual_return:.2%}{Style.RESET_ALL}",
                    f"{slope:.4f}",
                    f"{Fore.GREEN if r_squared >= self.min_r_squared else Fore.RED}{r_squared:.2f}{Style.RESET_ALL}",
                    f"{avg_volume:.0f}",
                    f"{Fore.RED if max_drawdown < self.max_drawdown_threshold else Fore.GREEN}{max_drawdown:.2%}{Style.RESET_ALL}",
                    f"{Fore.GREEN if is_stable else Fore.RED}{'符合' if is_stable else '不符合'}{Style.RESET_ALL}"
                ])

            except Exception as e:
                print(f"{Fore.RED}分析股票 {stock_code} 时出错: {e}{Style.RESET_ALL}")
                table.add_row([
                    i, stock_code, stock_name, current_price,
                    f"{change_rate:.2f}%", "-", "-", "-", "-",
                    f"{Fore.RED}分析错误{Style.RESET_ALL}"
                ])
                continue


if __name__ == "__main__":
    analyzer = StableRiseStockAnalyzer()

    print(f"{Fore.CYAN}=== 稳定小碎步上涨股票分析工具 ===")
    print("当前参数配置:")
    print(f"1. 最小日斜率: {analyzer.min_slope} (年化约{analyzer.min_slope * 250:.0%})")
    print(f"2. 最小R平方: {analyzer.min_r_squared}")
    print(f"3. 最小分析天数: {analyzer.min_days}")
    print(f"4. 最大单日涨幅: {analyzer.max_daily_change:.0%}")
    print(f"5. 成交量稳定性阈值: {analyzer.volume_stability_threshold}")
    print(f"6. 最小平均成交量: {analyzer.min_avg_volume:.0f}手")
    print(f"7. 最大回撤阈值: {analyzer.max_drawdown_threshold:.0%}")
    print(f"8. 最小年化收益率: {analyzer.min_annual_return:.0%}")
    print(f"9. 黑名单关键词: {analyzer.blacklist}")
    print(f"============================={Style.RESET_ALL}\n")

    while True:
        print("\n选择操作:")
        print("1. 分析单只股票")
        print("2. 扫描热股榜")
        print("3. 修改参数配置")
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
            print(f"\n{Fore.YELLOW}=== 修改参数配置 ===")
            print("1. 最小日斜率 (当前: %.4f)" % analyzer.min_slope)
            print("2. 最小R平方 (当前: %.2f)" % analyzer.min_r_squared)
            print("3. 最小分析天数 (当前: %d)" % analyzer.min_days)
            print("4. 最大单日涨幅 (当前: %.0f%%)" % (analyzer.max_daily_change * 100))
            print("5. 成交量稳定性阈值 (当前: %.1f)" % analyzer.volume_stability_threshold)
            print("6. 最小平均成交量 (当前: %.0f手)" % analyzer.min_avg_volume)
            print("7. 最大回撤阈值 (当前: %.0f%%)" % (analyzer.max_drawdown_threshold * 100))
            print("8. 最小年化收益率 (当前: %.0f%%)" % (analyzer.min_annual_return * 100))
            print("9. 是否保存趋势图 (当前: %s)" % ("是" if analyzer.plot_enabled else "否"))
            param_choice = input("请选择要修改的参数(1-9/q): ").strip().lower()

            if param_choice == 'q':
                continue

            try:
                param_idx = int(param_choice) - 1
                if param_idx < 0 or param_idx > 8:
                    raise ValueError

                new_value = input("请输入新值: ").strip()
                if param_idx in [0, 1, 3, 5, 6, 7]:  # 浮点参数
                    new_value = float(new_value)
                    if param_idx == 3:  # 最大单日涨幅转换为小数
                        new_value /= 100
                elif param_idx in [2, 8]:  # 整数参数
                    new_value = int(new_value)
                elif param_idx == 4:  # 布尔参数
                    new_value = new_value.lower() in ('y', 'yes', 'true', '1')

                # 更新参数
                if param_idx == 0:
                    analyzer.min_slope = new_value
                elif param_idx == 1:
                    analyzer.min_r_squared = new_value
                elif param_idx == 2:
                    analyzer.min_days = new_value
                elif param_idx == 3:
                    analyzer.max_daily_change = new_value
                elif param_idx == 4:
                    analyzer.volume_stability_threshold = new_value
                elif param_idx == 5:
                    analyzer.min_avg_volume = new_value
                elif param_idx == 6:
                    analyzer.max_drawdown_threshold = new_value / 100
                elif param_idx == 7:
                    analyzer.min_annual_return = new_value / 100
                elif param_idx == 8:
                    analyzer.plot_enabled = new_value

                print(f"{Fore.GREEN}参数修改成功!{Style.RESET_ALL}")
            except:
                print(f"{Fore.RED}参数修改失败，请输入有效值{Style.RESET_ALL}")
        elif choice == '4':
            print(f"\n{Fore.YELLOW}=== 使用帮助 ===")
            print("1. 分析单只股票: 输入6位股票代码")
            print("2. 扫描热股榜: 分析东财热股榜前N只股票")
            print("3. 修改参数: 可以调整各项筛选条件")
            print("4. 结果会保存在当前目录下的stable_rise_plots文件夹和json文件中")
            print(f"================{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}无效的选择，请重新输入{Style.RESET_ALL}")