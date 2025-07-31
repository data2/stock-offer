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
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化colorama
init(autoreset=True)
import matplotlib
matplotlib.use('Agg')  # 在import pyplot之前设置
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']  # 尝试的字体列表
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class EnhancedStableRiseStockAnalyzer:
    """增强版稳定小碎步上涨股票分析器"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Referer": "https://emrnweb.eastmoney.com/"
        }

        # ========== 优化后的关键参数 ==========
        # 趋势相关参数
        self.min_slope = 0.001  # 降低斜率标准，增加筛选范围
        self.min_r_squared = 0.48  # 降低趋势稳定性要求
        self.min_days = 30  # 适中分析周期
        self.max_daily_change = 0.15  # 单日最大涨幅(8%)

        # 新增趋势确认参数
        self.ma_cross_threshold = 0.05  # 5日/20日均线最大偏离阈值
        self.min_positive_days_ratio = 0.49  # 最小上涨天数比例

        # 成交量优化参数
        self.volume_price_correlation_threshold = 0.25  # 量价相关性最低阈值
        self.volume_increase_threshold = 0.9  # 近期成交量放大阈值

        # 风险控制优化
        self.max_volatility = 0.12  # 最大日波动率
        self.max_consecutive_down_days = 8  # 最大连续下跌天数
        self.max_drawdown_threshold = -0.35  # 放宽回撤要求(-25%)
        self.min_annual_return = 0.1  # 降低年化收益要求(12%)

        # 其他参数
        self.blacklist = ['ST', '*ST', '退市', 'N', 'U']  # 扩展黑名单
        self.plot_enabled = True  # 是否保存趋势图
        self.max_workers = 1  # 并发线程数
        # ================================

        # 创建输出目录
        os.makedirs("stable_rise_plots", exist_ok=True)
        os.makedirs("stock_data_cache", exist_ok=True)

    @lru_cache(maxsize=500)
    def get_daily_kline(self, stock_code: str, years: float = 1.0) -> Optional[List[Dict]]:
        time.sleep(1)
        """获取日K线数据（带缓存和本地存储）"""
        cache_file = f"stock_data_cache/{stock_code}.json"

        # 检查本地缓存
        if os.path.exists(cache_file):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mtime).days < 1:  # 1天内缓存有效
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

        # 从API获取数据
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

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            response.raise_for_status()

            json_str = response.text.strip()
            if json_str.startswith("jsonp") and json_str.endswith(")"):
                json_str = json_str[json_str.index("(") + 1:-1]

            data = json.loads(json_str)
            if data.get("rc") == 0 and data.get("data", {}).get("klines"):
                klines = [self.parse_kline(k) for k in data["data"]["klines"]]
                # 保存到本地缓存
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(klines, f, ensure_ascii=False)
                return klines
            return None

        except Exception as e:
            print(f"{Fore.RED}获取日K线失败({stock_code}): {e}{Style.RESET_ALL}")
            return None

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

    @lru_cache(maxsize=500)
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

    def get_hot_stocks(self, top_n: int = 100) -> List[Dict]:
        """获取热股榜数据（增强版）"""
        url = "https://datacenter.eastmoney.com/stock/selection/api/data/get/"

        multipart_data = MultipartEncoder(
            fields={
                "type": "RPTA_SECURITY_STOCKSELECT",
                "sty": "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,NEW_PRICE,CHANGE_RATE,TOTAL_MARKET_CAP,POPULARITY_RANK",
                "filter": "(POPULARITY_RANK>0)(POPULARITY_RANK<=4000)",
                "p": "1",
                "ps": str(top_n),
                "sr": "-1",
                "st": "POPULARITY_RANK",
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
                return [
                    {
                        "code": item["SECURITY_CODE"],
                        "name": item["SECURITY_NAME_ABBR"],
                        "price": float(item["NEW_PRICE"]) if item["NEW_PRICE"] not in [None, "", "-"] else 0.0,
                        "change": float(item["CHANGE_RATE"]) if item["CHANGE_RATE"] not in [None, "", "-"] else 0.0,
                        "market_cap": float(item["TOTAL_MARKET_CAP"]) if item["TOTAL_MARKET_CAP"] not in [None, "", "-"] else 0.0,
                        "rank": idx + 1
                    }
                    for idx, item in enumerate(result["result"]["data"])
                ]

        except Exception as e:
            print(f"{Fore.RED}获取热股榜失败: {e}{Style.RESET_ALL}")

        return []

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

    def enhanced_analyze_trend_stability(self, klines: List[Dict]) -> Dict:
        """增强版趋势稳定性分析"""
        if len(klines) < self.min_days:
            return {"error": f"数据不足，至少需要{self.min_days}天数据"}

        closes = np.array([k['close'] for k in klines])
        volumes = np.array([k['volume'] for k in klines])
        changes = np.array([k['change_rate'] for k in klines])
        dates = np.arange(len(closes)).reshape(-1, 1)

        # 基础线性回归分析
        model = LinearRegression().fit(dates, closes)
        slope = model.coef_[0]
        r_squared = model.score(dates, closes)
        trend_line = model.predict(dates)

        # 新增分析指标 -------------------------------------------------

        # 1. 移动平均线验证
        ma5 = np.convolve(closes, np.ones(5) / 5, mode='valid')
        ma20 = np.convolve(closes, np.ones(20) / 20, mode='valid')
        ma_diff = (ma5[-10:] - ma20[-10:]).mean() / closes.mean()  # 最近10天均线差异

        # 2. 上涨天数比例
        positive_days = sum(1 for c in changes if c > 0)
        positive_days_ratio = positive_days / len(changes)

        # 3. 量价相关性
        volume_price_corr = np.corrcoef(closes[-20:], volumes[-20:])[0, 1]  # 最近20天量价相关性

        # 4. 成交量放大分析
        early_volume = volumes[:len(volumes) // 3].mean()
        late_volume = volumes[len(volumes) // 3 * 2:].mean()
        volume_increase_ratio = late_volume / early_volume if early_volume > 0 else 1

        # 5. 波动率和连续下跌分析
        daily_volatility = np.std(closes) / np.mean(closes)

        consecutive_down = 0
        max_consecutive_down = 0
        for c in changes:
            if c < 0:
                consecutive_down += 1
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
            else:
                consecutive_down = 0

        # 6. 回撤深度分析
        peak = np.maximum.accumulate(closes)
        drawdowns = (closes - peak) / peak
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns.mean()

        # 7. 最大单日涨幅
        max_daily_change = max(abs(c) for c in changes)

        # 计算年化收益率
        days = (datetime.strptime(klines[-1]["date"], "%Y-%m-%d") -
                datetime.strptime(klines[0]["date"], "%Y-%m-%d")).days
        annualized_return = self.calculate_annualized_return(
            klines[0]["close"], klines[-1]["close"], days)

        return {
            # 基础指标
            "slope": slope,
            "r_squared": r_squared,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "max_daily_change": max_daily_change,
            "avg_volume": np.mean(volumes),

            # 新增指标
            "ma_diff_ratio": ma_diff,
            "positive_days_ratio": positive_days_ratio,
            "volume_price_corr": volume_price_corr,
            "volume_increase_ratio": volume_increase_ratio,
            "daily_volatility": daily_volatility,
            "max_consecutive_down": max_consecutive_down,
            "avg_drawdown": avg_drawdown,

            # 原始数据
            "trend_line": trend_line,
            "closes": closes,
            "volumes": volumes,
            "start_date": klines[0]["date"],
            "end_date": klines[-1]["date"],
            "analysis_days": days
        }

    def is_stable_rising(self, klines: List[Dict], stock_code: str, stock_name: str) -> Tuple[bool, Dict]:
        """优化后的稳定上涨判断逻辑"""
        if self.is_blacklisted(stock_name):
            return False, {"reason": "股票在黑名单中"}

        analysis = self.enhanced_analyze_trend_stability(klines)
        if "error" in analysis:
            return False, {"reason": analysis["error"]}

        # 基础条件检查（移除了成交量最低限制）
        base_checks = [
            (analysis["slope"] >= self.min_slope,
             f"斜率不足({analysis['slope']:.4f}<{self.min_slope})"),
            (analysis["r_squared"] >= self.min_r_squared,
             f"趋势不稳(R²={analysis['r_squared']:.2f}<{self.min_r_squared})"),
            (analysis["annualized_return"] >= self.min_annual_return,
             f"年化收益不足({analysis['annualized_return']:.2%}<{self.min_annual_return:.2%})"),
            (analysis["max_drawdown"] >= self.max_drawdown_threshold,
             f"最大回撤过大({analysis['max_drawdown']:.2%}<{self.max_drawdown_threshold:.2%})"),
            (analysis["max_daily_change"] <= self.max_daily_change * 100,
             f"单日涨幅过大({analysis['max_daily_change']:.2f}%>{self.max_daily_change * 100:.1f}%)")
        ]

        # 新增条件检查（保留量价相关性检查）
        enhanced_checks = [
            (abs(analysis["ma_diff_ratio"]) <= self.ma_cross_threshold,
             f"均线偏离过大({analysis['ma_diff_ratio']:.2%}>{self.ma_cross_threshold:.2%})"),
            (analysis["positive_days_ratio"] >= self.min_positive_days_ratio,
             f"上涨天数不足({analysis['positive_days_ratio']:.2%}<{self.min_positive_days_ratio:.2%})"),
            (analysis["volume_price_corr"] >= self.volume_price_correlation_threshold,
             f"量价相关性低({analysis['volume_price_corr']:.2f}<{self.volume_price_correlation_threshold:.2f})"),
            (analysis["volume_increase_ratio"] >= self.volume_increase_threshold,
             f"成交量放大不足({analysis['volume_increase_ratio']:.2f}<{self.volume_increase_threshold:.2f})"),
            (analysis["daily_volatility"] <= self.max_volatility,
             f"波动率过高({analysis['daily_volatility']:.2%}>{self.max_volatility:.2%})"),
            (analysis["max_consecutive_down"] <= self.max_consecutive_down_days,
             f"连续下跌天数过多({analysis['max_consecutive_down']}>{self.max_consecutive_down_days})")
        ]

        failed_checks = [reason for passed, reason in base_checks + enhanced_checks if not passed]

        if failed_checks:
            return False, {
                "reason": " | ".join(failed_checks),
                **analysis
            }

        # 保存趋势图
        if self.plot_enabled:
            self.save_enhanced_trend_plot(stock_code, stock_name, analysis)

        return True, {
            "reason": "符合所有条件",
            **analysis
        }

    def save_enhanced_trend_plot(self, stock_code: str, stock_name: str, analysis: Dict):
        """增强版趋势图保存"""
        plt.figure(figsize=(14, 10))
        dates = np.arange(len(analysis["closes"]))

        # 主图：价格和趋势线
        ax1 = plt.subplot(211)
        ax1.plot(dates, analysis["closes"], label="收盘价", color='blue', alpha=0.7)
        ax1.plot(dates, analysis["trend_line"], linestyle='--', label="趋势线", color='red')

        # 添加均线
        ma5 = np.convolve(analysis["closes"], np.ones(5) / 5, mode='valid')
        ma20 = np.convolve(analysis["closes"], np.ones(20) / 20, mode='valid')
        ax1.plot(dates[4:], ma5, label="5日均线", color='orange', alpha=0.7)
        ax1.plot(dates[19:], ma20, label="20日均线", color='green', alpha=0.7)

        # 标注关键信息
        info_text = (
            f"代码: {stock_code}  名称: {stock_name}\n"
            f"年化收益: {analysis['annualized_return']:.2%}  "
            f"斜率: {analysis['slope']:.4f}  R²: {analysis['r_squared']:.2f}\n"
            f"上涨天数: {analysis['positive_days_ratio']:.2%}  "
            f"量价相关: {analysis['volume_price_corr']:.2f}  "
            f"成交量放大: {analysis['volume_increase_ratio']:.2f}x\n"
            f"最大回撤: {analysis['max_drawdown']:.2%}  "
            f"波动率: {analysis['daily_volatility']:.2%}"
        )
        ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax1.set_title(f"{stock_code}")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left')

        # 副图1：成交量
        ax2 = plt.subplot(212)
        ax2.bar(dates, analysis["volumes"], color='gray', alpha=0.7, label="成交量")

        # 添加成交量均线
        volume_ma5 = np.convolve(analysis["volumes"], np.ones(5) / 5, mode='valid')
        ax2.plot(dates[4:], volume_ma5, color='red', label="5日成交量均线")

        ax2.set_xlabel("交易日")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join("stable_rise_plots", f"{stock_code}_enhanced.png")
        plt.savefig(plot_path, dpi=120)
        plt.close()

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
        print(f"累计涨幅: {Fore.RED if analysis['annualized_return'] < 0 else Fore.GREEN}"
              f"{(klines[-1]['close'] / klines[0]['close'] - 1):.2%}{Style.RESET_ALL}")
        print(f"年化收益: {Fore.GREEN if analysis['annualized_return'] >= self.min_annual_return else Fore.RED}"
              f"{analysis['annualized_return']:.2%}{Style.RESET_ALL}")
        print(f"日斜率: {analysis['slope']:.6f} (年化斜率: {analysis['slope'] * 250:.2f})")
        print(f"趋势稳定性(R²): {Fore.GREEN if analysis['r_squared'] >= self.min_r_squared else Fore.RED}"
              f"{analysis['r_squared']:.3f}{Style.RESET_ALL}")
        print(f"上涨天数比例: {analysis['positive_days_ratio']:.2%}")
        print(f"平均成交量: {analysis['avg_volume']:.0f}手")
        print(f"成交量放大: {analysis['volume_increase_ratio']:.2f}x")
        print(f"量价相关性: {analysis['volume_price_corr']:.2f}")
        print(f"最大回撤: {Fore.RED if analysis['max_drawdown'] < self.max_drawdown_threshold else Fore.GREEN}"
              f"{analysis['max_drawdown']:.2%}{Style.RESET_ALL}")
        print(f"平均回撤: {analysis['avg_drawdown']:.2%}")
        print(f"最大单日涨幅: {Fore.RED if analysis['max_daily_change'] > self.max_daily_change * 100 else Fore.GREEN}"
              f"{analysis['max_daily_change']:.2f}%{Style.RESET_ALL}")
        print(f"波动率: {analysis['daily_volatility']:.2%}")
        print(f"连续下跌天数: {analysis['max_consecutive_down']}")
        print(f"综合结论: {Fore.GREEN if is_stable else Fore.RED}"
              f"{'符合' if is_stable else '不符合'}稳定小碎步上涨条件{Style.RESET_ALL}")
        print(f"失败原因: {analysis.get('reason', '无')}")
        if self.plot_enabled and is_stable:
            print(f"趋势图已保存: stable_rise_plots/{stock_code}_enhanced.png")

        return {
            "code": stock_code,
            "name": stock_name,
            "qualified": is_stable,
            "analysis": analysis
        }

    def analyze_stock_list(self, stock_list: List[Dict]) -> List[Dict]:
        """分析股票列表（单线程版本）"""
        qualified_stocks = []
        table = PrettyTable()
        table.field_names = [
            "代码", "名称", "当前价", "涨幅", "年化收益",
            "斜率", "R平方", "上涨天数", "成交量", "回撤", "结果"
        ]
        table.align = "r"
        table.align["名称"] = "l"

        for stock in stock_list:
            try:
                result = self.analyze_single_stock(stock["code"])
                if result and result["qualified"]:
                    qualified_stocks.append(result)
                    analysis = result["analysis"]

                    # 添加表格行
                    table.add_row([
                        result["code"],
                        result["name"],
                        stock.get("price", "-"),
                        f"{Fore.RED if stock.get('change', 0) > 0 else Fore.GREEN}"
                        f"{stock.get('change', 0):.2f}%{Style.RESET_ALL}",
                        f"{Fore.GREEN if analysis['annualized_return'] >= self.min_annual_return else Fore.RED}"
                        f"{analysis['annualized_return']:.2%}{Style.RESET_ALL}",
                        f"{analysis['slope']:.4f}",
                        f"{Fore.GREEN if analysis['r_squared'] >= self.min_r_squared else Fore.RED}"
                        f"{analysis['r_squared']:.2f}{Style.RESET_ALL}",
                        f"{analysis['positive_days_ratio']:.2%}",
                        f"{analysis['avg_volume'] / 1e4:.1f}万",
                        f"{Fore.RED if analysis['max_drawdown'] < self.max_drawdown_threshold else Fore.GREEN}"
                        f"{analysis['max_drawdown']:.2%}{Style.RESET_ALL}",
                        f"{Fore.GREEN}符合{Style.RESET_ALL}"
                    ])
            except Exception as e:
                print(f"{Fore.RED}分析股票 {stock['code']} 时出错: {e}{Style.RESET_ALL}")
                traceback.print_exc()

        # 打印结果表格
        if qualified_stocks:
            print(f"\n{Fore.GREEN}=== 符合条件的股票 ==={Style.RESET_ALL}")
            print(table)
        else:
            print(f"\n{Fore.YELLOW}⚠️ 没有找到符合条件的股票{Style.RESET_ALL}")

        return qualified_stocks

    def analyze_hot_stocks(self, top_n: int = 50) -> List[Dict]:
        """分析热股榜股票"""
        print(f"\n{Fore.CYAN}=== 开始分析热股榜前{top_n}只股票 ==={Style.RESET_ALL}")

        hot_stocks = self.get_hot_stocks(top_n)
        if not hot_stocks:
            print(f"{Fore.RED}无法获取热股榜数据{Style.RESET_ALL}")
            return []

        return self.analyze_stock_list(hot_stocks)

    def analyze_custom_list(self, stock_codes: List[str]) -> List[Dict]:
        """分析自定义股票列表"""
        print(f"\n{Fore.CYAN}=== 开始分析自定义股票列表 ==={Style.RESET_ALL}")

        stock_list = [{"code": code, "name": self.get_stock_name(code)} for code in stock_codes]
        return self.analyze_stock_list(stock_list)

    def interactive_mode(self):
        """交互式分析模式"""
        print(f"{Fore.CYAN}=== 稳定小碎步上涨股票分析工具 ===")
        print("当前参数配置:")
        print(f"1. 最小日斜率: {self.min_slope} (年化约{self.min_slope * 250:.0%})")
        print(f"2. 最小R平方: {self.min_r_squared}")
        print(f"3. 最小分析天数: {self.min_days}")
        print(f"4. 最大单日涨幅: {self.max_daily_change:.0%}")
        print(f"5. 成交量稳定性阈值: {self.volume_price_correlation_threshold}")
        print(f"6. 最大回撤阈值: {self.max_drawdown_threshold:.0%}")
        print(f"7. 最小年化收益率: {self.min_annual_return:.0%}")
        print(f"8. 上涨天数比例要求: {self.min_positive_days_ratio:.0%}")
        print(f"9. 最大连续下跌天数: {self.max_consecutive_down_days}")
        print(f"10. 是否保存趋势图 (当前: {'是' if self.plot_enabled else '否'})")
        print(f"11. 并发线程数 (当前: {self.max_workers})")
        print(f"============================={Style.RESET_ALL}\n")

        while True:
            print("\n选择操作:")
            print("1. 分析单只股票")
            print("2. 扫描热股榜")
            print("3. 分析自定义股票列表")
            print("4. 修改参数配置")
            print("5. 查看帮助")
            print("q. 退出")

            choice = input("请输入选择(1-5/q): ").strip().lower()

            if choice == 'q':
                break

            if choice == '1':
                stock_code = input("请输入股票代码(如600000): ").strip()
                if not stock_code.isdigit() or len(stock_code) != 6:
                    print(f"{Fore.RED}股票代码应为6位数字{Style.RESET_ALL}")
                    continue
                self.analyze_single_stock(stock_code)
            elif choice == '2':
                top_n = input(f"请输入要分析的热股数量(默认50, 最大3000): ").strip()
                try:
                    top_n = int(top_n) if top_n else 50
                    top_n = min(max(top_n, 1), 200)
                    self.analyze_hot_stocks(top_n)
                except ValueError:
                    print(f"{Fore.RED}请输入有效的数字{Style.RESET_ALL}")
            elif choice == '3':
                codes = input("请输入股票代码列表(用逗号分隔, 如600000,000001): ").strip()
                stock_codes = [c.strip() for c in codes.split(",") if c.strip()]
                invalid_codes = [c for c in stock_codes if not c.isdigit() or len(c) != 6]
                if invalid_codes:
                    print(f"{Fore.RED}以下股票代码无效: {', '.join(invalid_codes)}{Style.RESET_ALL}")
                    continue
                self.analyze_custom_list(stock_codes)
            elif choice == '4':
                self.adjust_parameters()
            elif choice == '5':
                self.show_help()
            else:
                print(f"{Fore.RED}无效的选择，请重新输入{Style.RESET_ALL}")

    def adjust_parameters(self):
        """调整参数配置"""
        print(f"\n{Fore.YELLOW}=== 修改参数配置 ===")
        print("1. 最小日斜率 (当前: %.4f)" % self.min_slope)
        print("2. 最小R平方 (当前: %.2f)" % self.min_r_squared)
        print("3. 最小分析天数 (当前: %d)" % self.min_days)
        print("4. 最大单日涨幅 (当前: %.0f%%)" % (self.max_daily_change * 100))
        print("5. 量价相关性阈值 (当前: %.2f)" % self.volume_price_correlation_threshold)
        print("6. 最大回撤阈值 (当前: %.0f%%)" % (self.max_drawdown_threshold * 100))
        print("7. 最小年化收益率 (当前: %.0f%%)" % (self.min_annual_return * 100))
        print("8. 上涨天数比例要求 (当前: %.0f%%)" % (self.min_positive_days_ratio * 100))
        print("9. 最大连续下跌天数 (当前: %d)" % self.max_consecutive_down_days)
        print("10. 是否保存趋势图 (当前: %s)" % ("是" if self.plot_enabled else "否"))
        print("11. 并发线程数 (当前: %d)" % self.max_workers)
        param_choice = input("请选择要修改的参数(1-11/q): ").strip().lower()

        if param_choice == 'q':
            return

        try:
            param_idx = int(param_choice) - 1
            if param_idx < 0 or param_idx > 10:
                raise ValueError

            new_value = input("请输入新值: ").strip()
            if param_idx in [0, 1, 4]:  # 浮点参数
                new_value = float(new_value)
            elif param_idx in [2, 8, 10]:  # 整数参数
                new_value = int(new_value)
            elif param_idx in [3, 5, 6, 7]:  # 百分比参数
                new_value = float(new_value) / 100
            elif param_idx == 9:  # 布尔参数
                new_value = new_value.lower() in ('y', 'yes', 'true', '1')

            # 更新参数
            if param_idx == 0:
                self.min_slope = new_value
            elif param_idx == 1:
                self.min_r_squared = new_value
            elif param_idx == 2:
                self.min_days = new_value
            elif param_idx == 3:
                self.max_daily_change = new_value
            elif param_idx == 4:
                self.volume_price_correlation_threshold = new_value
            elif param_idx == 5:
                self.max_drawdown_threshold = new_value
            elif param_idx == 6:
                self.min_annual_return = new_value
            elif param_idx == 7:
                self.min_positive_days_ratio = new_value
            elif param_idx == 8:
                self.max_consecutive_down_days = new_value
            elif param_idx == 9:
                self.plot_enabled = new_value
            elif param_idx == 10:
                self.max_workers = new_value

            print(f"{Fore.GREEN}参数修改成功!{Style.RESET_ALL}")
        except:
            print(f"{Fore.RED}参数修改失败，请输入有效值{Style.RESET_ALL}")

    def show_help(self):
        """显示帮助信息"""
        print(f"\n{Fore.YELLOW}=== 使用帮助 ===")
        print("1. 分析单只股票: 输入6位股票代码")
        print("2. 扫描热股榜: 分析东财热股榜前N只股票")
        print("3. 分析自定义股票列表: 输入多个股票代码(逗号分隔)")
        print("4. 修改参数: 可以调整各项筛选条件")
        print("5. 结果会保存在当前目录下的stable_rise_plots文件夹")
        print("6. 数据会自动缓存到stock_data_cache文件夹")
        print("\n筛选逻辑说明:")
        print("- 寻找日线呈现稳定小斜率上涨的股票")
        print("- 要求成交量温和放大且量价配合")
        print("- 控制单日涨幅和连续下跌天数")
        print("- 综合考虑年化收益和最大回撤")
        print(f"================{Style.RESET_ALL}")


if __name__ == "__main__":
    analyzer = EnhancedStableRiseStockAnalyzer()
    analyzer.interactive_mode()