import requests
import json
from datetime import datetime
from requests_toolbelt.multipart.encoder import MultipartEncoder
from colorama import Fore, Style, init
from typing import List, Dict, Optional
from prettytable import PrettyTable
import time

# 初始化colorama
init(autoreset=True)

# 导入分析类
from DealDaily import DailyAnalyzer
from DealMonth import MonthlyStockAnalyzer


class StockRank:
    """热股榜分析类 - 筛选符合技术条件的股票"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Referer": "https://emrnweb.eastmoney.com/"
        }
        self.daily_analyzer = DailyAnalyzer()
        self.monthly_analyzer = MonthlyStockAnalyzer()

    def get_hot_stocks(self, top_n: int = 100) -> Optional[List[Dict]]:
        """获取热股榜数据"""
        url = "https://datacenter.eastmoney.com/stock/selection/api/data/get/"

        multipart_data = MultipartEncoder(
            fields={
                "type": "RPTA_SECURITY_STOCKSELECT",
                "sty": "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,NEW_PRICE,CHANGE_RATE,TOTAL_MARKET_CAP,POPULARITY_RANK",
                "filter": "(TOTAL_MARKET_CAP<10000000000)(POPULARITY_RANK>0)(POPULARITY_RANK<=1500)(HOLDER_NEWEST>0)(HOLDER_NEWEST<=30000)",
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

    def get_daily_kline(self, stock_code: str, max_retries: int = 3) -> Optional[List[str]]:
        """获取日K线数据（带重试机制）

        Args:
            stock_code: 股票代码
            max_retries: 最大重试次数，默认3次

        Returns:
            成功返回K线数据列表，失败返回None
        """
        time.sleep(1)  # 基础暂停1秒，避免请求过于频繁

        params = {
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "beg": "0",
            "end": "20500101",
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            "rtntype": "6",
            "secid": f"1.{stock_code}" if stock_code.startswith('6') else f"0.{stock_code}",
            "klt": "101",  # 日线
            "fqt": "1"
        }

        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

        for attempt in range(max_retries + 1):  # 总尝试次数=重试次数+初始请求
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                response.raise_for_status()

                json_str = response.text.strip()
                if json_str.startswith("jsonp") and json_str.endswith(")"):
                    json_str = json_str[json_str.index("(") + 1:-1]

                data = json.loads(json_str)
                if data.get("rc") == 0 and data.get("data", {}).get("klines"):
                    return data["data"]["klines"]

            except Exception as e:
                if attempt < max_retries:  # 不是最后一次尝试
                    wait_time = 10  # 重试间隔10秒
                    # print( f"{Fore.YELLOW}获取日K线失败({stock_code})，第{attempt + 1}次重试，等待{wait_time}秒... 错误: {e}{Style.RESET_ALL}")
                    time.sleep(wait_time)
                else:  # 最后一次尝试仍然失败
                    print(
                        f"{Fore.RED}获取日K线失败({stock_code})，已达最大重试次数({max_retries}次): {e}{Style.RESET_ALL}")

        return None

    def get_monthly_kline(self, stock_code: str) -> Optional[List[str]]:
        time.sleep(1)  # 暂停 1 秒
        """获取月K线数据"""
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "beg": "0",
            "end": "20500101",
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            "rtntype": "6",
            "secid": f"1.{stock_code}" if stock_code.startswith('6') else f"0.{stock_code}",
            "klt": "103",  # 月线
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
                return data["data"]["klines"]
        except Exception as e:
            print(f"{Fore.RED}获取月K线失败({stock_code}): {e}{Style.RESET_ALL}")

        return None

    def analyze_stock(self, stock: dict) -> Optional[dict]:
        """分析单只股票的技术面"""
        if not stock or "SECURITY_CODE" not in stock:
            return None

        code = stock["SECURITY_CODE"]
        name = stock.get("SECURITY_NAME_ABBR", "未知")
        popularity_rank = stock.get("POPULARITY_RANK", 999)  # 热度排名

        # 获取日线和月线数据
        daily_klines = self.get_daily_kline(code)
        monthly_klines = self.get_monthly_kline(code)

        if not daily_klines or not monthly_klines:
            return None

        # 日线分析 - 突破均线
        is_break_ma, ma_type = self.daily_analyzer.is_price_break_ma(daily_klines)

        # 日线分析 - 底部调整
        # is_daily_bottom, daily_score = self.daily_analyzer.is_bottom_adjusted(daily_klines)

        # 日线分析 - 连续放量
        is_volume_increasing, consecutive_days = self.daily_analyzer.is_volume_increasing(daily_klines)

        # 月线分析 - 底部调整
        is_monthly_bottom, monthly_score = self.monthly_analyzer.is_bottom_adjusted(monthly_klines)

        return {
            "code": code,
            "name": name,
            "popularity_rank": popularity_rank,
            "price": stock.get("NEW_PRICE", 0),
            "change_rate": stock.get("CHANGE_RATE", 0),
            "market_cap": stock.get("TOTAL_MARKET_CAP", 0) / 1e8,
            "is_break_ma": bool(is_break_ma),
            "ma_type": ma_type,
            # "is_daily_bottom": is_daily_bottom,
            # "daily_score": daily_score,
            "is_monthly_bottom": bool(is_monthly_bottom),
            "monthly_score": monthly_score,
            "is_volume_increasing": bool(is_volume_increasing),
            "consecutive_days": consecutive_days,
            "qualified": bool(is_break_ma and is_volume_increasing and is_monthly_bottom)
        }

    def find_qualified_stocks(self, top_n: int = 100) -> List[Dict]:
        """寻找符合技术条件的股票"""
        print(f"{Fore.YELLOW}🚀 开始分析热股榜股票...{Style.RESET_ALL}")

        hot_stocks = self.get_hot_stocks(top_n)
        if not hot_stocks:
            print(f"{Fore.RED}❌ 获取热股榜失败!{Style.RESET_ALL}")
            return []

        analyzed_stocks = []
        for stock in hot_stocks:
            analysis = self.analyze_stock(stock)
            if analysis:
                analyzed_stocks.append(analysis)

                desc1 = f"{Fore.RED}突破均线{Style.RESET_ALL}" if analysis["is_break_ma"] else f"{Fore.GREEN}未突破均线{Style.RESET_ALL}"
                desc2 = f"{Fore.RED}连续放量{Style.RESET_ALL}" if analysis["is_volume_increasing"] else f"{Fore.GREEN}未连续放量{Style.RESET_ALL}"
                desc3 =  f"{Fore.RED}月线底部{Style.RESET_ALL}" if analysis["is_monthly_bottom"] else f"{Fore.GREEN}未月线底部{Style.RESET_ALL}"
                if analysis["qualified"]:
                    print(f"{Fore.RED}✅ 符合条件: {analysis['name']}({analysis['code']}){Style.RESET_ALL} | {desc1} | {desc2} | {desc3}")
                else:
                    print(f"{Fore.GREEN}⏩ 不符合: {analysis['name']}({analysis['code']}){Style.RESET_ALL} | {desc1} | {desc2} | {desc3}")

        # 按热度排名排序
        analyzed_stocks.sort(key=lambda x: x["popularity_rank"])

        # 保存结果
        if analyzed_stocks:
            filename = f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(analyzed_stocks, f, ensure_ascii=False, indent=4)
            print(f"{Fore.CYAN}💾 分析结果已保存到: {filename}{Style.RESET_ALL}")

        return analyzed_stocks

    def print_results(self, stocks: List[Dict]) -> None:
        """打印分析结果"""
        if not stocks:
            print(f"{Fore.RED}没有找到任何股票数据{Style.RESET_ALL}")
            return

        # 创建表格
        table = PrettyTable()
        table.title = f"{Fore.YELLOW}📊 热股榜技术分析 (共 {len(stocks)} 只股票){Style.RESET_ALL}"

        # 设置表格字段
        table.field_names = [
            "排名", "代码", "名称",
            "价格", "涨跌幅", "市值(亿)",
            "月线底部", "连续放量", "突破均线"
        ]

        # 设置对齐方式
        table.align["名称"] = "l"
        table.align["价格"] = "r"
        table.align["涨跌幅"] = "r"
        table.align["市值(亿)"] = "r"

        # 添加数据行
        for stock in stocks:
            # 涨跌幅颜色
            change_rate = stock['change_rate']
            change_color = Fore.RED if change_rate > 0 else Fore.GREEN if change_rate < 0 else ''
            change_str = f"{change_color}{change_rate:.2f}%{Style.RESET_ALL}"

            # 技术指标颜色和显示
            monthly_bottom = f"{Fore.RED}是{Style.RESET_ALL}" if stock[
                'is_monthly_bottom'] else f"{Fore.GREEN}否{Style.RESET_ALL}"
            volume_inc = f"{Fore.RED}是{Style.RESET_ALL}" if stock[
                'is_volume_increasing'] else f"{Fore.GREEN}否{Style.RESET_ALL}"
            break_ma = f"{Fore.RED}{stock['ma_type']}{Style.RESET_ALL}" if stock[
                'is_break_ma'] else f"{Fore.GREEN}否{Style.RESET_ALL}"

            table.add_row([
                stock['popularity_rank'],
                stock['code'],
                stock['name'],
                f"{stock['price']:.2f}",
                change_str,
                f"{stock['market_cap']:.2f}",
                monthly_bottom,
                volume_inc,
                break_ma
            ])

        # 打印表格
        print("\n" + table.get_string())

        # 打印统计信息
        qualified_count = sum(1 for s in stocks if s['qualified'])
        print(f"\n{Fore.CYAN}📈 统计信息:{Style.RESET_ALL}")
        print(
            f"符合所有技术条件的股票: {Fore.GREEN if qualified_count > 0 else Fore.RED}{qualified_count}只{Style.RESET_ALL}")
        print(f"月线底部股票: {sum(1 for s in stocks if s['is_monthly_bottom'])}只")
        print(f"连续放量股票: {sum(1 for s in stocks if s['is_volume_increasing'])}只")
        print(f"突破均线股票: {sum(1 for s in stocks if s['is_break_ma'])}只")


if __name__ == "__main__":
    analyzer = StockRank()
    analyzed_stocks = analyzer.find_qualified_stocks(top_n=300)
    analyzer.print_results(analyzed_stocks)