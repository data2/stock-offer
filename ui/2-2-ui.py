import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QTabWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView, QComboBox, QScrollArea, QFrame)
from PyQt5.QtGui import QIcon, QFont, QColor, QBrush
from PyQt5.QtCore import Qt, QDateTime
from strategy.Strategy22 import StrategyTwoAnalyzer
import traceback

class StockStrategyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = StrategyTwoAnalyzer()
        self.init_ui()
        self.setup_styles()

    def setup_styles(self):
        """设置全局样式"""
        self.setStyleSheet("""
            /* 主窗口样式 */
            QMainWindow {
                background-color: #f5f7fa;
            }

            /* 标签样式 */
            QLabel {
                color: #333333;
                font-size: 12px;
            }

            /* 按钮样式 */
            QPushButton {
                background-color: #4a6fa5;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                min-width: 80px;
                font-family: 'Microsoft YaHei';
            }
            QPushButton:hover {
                background-color: #3a5a8f;
            }
            QPushButton:pressed {
                background-color: #2a4a7f;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }

            /* 输入框样式 */
            QTextEdit, QComboBox {
                border: 1px solid #dddddd;
                border-radius: 4px;
                padding: 5px;
                font-family: Consolas;
                background: white;
            }

            /* 标签页样式 */
            QTabWidget::pane {
                border: 1px solid #dddddd;
                border-radius: 4px;
                padding: 5px;
                background: white;
            }
            QTabBar::tab {
                padding: 8px 15px;
                background: #e1e5eb;
                border: 1px solid #dddddd;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-family: 'Microsoft YaHei';
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
                margin-bottom: -1px;
            }

            /* 表格样式 */
            QTableWidget {
                border: 1px solid #dddddd;
                border-radius: 4px;
                background: white;
                gridline-color: #eeeeee;
                font-family: Consolas;
            }
            QHeaderView::section {
                background-color: #4a6fa5;
                color: white;
                padding: 5px;
                border: none;
                font-weight: bold;
            }

            /* 框架样式 */
            QFrame#sectionFrame {
                background: white;
                border: 1px solid #dddddd;
                border-radius: 4px;
                padding: 5px;
            }

            /* 日志区域 */
            QTextEdit#logArea {
                background: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 5px;
                font-family: Consolas;
            }
        """)

    def init_ui(self):
        """初始化主界面"""
        self.setWindowTitle('股票策略分析系统 - 专业版')
        self.setGeometry(300, 300, 1280, 800)
        self.setWindowIcon(QIcon('assets/stock.ico'))

        # 主控件
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 标题
        title = QLabel('股票策略分析系统')
        title.setFont(QFont('Microsoft YaHei', 18, QFont.Bold))
        title.setStyleSheet("color: #2c3e50;")
        title.setAlignment(Qt.AlignCenter)

        # 标签页
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont('Microsoft YaHei', 10))

        # 添加标签页
        self.setup_single_stock_tab()
        self.setup_hot_stocks_tab()

        # 日志区域
        log_frame = QFrame()
        log_frame.setObjectName("sectionFrame")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(10, 10, 10, 10)

        log_label = QLabel("系统日志")
        log_label.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        log_label.setStyleSheet("color: #2c3e50;")

        self.log_area = QTextEdit()
        self.log_area.setObjectName("logArea")
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(120)

        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_area)

        # 组装界面
        main_layout.addWidget(title)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(log_frame)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('系统就绪 | 等待用户操作')

    def setup_single_stock_tab(self):
        """设置单股分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # 输入区域
        input_frame = QFrame()
        input_frame.setObjectName("sectionFrame")
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 10, 10, 10)

        input_title = QLabel("股票代码输入")
        input_title.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        input_title.setStyleSheet("color: #2c3e50;")

        self.single_input = QTextEdit()
        self.single_input.setPlaceholderText("请输入股票代码，每行一个...")
        self.single_input.setMaximumHeight(80)

        analyze_btn = QPushButton("分析股票")
        analyze_btn.setFont(QFont('Microsoft YaHei', 10))
        analyze_btn.clicked.connect(self.analyze_single_stock)

        input_layout.addWidget(input_title)
        input_layout.addWidget(self.single_input)
        input_layout.addWidget(analyze_btn)

        # 结果显示区域
        result_frame = QFrame()
        result_frame.setObjectName("sectionFrame")
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(10, 10, 10, 10)

        result_title = QLabel("分析结果")
        result_title.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        result_title.setStyleSheet("color: #2c3e50;")

        self.single_result = QTableWidget()
        self.single_result.setColumnCount(6)
        self.single_result.setHorizontalHeaderLabels(["代码", "名称", "当前价", "结论", "原因", "详情"])
        self.single_result.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.single_result.setAlternatingRowColors(True)
        self.single_result.verticalHeader().setVisible(False)

        # 设置列宽
        self.single_result.setColumnWidth(0, 100)  # 代码
        self.single_result.setColumnWidth(1, 100)  # 名称
        self.single_result.setColumnWidth(2, 80)  # 当前价
        self.single_result.setColumnWidth(3, 180)  # 结论
        self.single_result.setColumnWidth(4, 100)  # 原因
        self.single_result.horizontalHeader().setStretchLastSection(True)

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.single_result)

        # 组装标签页
        layout.addWidget(input_frame)
        layout.addWidget(result_frame)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "单股分析")

    def setup_hot_stocks_tab(self):
        """设置热股分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # 控制区域
        control_frame = QFrame()
        control_frame.setObjectName("sectionFrame")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 10, 10, 10)

        control_title = QLabel("热股榜分析设置")
        control_title.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        control_title.setStyleSheet("color: #2c3e50;")

        count_label = QLabel("分析数量:")
        count_label.setFont(QFont('Microsoft YaHei', 10))

        self.hot_count = QComboBox()
        self.hot_count.addItems(["20", "50", "100", "200", "500"])
        self.hot_count.setCurrentIndex(0)
        self.hot_count.setFont(QFont('Microsoft YaHei', 10))

        analyze_btn = QPushButton("分析热股榜")
        analyze_btn.setFont(QFont('Microsoft YaHei', 10))
        analyze_btn.clicked.connect(self.analyze_hot_stocks)

        control_layout.addWidget(control_title)
        control_layout.addWidget(count_label)
        control_layout.addWidget(self.hot_count)
        control_layout.addWidget(analyze_btn)

        # 结果显示区域
        result_frame = QFrame()
        result_frame.setObjectName("sectionFrame")
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(10, 10, 10, 10)

        result_title = QLabel("热股榜分析结果")
        result_title.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        result_title.setStyleSheet("color: #2c3e50;")

        self.hot_result = QTableWidget()
        self.hot_result.setColumnCount(12)
        self.hot_result.setHorizontalHeaderLabels([
            "排名", "代码", "名称", "当前价", "涨跌", "涨幅",
            "大涨日", "调整天数", "量比", "近均线", "拟合得分", "结果"
        ])
        self.hot_result.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.hot_result.setAlternatingRowColors(True)
        self.hot_result.verticalHeader().setVisible(False)

        # 设置列宽
        self.hot_result.setColumnWidth(0, 80)  # 排名
        self.hot_result.setColumnWidth(1, 100)  # 代码
        self.hot_result.setColumnWidth(2, 120)  # 名称
        self.hot_result.setColumnWidth(3, 100)  # 当前价
        self.hot_result.setColumnWidth(4, 100)  # 涨跌
        self.hot_result.setColumnWidth(5, 100)  # 涨幅
        self.hot_result.setColumnWidth(6, 100)  # 大涨日
        self.hot_result.setColumnWidth(7, 100)  # 调整天数
        self.hot_result.setColumnWidth(8, 100)  # 量比
        self.hot_result.setColumnWidth(9, 100)  # 近均线
        self.hot_result.setColumnWidth(10, 100)  # 拟合得分
        self.hot_result.setColumnWidth(11, 100)  # 结果

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.hot_result)

        # 组装标签页
        layout.addWidget(control_frame)
        layout.addWidget(result_frame)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "热股分析")

    def log(self, message, level="info"):
        """记录带时间戳和级别的日志"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")

        # 根据级别设置颜色
        if level == "error":
            colored_msg = f"[{timestamp}] <span style='color:#e74c3c;'>ERROR: {message}</span>"
        elif level == "warning":
            colored_msg = f"[{timestamp}] <span style='color:#f39c12;'>WARNING: {message}</span>"
        elif level == "success":
            colored_msg = f"[{timestamp}] <span style='color:#27ae60;'>{message}</span>"
        else:
            colored_msg = f"[{timestamp}] {message}"

        self.log_area.append(colored_msg)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

        # 更新状态栏
        self.status_bar.showMessage(f"最后操作: {timestamp} - {message}")

    def analyze_single_stock(self):
        """分析单只股票"""
        codes = self.single_input.toPlainText().strip().split('\n')
        self.single_result.setRowCount(0)  # 清空表格

        if not codes or not any(codes):
            self.log("错误: 请输入有效的股票代码", "error")
            return

        for code in codes:
            code = code.strip()
            if not code:
                continue

            self.log(f"开始分析股票 {code}...")
            try:
                result = self.analyzer.analyze_single_stock(code)
                if result:
                    row = self.single_result.rowCount()
                    self.single_result.insertRow(row)

                    # 设置表格项
                    items = [
                        QTableWidgetItem(result['code']),
                        QTableWidgetItem(result['name']),
                        QTableWidgetItem(f"{result.get('price', 0):.2f}"),
                        QTableWidgetItem("符合" if result['qualified'] else "不符合"),
                        QTableWidgetItem(result['analysis'].get('reason', '')),
                        QTableWidgetItem(str(result['analysis']))
                    ]

                    # 设置对齐方式
                    for i in [2, 3]:  # 需要右对齐的列
                        items[i].setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                    # 根据结论设置颜色
                    if result['qualified']:
                        items[3].setForeground(QColor('#27ae60'))  # 绿色
                        self.log(f"股票 {code} 分析完成 - 符合条件", "success")
                    else:
                        items[3].setForeground(QColor('#e74c3c'))  # 红色
                        self.log(f"股票 {code} 分析完成 - 不符合条件")

                    # 添加到表格
                    for col, item in enumerate(items):
                        self.single_result.setItem(row, col, item)
                else:
                    self.log(f"股票 {code} 无分析结果", "warning")
            except Exception as e:
                self.log(f"分析股票 {code} 出错: {str(e)}", "error")

    def analyze_hot_stocks(self):
        """分析热股榜"""
        try:
            top_n = int(self.hot_count.currentText())
            self.log(f"开始分析热股榜前 {top_n} 只股票...")

            # 获取分析结果
            qualified_stocks,table = self.analyzer.analyze_hot_stocks(top_n)
            self.hot_result.setRowCount(0)  # 清空表格

            # 决定展示哪些数据
            display_stocks = qualified_stocks
            display_message = f"找到 {len(qualified_stocks)} 只符合条件股票" if qualified_stocks else "无符合条件股票，展示所有分析结果"

            self.log(f"分析完成: {display_message}")
            self.log(f"共分析 {len(qualified_stocks)} 只股票")

            # 填充表格
            for idx, stock in enumerate(display_stocks, 1):
                row = self.hot_result.rowCount()
                self.hot_result.insertRow(row)

                # 创建表格项
                items = [
                    QTableWidgetItem(str(idx)),
                    QTableWidgetItem(stock['code']),
                    QTableWidgetItem(stock['name']),
                    QTableWidgetItem(f"{stock.get('price', 0):.2f}"),
                    QTableWidgetItem(f"{stock.get('change_amount', 0):.2f}"),
                    QTableWidgetItem(f"{stock.get('change_rate', 0):.2f}%"),
                    QTableWidgetItem(stock['analysis'].get('peak_day', '无')),
                    QTableWidgetItem(str(stock['analysis'].get('adjust_days', 0))),
                    QTableWidgetItem(f"{stock['analysis'].get('current_volume_ratio', 0):.2f}"),
                    QTableWidgetItem(stock['analysis'].get('near_ma', '无')),
                    QTableWidgetItem(f"{stock['analysis'].get('ma_score', 0):.2f}"),
                    QTableWidgetItem("符合" if stock['qualified'] else "不符合")
                ]

                # 设置对齐方式
                for i in [0, 3, 4, 5, 7, 8, 10, 11]:  # 需要右对齐的列
                    items[i].setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                # 设置涨跌颜色
                if stock.get('change_rate', 0) > 0:
                    items[4].setForeground(QColor('#e74c3c'))  # 涨-红色
                    items[5].setForeground(QColor('#e74c3c'))
                elif stock.get('change_rate', 0) < 0:
                    items[4].setForeground(QColor('#27ae60'))  # 跌-绿色
                    items[5].setForeground(QColor('#27ae60'))

                # 设置结果颜色
                if stock['qualified']:
                    items[11].setForeground(QColor('#27ae60'))  # 绿色
                else:
                    items[11].setForeground(QColor('#e74c3c'))  # 红色

                # 设置分数颜色
                ma_score = stock['analysis'].get('ma_score', 0)
                if ma_score > 8:
                    items[10].setForeground(QColor('#e74c3c'))  # 高分-红色
                elif ma_score > 5:
                    items[10].setForeground(QColor('#f39c12'))  # 中分-橙色

                # 添加到表格
                for col, item in enumerate(items):
                    self.hot_result.setItem(row, col, item)

            # 添加表头颜色区分
            header = self.hot_result.horizontalHeader()
            if qualified_stocks:
                header.setStyleSheet("QHeaderView::section { background-color: #27ae60; color: white; }")
                self.log("表格显示: 符合条件股票(绿色表头)", "success")
            else:
                header.setStyleSheet("QHeaderView::section { background-color: #95a5a6; color: white; }")
                self.log("表格显示: 所有分析股票(灰色表头)")
                self.log(table)

        except Exception as e:
            self.log(f"分析热股榜出错: {str(e)}", "error")
            traceback.print_exc()
            self.log_area.append(f"<span style='color:#e74c3c;'>错误详情: {str(e)}</span>")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 10))  # 设置全局字体

    # 设置应用程序样式
    app.setStyle('Fusion')

    window = StockStrategyApp()
    window.show()
    sys.exit(app.exec_())