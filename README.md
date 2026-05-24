# Global Economy Lab

一个用于学习和理解全球经济的 Python 数据分析框架，系统化地采集、分析、可视化全球宏观经济和金融市场数据，并通过历史复盘理解资产联动关系。

> 本项目不追求实时交易，重在提供一个可扩展的经济观测沙盘。

---

## 项目简介

Global Economy Lab 帮助研究者与学习者：

- 📊 **采集数据**：接入 FRED（美联储经济数据）、yfinance（美/港股）、akshare（A股）等多个数据源
- 🔍 **分析周期**：根据 PMI、CPI、利率等指标识别经济周期阶段（复苏 / 过热 / 滞胀 / 衰退）
- 📈 **资产联动**：计算滚动相关系数矩阵，理解股、债、商品、汇率的联动关系
- 🏛️ **事件复盘**：以历史重大事件为锚点，分析各资产在冲击前后的表现

---

## 项目结构

```
global-economy-lab/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── Makefile
├── config/
│   ├── settings.py          # 全局配置，读取环境变量
│   └── data_sources.py      # 所有数据源定义
├── data/
│   ├── raw/                 # 原始数据缓存
│   └── processed/           # 处理后数据
├── notebooks/
│   ├── 01_hello_world.ipynb
│   ├── 02_economic_dashboard.ipynb
│   └── 03_event_replay.ipynb
├── src/
│   ├── data_fetcher/        # 数据获取层
│   ├── analysis/            # 分析层
│   └── visualization/       # 可视化层
├── tests/
│   └── test_data_fetcher.py
└── scripts/
    ├── update_data.py
    └── run_dashboard.py
```

---

## 安装

### 方式一：pip（推荐快速体验）

```bash
git clone https://github.com/your-org/global-economy-lab.git
cd global-economy-lab
pip install -r requirements.txt
```

### 方式二：Poetry

```bash
git clone https://github.com/your-org/global-economy-lab.git
cd global-economy-lab
poetry install
```

---

## 配置

复制环境变量模板并填入 API 密钥：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```
FRED_API_KEY=your_fred_api_key_here
```

> 🔑 **FRED API Key 免费申请**：[https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## 快速开始

### 1. 更新本地数据缓存

```bash
python scripts/update_data.py
```

### 2. 打开探索性分析 Notebook

```bash
make notebook
# 或者
jupyter notebook notebooks/01_hello_world.ipynb
```

### 3. 启动 Streamlit 仪表盘（可选）

```bash
python scripts/run_dashboard.py
# 或者
make dashboard
```

---

## 常用 Make 命令

| 命令 | 说明 |
|------|------|
| `make install` | 安装依赖 |
| `make update-data` | 更新所有数据源 |
| `make test` | 运行单元测试 |
| `make notebook` | 启动 Jupyter Notebook |
| `make dashboard` | 启动 Streamlit 仪表盘 |
| `make clean` | 清理缓存文件 |

---

## 扩展指南

### 添加新数据源

1. 在 `config/data_sources.py` 中注册新数据源配置（名称、描述、频率）。
2. 在 `src/data_fetcher/` 中新建或修改对应 fetcher 文件，继承 `BaseFetcher`。
3. 实现 `fetch(start_date, end_date)` 方法，返回 `pd.DataFrame`。
4. 在 `scripts/update_data.py` 中将新 fetcher 加入更新列表。

### 添加新分析模块

1. 在 `src/analysis/` 中新建 Python 文件。
2. 函数接收 `pd.DataFrame` 输入，返回分析结果 DataFrame 或 dict。
3. 在对应 Notebook 中调用验证逻辑正确性。

---

## 数据来源说明

| 数据类型 | 来源库/API |
|----------|-----------|
| 美国 GDP、CPI、PMI、国债收益率 | `fredapi` (FRED) |
| A 股行情 | `akshare` |
| 美股、港股指数及个股 | `yfinance` |
| 黄金、原油、外汇 | `yfinance` |
| VIX 恐慌指数 | `yfinance` (^VIX) |

---

## 许可证

MIT License
