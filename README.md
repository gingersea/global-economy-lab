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

当前推进路径见 [`docs/phase1_execution_plan.md`](docs/phase1_execution_plan.md)，其中细化了第一阶段的数据收集、回测目标、文档对齐方式和 TODO 清单。

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
│   ├── processed/           # 标准 Parquet 缓存
│   ├── snapshots/           # 不可变历史快照（可选）
│   └── _meta/               # 抓取 manifest / 元数据
├── notebooks/
│   ├── 01_hello_world.ipynb
│   ├── 02_economic_dashboard.ipynb
│   └── 03_event_replay.ipynb
├── src/
│   ├── data_fetcher/        # 数据获取层（含 backends/ 免费后端）
│   ├── analysis/            # 分析层（含 data_quality / research_panel / regime_labels）
│   ├── backtest/            # 月度回测引擎、指标、参考策略
│   └── visualization/       # 可视化层
├── tests/
│   ├── test_backends.py
│   ├── test_data_fetcher.py
│   └── test_phase1.py       # 数据质量 / 面板 / 周期 / 回测 端到端单测
└── scripts/
    ├── bootstrap_history.py
    ├── update_data.py
    ├── run_phase1_backtest.py
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

> ✅ **本项目无需任何 API Key 即可运行。**
> 美国宏观与国债数据通过 FRED 公共 CSV 端点（`fred_csv` 后端）和美国财政部 Daily Par Yield Curve CSV（`treasury_gov` 后端）拉取；全球宏观经由 World Bank / ECB SDW 公开 API；股指、商品、汇率使用 yfinance / akshare / stooq。
>
> 详见 [`docs/data_sources.md`](docs/data_sources.md)。

复制环境变量模板（可选）：

```bash
cp .env.example .env
```

`.env` 中所有 Key 均为可选。如需使用 `fredapi` 作为 FRED CSV 的备份路径，可填入：

```
FRED_API_KEY=your_fred_api_key_here
```

> 🔑 **FRED API Key 免费申请（可选）**：[https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## 快速开始

### 0. 一次性回灌 ≥15 年历史（可选）

```bash
python scripts/bootstrap_history.py --start-date 2008-01-01
```

### 1. 更新本地数据缓存

```bash
python scripts/update_data.py
# 按类目 / 区域 / 起始日期：
python scripts/update_data.py --category macro --region US --since 2010-01-01
# 等价写法：
python scripts/update_data.py --category macro --region US --start-date 2010-01-01
# 指定 legacy 分组、结束日期并跳过缓存：
python scripts/update_data.py --sources macro,bonds --end-date 2024-12-31 --force-refresh
```

`scripts/update_data.py` 支持 `--start-date`（或别名 `--since`）、`--end-date`、`--sources`、`--category`、`--region`、`--force-refresh`。其中 `--category` / `--region` 按 `config/data_sources.py` 注册表筛选；`--sources` 使用脚本内置的 legacy 分组（`macro`、`equities`、`bonds`、`commodities`、`fx`、`sentiment`）。

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
4. 默认通过 `scripts/update_data.py --category ... --region ...` 走注册表更新；只有需要加入 legacy `--sources` 分组时，才修改脚本内置分组。

### 添加新分析模块

1. 在 `src/analysis/` 中新建 Python 文件。
2. 函数接收 `pd.DataFrame` 输入，返回分析结果 DataFrame 或 dict。
3. 在对应 Notebook 中调用验证逻辑正确性。

### 运行第一阶段闭环

第一阶段的"数据校验 → 月度面板 → 周期标签 → 回测 → 报告"链路由 `scripts/run_phase1_backtest.py` 串起：

```bash
# 1. 先回灌历史数据（可选，若缓存已存在可跳过）
python scripts/bootstrap_history.py --start-date 2008-01-01

# 2. 运行第一阶段闭环（输出落到 data/_meta/phase1/）
python scripts/run_phase1_backtest.py --start-date 2008-01-01 --cost-bps 10
```

产物：

| 文件 | 说明 |
|---|---|
| `data_quality.csv` | 每个数据源的覆盖区间、缺失率、重复日期、异常跳变 |
| `monthly_panel.csv` | 月末对齐的研究宽表（资产 `_ret` 列 + 宏观信号列） |
| `regime_labels.csv` | 4 状态周期标签（复苏 / 过热 / 滞胀 / 衰退） |
| `backtest_metrics.csv` | 等权、SP500 买入持有、周期规则三组合的指标 |
| `equity_curves.csv` | 各组合月度净值曲线 |
| `phase1_summary.json` | 一站式汇总（执行参数、质量摘要、指标） |

新模块 API：

| 模块 | 主要入口 |
|---|---|
| `src.analysis.data_quality` | `check_panel(panels)` / `summarize_report(report)` |
| `src.analysis.research_panel` | `build_monthly_panel(assets, macros)` / `apply_signal_lag(panel, lag=1)` |
| `src.analysis.regime_labels` | `label_regimes(panel)` / `regime_target_weights(panel)` |
| `src.backtest` | `MonthlyBacktest(returns, cost_bps).run(weights)` / `summary_metrics(returns, weights)` |

阶段目标、TODO 与改进建议见 [`docs/phase1_execution_plan.md`](docs/phase1_execution_plan.md)。

---

## 数据来源说明

`config/data_sources.py` 是数据源注册表与事实源；下表只汇总当前已注册数据源和默认后端。默认运行不需要 API Key，少数后端（如 `fredapi`）仅作为可选 fallback 或覆盖率增强。

第一阶段优先围绕美国宏观、主要跨资产价格与风险指标形成研究闭环；详细执行方案和验收标准见 [`docs/phase1_execution_plan.md`](docs/phase1_execution_plan.md)。

| 数据类型 | 来源库/API |
|----------|-----------|
| 美国 GDP、CPI、PMI、失业率、工业产出、联邦基金利率、NFCI | `fred_csv`（FRED 公共 CSV）；部分指标可选 `fredapi` fallback |
| 美国 2Y / 10Y / 30Y 国债收益率 | `fred_csv`；可选 `treasury_gov` / `fredapi` fallback |
| 中国 CPI | `akshare` |
| 全球 GDP、欧元区 HICP | `worldbank` / `ecb_sdw` |
| 美股、A 股、港股、日股指数 | `yfinance`；部分指数可选 `akshare` / `stooq` fallback |
| 黄金、白银、WTI 原油 | `yfinance`；可选 `stooq` fallback |
| DXY、EUR/USD | `yfinance`；部分汇率可选 `ecb_sdw` / `stooq` fallback |
| VIX 恐慌指数 | `yfinance`；可选 `stooq` fallback |

---

## 许可证

MIT License
