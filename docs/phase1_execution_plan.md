# 第一阶段执行方案：全球经济观测与回测闭环

本文档用于把项目目标拆成可执行路径，优先完成“数据收集 → 数据校验 → 基础分析 → 第一阶段回测 → 仪表盘/报告输出”的闭环，后续阶段再扩展更多市场、策略与自动化能力。

## 1. 阶段目标与指导路径

### 1.1 总目标

在不依赖付费 API Key 的前提下，构建一个可复现的全球经济观测沙盘：

1. 稳定抓取核心宏观与资产数据。
2. 形成统一频率、统一字段、可追溯缓存的数据集。
3. 用基础经济周期信号验证资产轮动假设。
4. 输出第一版回测结果、关键图表与后续 TODO。

### 1.2 第一阶段范围

第一阶段只覆盖“能支撑宏观周期识别与跨资产验证”的最小闭环：

| 模块 | 第一阶段范围 | 暂不纳入 |
|---|---|---|
| 区域 | 美国为主，补充全球 GDP、欧元区 HICP、中国 CPI | 多国家全量宏观库 |
| 资产 | 美股、美债、黄金、原油、美元、VIX | 个股、行业、期权、信用债 |
| 频率 | 月度宏观 + 日度资产，回测时对齐到月度 | 高频/分钟级数据 |
| 方法 | 周期分层、滚动相关、事件复盘、规则型月度再平衡 | 机器学习择时、组合优化 |
| 输出 | 本地 Parquet 缓存、Notebook、Streamlit 图表、阶段报告 | 自动交易、实时告警 |

### 1.3 推荐推进路径

1. **确认数据口径**：以 `config/data_sources.py` 为事实源，核对 `docs/data_sources.md` 中的来源、频率、后端与历史长度。
2. **回灌历史数据**：使用 `scripts/bootstrap_history.py --start-date 2008-01-01` 建立不少于 15 年的历史样本。
3. **补齐数据质量检查**：检查缺失值、重复日期、时区、单位、频率和异常跳变。
4. **构建月度研究面板**：把日度资产收益聚合为月度，把宏观指标转化为同比、环比、扩张/收缩信号。
5. **定义经济阶段**：用 PMI、CPI、失业率、收益率曲线和金融条件构造可解释的周期标签。
6. **执行第一阶段回测**：用规则型资产配置验证不同周期标签下的资产表现。
7. **沉淀报告与 TODO**：把关键图表、结论、数据缺口与下一阶段事项写入文档。

## 2. 第一阶段数据收集方案

### 2.1 核心数据清单

下表与 `config/data_sources.py`、`docs/data_sources.md` 对齐，作为第一阶段优先抓取清单。

| 领域 | 数据源 key | 用途 | 频率 | 主后端 |
|---|---|---|---|---|
| 美国增长 | `us_gdp`, `us_industrial_production`, `us_pmi` | 增长趋势、扩张/收缩判断 | 季度/月度 | `fred_csv` |
| 美国通胀 | `us_cpi`, `us_core_cpi` | 通胀压力、滞胀/过热判断 | 月度 | `fred_csv` |
| 美国就业 | `us_unemployment` | 衰退确认、就业周期 | 月度 | `fred_csv` |
| 美国政策利率 | `us_fed_funds_rate` | 政策环境、利率周期 | 月度 | `fred_csv` |
| 美国收益率曲线 | `us_treasury_2y`, `us_treasury_10y`, `us_treasury_30y` | 曲线斜率、债券收益 | 日度 | `fred_csv` |
| 风险条件 | `us_nfci`, `vix` | 金融压力、风险偏好 | 周度/日度 | `fred_csv`, `yfinance` |
| 股票 | `sp500` | 风险资产代表 | 日度 | `yfinance` |
| 商品 | `gold`, `crude_oil_wti` | 避险与通胀资产 | 日度 | `yfinance` |
| 汇率 | `dxy`, `eurusd` | 美元周期与全球流动性 | 日度 | `yfinance` |
| 补充宏观 | `china_cpi`, `world_gdp_usd`, `ea_hicp` | 全球背景观察 | 月度/年度 | `akshare`, `worldbank`, `ecb_sdw` |

### 2.2 数据落地要求

1. 原始或标准化缓存写入 `data/processed/`，保持脚本当前的 Parquet 缓存约定。
2. 每次批量抓取应保留 manifest 信息，记录来源、时间、行数、日期范围和异常。
3. 日度资产统一保留 `date`、`open`、`high`、`low`、`close`、`volume`（如来源支持）等字段。
4. 宏观指标统一保留 `date`、`value`、`series_id`、`source` 等字段。
5. 回测输入面板统一使用月末日期索引，避免宏观月度数据与资产日度数据错位。

### 2.3 数据收集执行步骤

1. 安装依赖：`pip install -r requirements.txt`。
2. 全量回灌：`python scripts/bootstrap_history.py --start-date 2008-01-01`。
3. 按类目补抓：
   - 宏观：`python scripts/update_data.py --category macro --start-date 2008-01-01`
   - 债券：`python scripts/update_data.py --category bond --start-date 2008-01-01`
   - 股票：`python scripts/update_data.py --category equity --start-date 2008-01-01`
   - 商品：`python scripts/update_data.py --category commodity --start-date 2008-01-01`
   - 汇率：`python scripts/update_data.py --category fx --start-date 2008-01-01`
   - 情绪：`python scripts/update_data.py --category sentiment --start-date 2008-01-01`
4. 检查缓存文件数量、日期范围和空值比例。
5. 对失败源记录 fallback、错误信息和是否影响第一阶段回测。

### 2.4 数据质量验收

| 检查项 | 验收标准 |
|---|---|
| 日期覆盖 | 核心美国宏观与资产数据至少覆盖 2008-01-01 至最近可用日期 |
| 缺失率 | 月度研究面板核心字段缺失率可解释，关键资产月收益不得大面积为空 |
| 重复值 | 单个数据源同一日期不应存在重复记录 |
| 单位一致性 | 收益率、CPI、PMI、价格指数在文档中标明原始单位与转换口径 |
| 可复现性 | 重新运行抓取脚本后，缓存结构和字段保持稳定 |

## 3. 第一阶段回测方案

### 3.1 回测目标

第一阶段回测不是为了生成交易策略，而是验证项目的研究闭环：

1. 宏观周期标签是否能解释主要资产的相对表现。
2. 不同周期阶段下，股票、债券、黄金、原油、美元、现金代理的收益/波动是否符合经济直觉。
3. 数据抓取、转换、分析、可视化链路是否可复现。

### 3.2 基准与资产池

| 资产 | 数据源 key | 回测角色 |
|---|---|---|
| S&P 500 | `sp500` | 风险资产 |
| 10Y 美债收益率 | `us_treasury_10y` | 债券利率代理，后续可转为债券价格/ETF |
| 黄金 | `gold` | 避险/通胀资产 |
| WTI 原油 | `crude_oil_wti` | 周期/通胀资产 |
| DXY | `dxy` | 美元流动性代理 |
| VIX | `vix` | 风险压力观测，不直接作为多头资产 |

> 如第一阶段尚未实现债券价格转换，可先把收益率变化作为债券压力指标，在报告中明确“非真实债券总收益”限制。

### 3.3 周期信号定义

第一版周期标签采用可解释规则，优先避免过度拟合：

| 信号 | 指标 | 初始规则 |
|---|---|---|
| 增长 | PMI、工业产出同比、失业率变化 | PMI 高于/低于 50，或工业产出同比改善/走弱 |
| 通胀 | CPI 同比、核心 CPI 同比 | 同比高于过去 12 个月中位数视为通胀偏强 |
| 利率 | 联邦基金利率、10Y-2Y 曲线 | 加息/降息方向与曲线倒挂状态 |
| 风险 | NFCI、VIX | 金融条件收紧或波动率显著上升 |

周期状态先归纳为四类：

1. **复苏**：增长改善，通胀压力不高。
2. **过热**：增长较强，通胀偏强。
3. **滞胀**：增长走弱，通胀偏强。
4. **衰退/放缓**：增长走弱，通胀压力回落或金融压力上升。

### 3.4 回测规则

1. 样本区间：2008-01-01 至最近可用月末。
2. 频率：月度再平衡。
3. 信号滞后：使用上月末已知信号生成当月持仓，避免未来函数。
4. 成本假设：第一版可使用 0 成本；报告中同步列出后续加入交易成本的 TODO。
5. 对照组：
   - 等权资产组合。
   - S&P 500 买入持有。
   - 周期标签规则组合。
6. 指标：
   - 年化收益、年化波动、最大回撤、夏普比率。
   - 分周期资产平均收益与胜率。
   - 关键事件窗口表现。

### 3.5 第一版组合假设

| 周期状态 | 初始资产倾向 | 目的 |
|---|---|---|
| 复苏 | 股票、原油偏高 | 验证风险资产与周期资产弹性 |
| 过热 | 商品、现金/美元、降低久期 | 验证通胀阶段资产表现 |
| 滞胀 | 黄金、美元、防御 | 验证避险资产相对表现 |
| 衰退/放缓 | 债券代理、黄金、防御 | 验证避险与利率下行环境 |

第一阶段只要求规则透明、结果可复现，不要求收益最优。

## 4. 文档对齐要求

1. `README.md` 负责说明项目定位、快速开始和阶段路线入口。
2. `docs/data_sources.md` 负责维护数据源目录、后端、历史长度和注册状态。
3. 本文档负责维护阶段目标、执行路径、第一阶段回测目标和 TODO。
4. 若新增或删除数据源，先更新 `config/data_sources.py`，再同步 `docs/data_sources.md` 和本文档核心清单。
5. 若回测逻辑从 Notebook 固化为模块，应在 README 的“扩展指南”或新增文档中补充入口。

## 5. TODO 清单

### 5.1 数据收集

- [ ] 运行 2008 年以来的全量历史回灌并记录失败数据源。
- [x] 为核心数据源生成覆盖区间、缺失率、重复日期的检查表（`src/analysis/data_quality.py`，由 `scripts/run_phase1_backtest.py` 落地到 `data/_meta/phase1/data_quality.csv`）。
- [ ] 明确 10Y 美债收益率是否转换为债券价格或替换为可交易 ETF 数据。
- [ ] 补充数据单位说明，尤其是收益率、同比指标和价格指数。
- [ ] 记录每个 fallback 后端触发条件和错误日志样例。

### 5.2 研究面板

- [x] 统一日度资产到月末收益（`src.analysis.research_panel.daily_to_monthly_return`）。
- [x] 统一宏观指标到月度信号，并明确发布日期滞后处理（`macro_to_monthly` 支持 `level/yoy/mom/diff/yoy_diff`；`apply_signal_lag` 默认对非 `_ret` 列滞后 1 个月）。
- [x] 生成第一版月度宽表，字段包含周期信号、资产收益和风险指标（`build_monthly_panel`，端到端测试见 `tests/test_phase1.py::test_end_to_end_pipeline`）。
- [x] 增加面板构建的最小单元测试或 Notebook 校验单元（`tests/test_phase1.py`，25 个单测）。

### 5.3 回测

- [x] 实现等权组合、S&P 500 买入持有、周期规则组合三类基准（`src/backtest/strategies.py` + `src/analysis/regime_labels.py::regime_target_weights`）。
- [x] 输出年化收益、波动、最大回撤、夏普、换手率（`src/backtest/metrics.py::summary_metrics`）。
- [x] 增加信号滞后一月的防未来函数检查（`research_panel.apply_signal_lag` 默认 `lag=1`；`MonthlyBacktest.assert_no_lookahead` 辅助断言）。
- [x] 增加交易成本和再平衡日敏感性分析（`MonthlyBacktest(cost_bps=...)` 参数；CLI `--cost-bps`）。
- [ ] 形成第一版回测图表：净值曲线、回撤曲线、分周期收益热力图（已输出 `equity_curves.csv`，等待 Notebook 可视化封装）。

### 5.4 报告与复盘

- [x] 在 Notebook 或报告中记录第一阶段核心结论（产物：`data/_meta/phase1/phase1_summary.json`、`backtest_metrics.csv`、`regime_labels.csv`）。
- [ ] 列出无法解释或与经济直觉冲突的结果。
- [ ] 把关键事件窗口纳入复盘，例如 2008 金融危机、2020 疫情、2022 通胀与加息。
- [ ] 根据第一阶段结论决定第二阶段优先扩展方向。

## 6. 第一阶段完成定义

满足以下条件即可认为第一阶段完成：

1. 核心数据源可一键回灌并生成稳定缓存。
2. 月度研究面板可复现，核心字段有质量检查记录。
3. 至少完成三个基准的月度回测并输出关键指标。
4. 图表和结论能解释主要周期阶段下的资产表现。
5. TODO 已拆解到数据、研究面板、回测、报告四类，并能支撑第二阶段排期。

## 7. 执行总结（本次落地）

本次提交沿 §1.3 的推进路径，把"研究面板 → 周期标签 → 回测 → 报告"由空白补齐到可复现的离线闭环。新增模块与对应执行步骤的映射如下：

| 步骤 | 实现位置 | 关键能力 |
|---|---|---|
| 3 数据质量检查 | `src/analysis/data_quality.py` | `check_series` / `check_panel` / `summarize_report` / `write_report` |
| 4 月度研究面板 | `src/analysis/research_panel.py` | `daily_to_monthly_return` / `macro_to_monthly` / `build_monthly_panel` / `apply_signal_lag` |
| 5 周期标签 | `src/analysis/regime_labels.py` | `RegimeConfig` + `label_regimes`（PMI/CPI 平滑后分类）+ `DEFAULT_REGIME_WEIGHTS` + `regime_target_weights` |
| 6 第一阶段回测 | `src/backtest/{engine,metrics,strategies}.py` | `MonthlyBacktest` 月度再平衡、交易成本、年化收益/波动/夏普/MDD/Calmar/换手率、等权 & 单资产权重生成 |
| 7 报告与 TODO | `scripts/run_phase1_backtest.py` | 一键执行：装载缓存 → 质量报告 → 面板 → 周期 → 三类回测 → 输出 `data/_meta/phase1/*` |
| 单元测试 | `tests/test_phase1.py` | 25 个单测（合成数据 + 端到端管线），全套 72 项 pytest 通过 |

文档对齐：
- `README.md` "扩展指南"新增了"运行第一阶段闭环"章节并补充了模块布局。
- `config/data_sources.py` 字段未变，仍为事实源。
- 本文档 §5 TODO 标记已落地项，未落地项保持为 `[ ]` 供后续推进。

口径说明：
- `us_treasury_10y` 在面板中既作为月度水平宏观（用于收益率曲线），也按 `_ret` 列以"价格代理"方式参与回测。这沿用 §3.2 的临时方案——若后续接入真实债券 ETF，应同时修改 `ASSET_KEYS` 与 `DEFAULT_REGIME_WEIGHTS` 中 `us_treasury_10y_ret` 的语义。
- `regime_target_weights` 在面板缺失某资产时会自动对剩余资产重新归一化，避免触发权重不为 1 的回测错误；这一行为应在引入新资产后被覆盖性测试。

## 8. 对原计划的改进建议

执行过程中，发现下列改动可以让第一阶段研究链路更健壮、可解释、易扩展，建议在第二阶段排期之前先消化：

1. **明确"信号→持仓"的发布日期口径**：当前以"上月末已知信号生成当月持仓"作为统一规则，但 PMI / CPI 实际发布常滞后 5–15 个工作日。建议在 `DataSourceConfig` 增加 `publication_lag_days` 字段，并让 `apply_signal_lag` 按数据源分别滞后，避免"月末数据当晚就可交易"的隐含未来函数。
2. **把 10Y 收益率换成可交易债券代理**：第一阶段使用收益率水平的 pct_change 近似"债券回报"是显式的妥协，会高估长债波动。建议第二阶段引入 IEF / TLT / VGIT 任一作为 `us_treasury_etf` 数据源，并在 `regime_target_weights` 默认表中用其替换 `us_treasury_10y_ret`。
3. **周期阈值的可观测性**：`RegimeConfig` 暴露了阈值但缺少敏感性分析。建议增加一个 `scripts/regime_sensitivity.py`，对 (PMI 阈值, CPI 阈值) 网格批量跑回测，输出周期分布与分周期收益的稳健性图，避免阈值过拟合。
4. **质量报告的 manifest 联动**：当前 `data_quality.csv` 与 `data/_meta/manifest.jsonl` 互不引用。建议在 `BaseFetcher._record_manifest` 完成后追加一次 `check_series`，把缺失率与异常跳变写入 manifest，下一次抓取前先读取上一次的质量记录，做差分告警。
5. **回测指标增加分周期热力图与事件窗口**：metrics 已输出净值/回撤，但 §5.3 "分周期收益热力图"和 §5.4 "关键事件窗口"两个 TODO 仍未实现。建议新增 `src/analysis/regime_attribution.py`：按 `(regime, asset)` 聚合月度收益，并提供 2008 / 2020 / 2022 三个事件窗口的切片函数，配合 `notebooks/04_phase1_report.ipynb` 出图。
6. **CI/Notebook 一致性**：单元测试已覆盖核心逻辑，但 `notebooks/` 中的输入与新模块尚未对齐。建议第二阶段把 02 / 03 notebook 重构为先调用 `build_monthly_panel`、再调用旧分析函数，保证一份代码同时服务于研究与回测。
7. **资产池可配置化**：当前 `ASSET_KEYS` 与 `DEFAULT_REGIME_WEIGHTS` 在脚本与模块里两处硬编码。建议把它们抽到 `config/backtest.yaml`（或 Python 配置常量）一处定义，让"加一个资产"只改一行。
8. **健壮性测试覆盖外部依赖失败**：`_safe_fetch` 已对单个数据源失败做了兜底，但端到端测试未模拟"sp500 缺失但 gold 存在"等灾难场景。建议补充一组针对 `scripts/run_phase1_backtest.py::load_inputs` 的集成测试。
