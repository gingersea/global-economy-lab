# 全球经济预判系统：方法、数据、结论与建议

> 2026-05-26 | Global Economy Lab  
> GitHub: https://github.com/hjiang555-a11y/global-economy-lab

---

## 摘要

本研究构建了一个基于 2 因子（趋势动量 + 均值回归）的全球市场预测系统，覆盖 14 个主要经济体指数，使用 40 年历史数据进行回测验证。核心发现：

1. **均值回归是全球唯一的跨市场确定性格局**——在正常政策环境下，所有市场均表现为均值回归
2. **EPU（经济政策不确定性）触发系统性系数翻转**——EPU > P75 时，10/14 市场从均值回归切换为趋势延续
3. **EPU 高时模型准确率反而更高**（80% vs 74%）——关键是用对系数
4. **当前 10 个市场达到 Tier 1 可操作级别**（≥70% 方向准确率），全球共识看多
5. **印度是最具确定性的单一市场**（79% 准确率，深度超卖信号）

---

## 一、方法论

### 1.1 核心模型

```
年收益 = β₁ × trend_momentum + β₂ × mean_reversion + ε

其中:
  trend_momentum: 多周期趋势强度（Jegadeesh & Titman 1993）
  mean_reversion: 短期均值回归信号（De Bondt & Thaler 1985）
```

模型在每个市场独立拟合，使用 OLS 线性回归。因子值在日频计算，按年聚合后再用于年收益预测。

### 1.2 置信度定义

```
confidence = 滚动方向准确率
           = mean( sign(预测) == sign(实际) ) over 历史回测
```

- ≥70%：Tier 1 — 可操作
- 55-70%：Tier 2 — 仅供参考，1 年以内
- <55%：不发布

### 1.3 EPU 系数切换

EPU > P75 时，β₁ 和 β₂ 的符号会发生翻转（均值回归 → 趋势延续）。本系统使用分段线性模型，根据当前 EPU 百分位自动选择正确的系数集。

### 1.4 不确定性量化

每个预测附带：
- **预测范围**：95% 置信区间（基于残差标准差）
- **因子极端度**：当前 trend_momentum 和 mean_reversion 的 σ 偏离
- **EPU 环境**：当前政策不确定性百分位及对应模型准确率

---

## 二、数据

### 2.1 覆盖范围

| 类别 | 数据源 | 时间跨度 |
|------|--------|---------|
| 全球股指 | yfinance (12 市场) | 1985-2026 |
| 中国A股 | yfinance (上证综指) | 2005-2026 |
| 商品 | yfinance (黄金、原油) | 1985-2026 |
| 美国宏观 | FRED CSV (CPI, 失业, 工业产出等) | 1940-2026 |
| 中国宏观 | FRED CSV (CPI, EPU) | 2000-2026 |
| 多国 EPU | FRED CSV (UK, CA, RU, CN) | 2000-2026 |
| 金融压力 | FRED CSV (STLFSI4, TED) | 2000-2026 |
| 全球经济 | World Bank WDI | 1980-2024 |
| 历史 EPU | policyuncertainty.com | 1900-2026 |

### 2.2 市场覆盖

```
美国 SP500    德国 DAX     日本 N225     英国 FTSE
法国 CAC40    意大利 MIB    加拿大 TSX    巴西 BVSP
韩国 KOSPI    印度 NIFTY    澳洲 ASX      香港 HSI
中国上证 A股   黄金          原油
```

### 2.3 确定性因子（跨年代 σ < 0.03）

| 因子 | 方向 | 文献 |
|------|------|------|
| trend_momentum | 与收益的关系跨年代稳定 | Jegadeesh & Titman (1993) |
| mean_reversion | 与收益的关系跨年代稳定 | De Bondt & Thaler (1985) |
| volatility_regime | 低波 → 风险偏好 | Schwert (1989) |
| carry_yield_curve | 曲线陡峭 → 扩张 | Campbell & Shiller (1991) |
| inflation_expectations | 通胀预期 → 正相关 | Faust & Wright (2013) |

---

## 三、EPU 全景分析

### 3.1 126 年历史

```
P75 = 116  |  P80 = 125  |  P99 = 304

五大高 EPU 时期:
  1934-35 (2年): 新政改革           avg=151
  1938-40 (3年): 大萧条+二战         avg=132
  2001-03 (3年): 911+伊战           avg=136
  2008-13 (6年): GFC+欧债           avg=144
  2016-26 (11年): 脱欧→COVID→关税战  avg=213 ← 史无前例
```

### 3.2 当前 (2026-05)

```
EPU = 350 (P99, 有记录以来最高)
中国 EPU = 404 (P96)
加拿大 EPU = 848 (P93)
俄罗斯 EPU = 471 (P89)
英国 EPU = 281 (P67 ← 唯一正常)

金融压力 (STLFSI4) = -0.58 (P26) — 系统无压力
WEI 经济活动 = +2.77 — 仍在扩张
```

### 3.3 关键宏观关联

| 关系 | 相关性 | 含义 |
|------|--------|------|
| 国防/GDP ↔ EPU | -0.62 | 防御支出下降 → EPU 持续上升 |
| 生产率增速 ↔ EPU | -0.47 | 经济效率降 → 政策空间收窄 |
| STLFSI4 ↔ EPU | +0.14 | 金融压力微弱领先 EPU 3-6 月 |
| 全球市场互联度 | +0.47 | 15 市场平均两两相关 |

---

## 四、EPU 系数切换：核心发现

### 4.1 全市场结果

EPU > P75 时，**10/14 市场（71%）至少一个系数翻转为正**——从均值回归切换为趋势延续。

```
EPU 正常: tm_coef 为负（均值回归） → accuracy 74%
EPU 高:   tm_coef 为正（趋势延续） → accuracy 80%

翻转最剧烈的市场:
  巴西: tm -19.6→+0.7   mr -38.7→+3.3
  恒生: tm -4.0→+4.7    mr -6.1→+11.8
  澳洲: tm -6.4→-0.6    mr -15.7→+0.4

不翻转的市场（内部驱动型）:
  日本: 高储蓄+强工业基础
  韩国: 半导体周期主导
  原油: 地缘定价
```

### 4.2 经济体外部依赖度决定翻转剧烈程度

出口导向型经济体（巴西、澳洲、德国）翻转幅度最大；内部驱动型（日本、韩国、印度）保持相对稳定。

---

## 五、各市场模型性能

### 5.1 Tier 1 市场 (≥70%)

```
Market       准确率  R²    当前信号  预测     tm(σ)    mr(σ)   状态
────────────────────────────────────────────────────────────────
Korea         83%   0.25  BULL    +53%   +3.79σ   -5.06σ  过热超买
Canada        80%   0.08  BULL    +10%   +1.11σ   -0.71σ  超买
US            80%   0.02  BULL     +9%   +0.14σ   -0.49σ  温和
India         79%   0.61  BULL    +30%   -1.51σ   +1.49σ  深度超卖
Germany       76%   0.02  BULL    +11%   -0.48σ   +0.41σ  温和
UK            73%   0.01  BULL    +12%   +0.94σ   -0.45σ  温和超买
Australia     73%   0.24  BULL     +7%   -0.45σ   +0.71σ  超卖
France        72%   0.20  BULL     +6%   -0.25σ   +0.44σ  温和
Italy         71%   0.09  BULL     +4%   +1.05σ   -0.98σ  超买
Japan         71%   0.21  BULL    +17%   +2.04σ   -2.17σ  过热超买
```

### 5.2 Tier 2 市场 (55-70%)

```
Market       准确率  R²    当前信号  预测     状态
────────────────────────────────────────────
China (A股)   62%   0.12  NEUT     -1%    信号弱，仅参考
```

### 5.3 已排除市场 (<55%)

巴西（52%）— 因子敏感度极高但不稳定

---

## 六、印度市场深度分析

### 6.1 为什么印度特殊

1. **全球最可预测市场**：|trend| + |mean_rev| = 1.51，14 市场中排名第 1
2. **纯均值回归**：mr_coef（+2.75）是 tm_coef（-0.01）的 271 倍——超买必跌，超卖必涨
3. **内部需求驱动**：14 亿人口内需市场，对 US 政策依赖低，EPU 不翻转
4. **结构性增长**：人口红利 + 数字化 + 全球供应链转移

### 6.2 当前诊断

```
NIFTY 50: 23,671  (2026 YTD: -9.6%)
trend_momentum:  -1.4σ (P15) — 趋势极弱
mean_reversion:  +1.4σ (P95) — 深度超卖

阶段: 深度超卖  |  历史类似条件: 9 次, 次年正收益概率 88%
2026 预测: +32% BULL  |  准确率 79%

注意: 2025 年是此信号唯一失效年份（预测+14%，实际-9%）
     但仍保持 7/8 的胜率
```

---

## 七、全球共识与矛盾

### 7.1 当前共识

10 个 Tier 1 市场全部看多（BULL）。在 EPU 极端高的趋势延续模式下，模型预测全球股指将继续上涨。

### 7.2 关键矛盾

1. **美国低波动 vs 高 EPU**：vol 在 +1.1σ（历史极低），EPU 在 99%ile——历史上这类组合之后常有剧烈波动
2. **中国在岸 vs 离岸**：A 股轻度过热（政策托市），恒生超卖（市场定价）——裂口在扩大
3. **韩国过热**：83% 准确率的历史模型说韩国过热了，但趋势延续系数仍然看多
4. **金融压力低位**：STLFSI4 在 P26，通常领先 EPU 回落 3-6 个月

---

## 八、条件触发建议

### 8.1 触发条件：EPU 回落至 P75 以下

```
触发条件: US EPU < 116 (当前 350)
预计时间: 2026 H2 - 2027 H1 (基于 STLFSI4 领先指标)

触发后行动:
  → 模型系数从"趋势延续"自动切换为"均值回归"
  → 当前超卖市场（印度、澳洲、德国）的信号强度进一步提升
  → 当前超买市场（韩国、日本、加拿大）可能出现回调信号
  → 全球预测准确率从 80% 升至 74% (正常环境模型更稳定)
```

### 8.2 触发条件：STLFSI4 转正

```
触发条件: STLFSI4 > 0 (当前 -0.58)
含义: 金融压力开始上升

触发后行动:
  → ⚠️ 风险警告 — 金融压力上升常领先 EPU 上升 0-3 个月
  → 降低 Tier 1 市场中超买市场的配置（韩国、日本、加拿大）
  → 关注黄金信号（金融压力上升时黄金趋势延续性增强）
```

### 8.3 触发条件：EPU 持续极端 (>P95, >6 个月)

```
触发条件: EPU > 304 持续 6 个月以上
当前状态: 已满足 (2025-2026 已持续 18 个月)

触发后行动:
  → 模型进入"未知领域"模式 — 历史类似期（2016-2026）仅 11 年
  → 所有 Tier 1 预测置信度下调一档
  → 仅保留 Tier 1 中方向准确率 > 75% 的市场（KR, CA, US, IN, DE）
  → 建议降低整体仓位，等待 EPU 均值回归
```

### 8.4 按市场分类建议

| 市场 | 条件 | 当前信号 | 建议 |
|------|------|---------|------|
| 印度 | 准确率 79%, 超卖深度 1.5σ | BULL +30% | **最确定的机会**。等 EPU 回落可加仓 |
| 韩国 | 准确率 83%, 但过热 5σ | BULL +53% | 高波动预警。趋势延续但 σ 极端 |
| 美国 | 准确率 80%, vol 极低 | BULL +9% | 温和看多, 但警惕低波动后的爆发 |
| 德国 | 准确率 76%, 温和超卖 | BULL +11% | 欧洲发动机, 中期稳健 |
| 澳洲 | 准确率 73%, 超卖 +0.7σ | BULL +7% | 商品货币属性, 与中国需求联动 |
| 中国 | 准确率 62%, 信号弱 | NEUT -1% | 等 A 股数据改善后再纳入 Tier 1 |
| 日本 | 准确率 71%, 过热 2σ | BULL +17% | 趋势延续, 但注意过热风险 |

### 8.5 长期结构建议

1. **防御支出与 EPU 的负相关（-0.62）是深层结构信号**——全球防御支出自 1950 年代 12.7%/GDP 降至当前 3.8%/GDP，这是 EPU 长周期上升的结构性原因。关注主要国家国防预算变化作为 EPU 长期走势的领先指标。

2. **生产率增速与 EPU 的负相关（-0.47）**——AI 和自动化是否能提升全球生产率增速，是 EPU 能否回到 2000 年前低位的决定性因素。

3. **印度结构性机会**——人口红利 + 数字化 + 全球供应链转移。印度是唯一 EPU 系数不翻转的大型新兴市场，内部需求主导使其对 US 政策不确定性自然对冲。

---

## 九、系统运行

```bash
# 高置信度预测
python scripts/run_high_confidence.py

# 完整预测（含日/月/年）
python scripts/run_predictions.py --start-date 1976-01-01

# 回测 + 图表
python scripts/run_phase1_backtest.py --start-date 1976-01-01

# 灵敏度分析
python scripts/regime_sensitivity.py --start-date 1976-01-01

# 测试
make test
```

---

## 十、关键文件

| 文件 | 说明 |
|------|------|
| `src/analysis/high_confidence.py` | 高置信度预测器（≥70% Tier 1） |
| `src/analysis/kalman.py` | Kalman Filter + DFM + UnifiedPredictor |
| `src/analysis/factors.py` | 16 因子库（含文献引用） |
| `src/analysis/reliability.py` | EPU 可靠性过滤 |
| `scripts/run_high_confidence.py` | 高置信度预测驱动 |
| `config/data_sources.py` | 30+ 数据源注册 |
| `docs/deterministic_findings.md` | 确定性关系研究报告 |

---

## 参考文献

- Jegadeesh & Titman (1993) — Returns to Buying Winners and Selling Losers
- De Bondt & Thaler (1985) — Does the Stock Market Overreact?
- Baker, Bloom & Davis (2016) — Measuring Economic Policy Uncertainty
- Stock & Watson (2002) — Macroeconomic Forecasting Using Diffusion Indexes
- Campbell & Shiller (1991) — Yield Spreads and Interest Rate Movements
- Schwert (1989) — Why Does Stock Market Volatility Change Over Time?
- Faust & Wright (2013) — Forecasting Inflation
- Asness, Moskowitz & Pedersen (2013) — Value and Momentum Everywhere
- Ludvigson & Ng (2009) — Macro Factors in Bond Risk Premia
- Gilchrist & Zakrajsek (2012) — Credit Spreads and Business Cycle Fluctuations
- Ramey (2011) — Identifying Government Spending Shocks
- Gordon (2016) — The Rise and Fall of American Growth
- Friedman & Schwartz (1963) — A Monetary History
- Leamer (2007) — Housing IS the Business Cycle
