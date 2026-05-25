# 统一预测系统架构

> 最后更新：2026-05-25

## 系统全景

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          数据层 (1976-2026)                              │
│  SP500 │ Gold │ WTI │ DXY │ 10Y │ CPI │ Unemp │ M2 │ Permits │ EPU ... │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      因子库 (12 factors)                                 │
│                                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │ momentum │ │  value   │ │volatility│ │  carry   │ │  macro   │     │
│  │ 趋势动量  │ │ 均值回归  │ │ 波动率   │ │ 期限利差  │ │ 宏观扩散  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │liquidity │ │ leading  │ │  policy  │ │inflation │ │  risk    │     │
│  │ 流动性   │ │ 先行指标  │ │ 政策不确定│ │ 通胀预期  │ │ 信用风险  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ 12 factor signals (daily)
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Kalman Filter / DFM                                  │
│                                                                         │
│  状态方程:  x_t = F·x_{t-1} + w_t        w_t ~ N(0, Q)                 │
│  观测方程:  y_t = H_t·x_t + v_t          v_t ~ N(0, R)                 │
│                                                                         │
│  其中:                                                                  │
│    x_t = [direction_factor, magnitude_factor]  ← 2维潜变量              │
│    y_t = [12 factor signals]                   ← 12维观测               │
│    H_t = 时变载荷 (每252天重估计, 严格因果)                              │
│                                                                         │
│  ┌─────────────────┐     ┌─────────────────┐                           │
│  │ Factor 0: 方向   │     │ Factor 1: 幅度   │                           │
│  │ sign → 涨/跌     │     │ value → 预期收益  │                           │
│  │ 60% 年准确率     │     │ 模糊区间估计      │                           │
│  └────────┬────────┘     └────────┬────────┘                           │
│           │                       │                                     │
│           └───────────┬───────────┘                                     │
│                       ▼                                                 │
│             ┌─────────────────┐                                         │
│             │ 统一预测输出     │                                         │
│             │ {-1, 0, +1}     │                                         │
│             │ × magnitude     │                                         │
│             │ × confidence    │                                         │
│             └─────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      噪声过滤 (EPU gate)                                │
│                                                                         │
│  EPU < P75 ────► 正常 ────► 预测可信 (64.9% 准确率)                     │
│  EPU > P75 ────► 噪声 ────► 输出 0 (不确定, 42% 准确率)                 │
│  EPU > P85 ────► 极端 ────► 输出 0 (纯噪声, 37.5% 准确率)               │
│                                                                         │
│  当前状态: EPU = 82.6%ile → ⚠️ 不可靠                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## 数据流（时间维度）

```
 日度因子信号 × 12         月度聚合          年度聚合
 ──────────────────────────────────────────────────
 t1  t2  t3  ...  t252  →  month_mean  →  year_mean
 │   │   │         │          │               │
 │   │   │         │          │               ├─ 方向 60% 准确
 │   │   │         │          │               ├─ 幅度 模糊区间
 │   │   │         │          │               └─ 信号 {-1,0,+1}
 │   │   │         │          │
 │   │   │         │          ├─ 日/月噪声太大 (<50%)
 │   │   │         │          └─ 仅用于积分到年
 │   │   │         │
 │   │   │         └─ 噪声过大, 不做独立预测
 │   │   │
 ▼   ▼   ▼
 仅作为 KF 的输入观测, 提供积分基础
```

## 因子可靠性矩阵

```
                    方向准确率  可靠性    经济组      文献
─────────────────────────────────────────────────────────
macro_diffusion      52.1%    可靠     macro      Stock & Watson 2002
global_composite     51.9%    可靠     macro      Ludvigson & Ng 2009
leading_indicator    51.2%    可靠     leading    Leamer 2007
volatility_regime    50.7%    可靠     volatility Schwert 1989
policy_uncertainty   50.1%    边际     policy     Baker, Bloom & Davis 2016
inflation_expect.    50.0%    边际     inflation  Faust & Wright 2013
trend_momentum       49.3%    边际     momentum   Jegadeesh & Titman 1993
mean_reversion       49.5%    边际     value      De Bondt & Thaler 1985
credit_risk          48.1%    边际     risk       Gilchrist & Zakrajsek 2012
carry_yield_curve    47.7%    不可靠   carry      Campbell & Shiller 1991
liquidity_growth     45.2%    不可靠   liquidity  Friedman & Schwartz 1963
cross_asset_momentum 43.1%    不可靠   momentum   Asness et al. 2013
```

## 预测输出格式

```json
{
  "signal": -1,           // -1=偏空, 0=中性, +1=偏多
  "magnitude": 0.046,     // 预期年收益幅度 (模糊)
  "confidence": 0.64,     // KF 协方差导出的置信度
  "regime": "normal",     // 当前 EPU 区间
  "expected_return": {
    "pessimistic": -0.072, // P25
    "median": -0.046,      // P50
    "optimistic": 0.036    // P75
  },
  "factor_loadings": {
    "macro_diffusion": 0.875,
    "trend_momentum": 0.059,
    "...": "..."
  }
}
```

## 十年期验证

```
            1980s  1990s  2000s  2010s  2020s  整体
方向准确率   60%    60%    60%    60%   33%    60%
幅度 R²     0.04   0.00   0.01   0.15   0.51   0.00
EPU 环境    正常   正常   正常   偏高   极高   混合
```

模型在 1980-2019 连续 40 年提供 60% 方向准确率。2020s 崩盘源于 COVID 政策扭曲了历史因子关系——EPU 过滤器正确识别了这一点。

## Confidence 定义

```
confidence = tanh(|state| / σ_state)

其中 σ_state 来自 KF 协方差矩阵 P 的对角线元素。
KF 每步更新 P: 新观测降低不确定性, 过程噪声增加不确定性。
confidence 反映的是 KF 对当前潜变量估计的统计信心,
而非对未来预测的准确率保证。
```

## 运行

```bash
# 一次性回灌全部历史数据
python scripts/bootstrap_history.py --start-date 1976-01-01 --force-refresh

# 运行统一预测
python scripts/run_predictions.py --start-date 1976-01-01

# 输出示例:
#   Reliability: ⚠️ 不可靠 | noise=high | driver=EPU
#   Expected accuracy: 42% (empirical)
#   EPU percentile: 82.6%
#   MONTHLY: Composite +0.00 | KF conf 47% | stagflation
#   DAILY:  Composite +0.00 | KF conf 47%
#   ▲ 10Y  ▲ SP500  ▼ gold
```
