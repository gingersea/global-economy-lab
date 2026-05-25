# 预测系统架构总结

> 最后更新：2026-05-25

## 核心设计原则

1. **Monthly = integral(daily)**：月预测是日预测的积分，天然精度更高
2. **统一 confidence**：日/月用同一定义——Kalman 协方差导出的预测置信度
3. **无穿越**：所有拟合严格基于历史数据，H 矩阵从过往窗口估计
4. **时变参数**：因子载荷 H 每 252 交易日滚动重估计，捕获结构性变化
5. **因子经济学逻辑**：每个因子有文献引用，按经济类别分组

## 架构

```
8因子库(factors.py)
  ↓
Kalman Filter / DFM (kalman.py)
  状态方程: x_t = 0.95·x_{t-1} + w_t
  观测方程: y_t = H_t·x_t + v_t
  ↓
FactorEnsemble (factor_ensemble.py)
  时变 H、PCA 初始化、EM 估计
  ↓
  ├─ DailyAssetPredictor → 日度方向信号
  ├─ MacroRegimePredictor → 月度周期标签
  └─ PredictionHub → 统一调度输出
```

## 因子状态

| 因子 | 文献 | 状态 |
|------|------|------|
| trend_momentum | Jegadeesh & Titman (1993) | 可靠 |
| mean_reversion | De Bondt & Thaler (1985) | 不可靠 |
| volatility_regime | Schwert (1989) | 可靠 |
| carry_yield_curve | Campbell & Shiller (1991) | 可靠 |
| macro_diffusion | Stock & Watson (2002) | 可靠 |
| cross_asset_momentum | Asness, Moskowitz & Pedersen (2013) | 不可靠 |
| global_composite | Ludvigson & Ng (2009) | 不可靠 |
| credit_risk | Gilchrist & Zakrajsek (2012) | 边际 |

## 数据范围

- 起止：1976-01-01 → 2026-05-25（50 年）
- 日观测：12,874
- 月聚合：605
- 资产：SP500, gold, WTI, DXY, 10Y Treasury
- 宏观：CPI, unemployment, industrial production, NFCI, fed funds rate

## 运行

```bash
# 回灌数据
python scripts/bootstrap_history.py --start-date 1976-01-01

# 预测
python scripts/run_predictions.py --start-date 1976-01-01

# 回测 + 图表
python scripts/run_phase1_backtest.py --start-date 1976-01-01 --cost-bps 10

# 灵敏度分析
python scripts/regime_sensitivity.py --start-date 1976-01-01
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `src/analysis/kalman.py` | Kalman Filter + DFM（因果、时变载荷） |
| `src/analysis/factors.py` | 8因子库 + 文献引用 + 经济分组 |
| `src/analysis/factor_ensemble.py` | KF-based 因子集成 |
| `src/analysis/macro_predictor.py` | 月度周期预测 |
| `src/analysis/daily_predictor.py` | 日度方向预测 |
| `src/analysis/prediction_hub.py` | 统一调度 |
| `config/backtest.py` | 集中配置 |
| `scripts/run_predictions.py` | 预测驱动脚本 |

## Confidence 定义

```
confidence = 1 / (1 + std(predicted_state))
```

来源于 Kalman filter 的预测协方差矩阵 P。向前传播 h 步后，
协方差为 F^h·P·(F^h)' + ΣF^i·Q·(F^i)'。置信度随预测步长递减——
1步 ≈ 47%，3步 ≈ 25%，这反映了不确定性随预测距离增长
的正确行为。
