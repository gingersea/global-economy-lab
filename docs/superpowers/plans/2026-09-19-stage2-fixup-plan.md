# 阶段2 收尾修复 — 实施计划

> 2026-09-19 · global-economy-lab · 对应 spec: 2026-09-19-hk-gb-monthly-threshold-design.md

## 任务分解

### P1: flat market 窗口月度化（瑕疵 1）
- `src/analysis/high_confidence.py` 的 `_check_flat_market`：`prices.iloc[-1]/prices.iloc[-11]`（10d）→ 月度 ~21 交易日窗口（`iloc[-22]`）
- 日志/文案 "|10d ret|" → "|月度ret|"
- 与 predict() 里"月频 flat check uses monthly-window"注释对齐

### P2: 分市场阈值 + 页面标注（瑕疵 2）
- HK/GB 的 min_confidence 降为 0.50，其余市场保持 0.55
- 实现建议：HighConfidencePredictor 加 `_market_min_confidence` dict（fit_market 里按 market 取阈值），auto_predict.py 里 `set_market_params` 或新接口传入 HK/GB 的 0.50
- `build_prediction_html.py` 对 HK/GB 行加"接近随机，仅参考"标注（低置信提示）

### P3: 市场数文案动态化（瑕疵 3）
- `build_prediction_html.py` 的"13 市场"硬编码 → 动态 `len(markets)` 市场

### P4: prediction_period 月度格式（瑕疵 4）
- `auto_predict.py` 的 prediction_period 从"第X周 (M/D-M/D)"改为月度格式（如"2026年9月"或"未来一个月"）
- `build_prediction_html.py` 里所有"第X周"周期文案统一为月度
- 注意：data-only 仍每周跑，但预测周期标签改为月度语义

### P5: 死代码清理（瑕疵 5）
- 移除 `fit_sentiment`/`_compute_sentiment_factor`/`_sentiment_active` 等 predict() 不再调用的方法/字段（残留自阶段2 简化）

## 验收标准

1. `_check_flat_market` 用月度 ~21d 窗口，文案无"10d ret"
2. HK/GB 重新出现在预测输出，市场回到 13 个
3. HK/GB 行有"接近随机，仅参考"标注
4. 页面市场数文案与实际一致（动态）
5. prediction_period 及页面周期文案为月度格式
6. fit_sentiment 等死代码已移除
7. `python scripts/auto_predict.py --mode data-only` + `build_prediction_html.py` 端到端跑通，无报错
8. 语法检查通过，改完清 __pycache__ 再验证
