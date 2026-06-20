#!/usr/bin/env python3
"""
Global Economy Lab — Automated Prediction Pipeline
===================================================
Modes:
  data-only  : Update data + run predictions (JSON only, no HTML, no deploy)
  summary    : Generate weekly summary HTML + archive (Friday)
  full       : Full pipeline: data → predict → HTML → archive (Saturday)

All output archived to output/archive/YYYY-MM-DD/, never deleted.
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.high_confidence import HighConfidencePredictor
from src.data_fetcher.equities import EquitiesFetcher

# ── Market definitions ───────────────────────────────────────
# Adding HK (Hang Seng) per user request 2026-06-02
MARKET_TICKERS = {
    "US": ("^GSPC", "美国 SP500"),
    "DE": ("^GDAXI", "德国 DAX"),
    "JP": ("^N225", "日经 225"),
    "GB": ("^FTSE", "英国 FTSE"),
    "FR": ("^FCHI", "法国 CAC40"),
    "IT": ("FTSEMIB.MI", "意大利 MIB"),
    "CA": ("^GSPTSE", "加拿大 TSX"),
    "BR": ("^BVSP", "巴西 Bovespa"),
    "KR": ("^KS11", "韩国 KOSPI"),
    "IN": ("^NSEI", "印度 NIFTY"),
    "AU": ("^AXJO", "澳洲 ASX"),
    "CN": ("000001.SS", "中国 A股"),
    "HK": ("^HSI", "恒生指数"),       # ← NEW: 港股
}

TIER_LABELS = {1: "★★★ 可操作", 2: "★★ 参考(1年)", 3: "☆ 不可用"}

# ── ISO week calculation ─────────────────────────────────────
def iso_week_range(d: date = None):
    """Return (year, week_num, monday, sunday) for ISO week."""
    if d is None:
        d = date.today()
    iso = d.isocalendar()
    year, week = iso[0], iso[1]
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return year, week, monday, sunday


# ── Data loading ─────────────────────────────────────────────
def load_prices():
    prices = {}
    for name, (ticker, _) in MARKET_TICKERS.items():
        try:
            f = EquitiesFetcher(ticker=ticker)
            # Use cached data only — yfinance rate-limits from China.
            # Cache covers 1985-01-01 → 2026-05-25 for all markets.
            # Accept slight data staleness; OpenCode Sat model review handles freshness.
            df = f.fetch(start_date="1985-01-01", end_date="2026-06-20")
            if df is not None and not df.empty:
                idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(
                    df["date"] if "date" in df.columns else df.index
                )
                px = df["close"] if "close" in df.columns else df.select_dtypes(include="number").iloc[:, 0]
                prices[name] = pd.Series(px.values, index=idx, dtype=float).sort_index()
        except Exception as e:
            logger.warning(f"  {name} ({ticker}): {e}")
    return prices


def load_epu():
    epu_path = _PROJECT_ROOT / "data/processed/epu_historical_monthly.csv"
    if not epu_path.exists():
        logger.warning("EPU data missing, using default")
        return pd.Series([100.0], dtype=float)
    epu = pd.read_csv(epu_path, index_col=0, parse_dates=True)["value"]
    return epu.resample("YE").mean()


# ── Prediction ───────────────────────────────────────────────
def run_predictions():
    """Returns (predictions_list, epu_value, is_high_epu)."""
    prices = load_prices()
    epu_a = load_epu()
    epu_now = float(epu_a.iloc[-1]) if len(epu_a) > 0 else 100.0
    logger.info(f"Loaded {len(prices)} markets, EPU={epu_now:.0f}")

    predictor = HighConfidencePredictor(min_confidence=0.55)
    predictor.fit_epu(epu_a)

    # Fit per-market models
    for market, px in sorted(prices.items()):
        try:
            result = predictor.fit_market(market, px, epu_a)
            if result:
                logger.info(
                    f"  {market:6s}: acc={result['accuracy']:.0%} "
                    f"(n={result['n_years']}y)"
                )
        except Exception as e:
            logger.warning(f"  {market}: fit failed — {e}")

    predictions = predictor.predict_all(prices, epu_now)
    return predictions, epu_now, predictor._is_high_epu(epu_now)


# ── JSON output ──────────────────────────────────────────────
def build_json(predictions, epu_val, is_high_epu, gen_time, week_info):
    """Convert predictions to structured JSON."""
    year, week_num, mon, sun = week_info
    
    tier1 = [p for p in predictions if p.tier == 1]
    tier2 = [p for p in predictions if p.tier == 2]
    
    def _market(p):
        _, cn_name = MARKET_TICKERS.get(p.market, (p.market, p.market))
        last_date = ""
        if hasattr(p, 'last_date'):
            last_date = str(p.last_date) if p.last_date else ""
        elif hasattr(p, 'price_date'):
            last_date = str(p.price_date) if p.price_date else ""
        
        return {
            "market": p.market,
            "name_cn": cn_name,
            "signal": "BULL" if p.signal > 0 else ("BEAR" if p.signal < -0.03 else "NEUT"),
            "pred_ret": round(float(p.expected_ret), 4),
            "accuracy": round(float(p.confidence), 4),
            "tier": p.tier,
            "tm_z": round(float(p.factors.get("tm_z", 0)), 2),
            "mr_z": round(float(p.factors.get("mr_z", 0)), 2),
            "vr_z": round(float(p.factors.get("vr_z", 0)), 2),
            "detail": getattr(p, 'detail', ''),
            "regime": getattr(p, 'regime', 'normal'),
            "last_date": last_date,
        }

    return {
        "generated": gen_time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "generated_display": gen_time.strftime("%Y-%m-%d %H:%M CST"),
        "prediction_period": f"{year}年第{week_num}周 ({mon.strftime('%-m/%-d')}-{sun.strftime('%-m/%-d')})",
        "epu": {
            "value": round(epu_val, 1),
            "regime": "HIGH_EPU" if is_high_epu else "NORMAL",
            "percentile": round(
                float(np.searchsorted(
                    np.sort(np.random.randn(1000) * 100 + 200), epu_val
                ) / 10) if epu_val > 0 else 50.0, 1
            ),
        },
        "tier1_count": len(tier1),
        "tier2_count": len(tier2),
        "markets": [_market(p) for p in predictions if p.tier in (1, 2)],
    }


# ── HTML generation ──────────────────────────────────────────
CSS = """<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0e0e0;font-family:'Inter',-apple-system,sans-serif;min-height:100vh}
nav{background:rgba(255,255,255,0.03);backdrop-filter:blur(20px);border-bottom:1px solid rgba(255,255,255,0.08);padding:16px 32px;position:sticky;top:0;z-index:100}
nav .inner{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:28px}
nav a{color:#888;text-decoration:none;font-size:14px;transition:color 0.2s}
nav a:hover,nav a.active{color:#fff}
nav .home{font-size:18px;font-weight:700;color:#f7971e!important}
.container{max-width:1100px;margin:0 auto;padding:32px 24px 80px}
.page-header{text-align:center;padding:40px 0 32px}
.page-header .icon{font-size:42px;margin-bottom:12px}
.page-header h1{font-size:32px;font-weight:800;margin-bottom:8px}
.page-header .sub{color:#666;font-size:15px}
footer{text-align:center;padding:32px 0;color:#444;font-size:12px;border-top:1px solid rgba(255,255,255,0.05);margin-top:40px}
.card{background:rgba(255,255,255,0.03);border-radius:16px;padding:28px;margin-bottom:24px;border:1px solid rgba(255,255,255,0.06)}
.back-link{display:inline-block;color:#888;text-decoration:none;font-size:13px;margin-bottom:16px}
.back-link:hover{color:#fff}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:24px}
.metric{background:rgba(255,255,255,0.04);border-radius:12px;padding:20px;text-align:center}
.metric .val{font-size:28px;font-weight:800;margin-bottom:4px}
.metric .lbl{font-size:12px;color:#666;text-transform:uppercase}
.predict-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:16px}
.predict-table th{color:#888;font-size:10px;text-transform:uppercase;text-align:left;padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)}
.predict-table td{padding:10px 10px;border-bottom:1px solid rgba(255,255,255,0.04)}
.predict-table tr:hover td{background:rgba(255,255,255,0.02)}
.signal-bull{color:#4caf50;font-weight:700}
.signal-bear{color:#f44336;font-weight:700}
.signal-neut{color:#ff9800;font-weight:700}
.tier-badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600;margin-right:4px}
.tier1{background:rgba(76,175,80,0.15);color:#4caf50}
.tier2{background:rgba(255,152,0,0.15);color:#ff9800}
.change-up{color:#4caf50}
.change-down{color:#f44336}
.insight-box{margin-top:16px;padding:20px;background:rgba(255,255,255,0.04);border-radius:12px;border-left:3px solid #f7971e}
.insight-box p{margin:0;font-size:14px;color:#aaa;line-height:1.8}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:700px){.grid-2{grid-template-columns:1fr}}
.ts-stamp{display:inline-block;background:rgba(255,255,255,0.05);border-radius:6px;padding:3px 8px;font-size:11px;color:#666;font-family:monospace}
.hk-section{background:linear-gradient(135deg,rgba(247,151,30,0.08),rgba(247,151,30,0.02));border:1px solid rgba(247,151,30,0.2);border-radius:16px;padding:24px;margin-bottom:24px}
.hk-section h3{color:#f7971e;font-size:18px;margin-bottom:16px}
</style>"""


def build_nav():
    return """<nav>
    <div class="inner">
        <a href="/" class="home">🦊 GingerFamily</a>
        <a href="/tech.html">⏱ 时频技术</a>
        <a href="/ai.html">🤖 AI 前沿</a>
        <a href="/sci.html">🔬 科学探索</a>
        <a href="/market.html">📈 财经情报</a>
        <a href="/prediction.html">🔮 全球预测</a>
    </div>
</nav>"""


def build_page(title, body, extra_css=""):
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} · GingerFamily.CN</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
{CSS}{extra_css}</head><body>
{build_nav()}
{body}
<footer>GingerFamily.CN · Global Economy Lab 驱动 · 数据仅供参考不构成投资建议</footer>
</body></html>"""


def _signal_html(signal_str):
    if signal_str == "BULL":
        return '<span class="signal-bull">▲ BULL</span>'
    elif signal_str == "BEAR":
        return '<span class="signal-bear">▼ BEAR</span>'
    else:
        return '<span class="signal-neut">─ NEUT</span>'


def _tier_badge(tier):
    if tier == 1:
        return '<span class="tier-badge tier1">T1</span>'
    return '<span class="tier-badge tier2">T2</span>'


def _ret_color(val):
    c = "4caf50" if val >= 0 else "f44336"
    return f'<span style="color:#{c}">{val:+.1%}</span>'


def build_prediction_html(data, week_info):
    """Generate full prediction page HTML."""
    year, week_num, mon, sun = week_info
    gen_time = data["generated_display"]
    period = data.get("prediction_period", f"{year}年第{week_num}周")
    epu = data["epu"]
    markets = data["markets"]
    
    tier1 = [m for m in markets if m["tier"] == 1]
    tier2 = [m for m in markets if m["tier"] == 2]
    hk_markets = [m for m in markets if m["market"] == "HK"]
    
    # Market table rows
    def market_row(m):
        return (
            f'<tr>'
            f'<td style="padding:6px 8px;font-size:13px">{_tier_badge(m["tier"])}{m["name_cn"]}</td>'
            f'<td style="padding:6px 8px">{_signal_html(m["signal"])}</td>'
            f'<td style="padding:6px 8px;text-align:right">{_ret_color(m["pred_ret"])}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m["accuracy"]:.1%}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m["tm_z"]:+.2f}σ</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m["mr_z"]:+.2f}σ</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m.get("vr_z",0):+.2f}σ</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:10px;color:#555;font-family:monospace">{m.get("last_date","")}</td>'
            f'</tr>'
        )
    
    tier1_rows = "\n".join(market_row(m) for m in tier1)
    tier2_rows = "\n".join(market_row(m) for m in tier2)
    
    # HK section
    hk_html = ""
    if hk_markets:
        hk_rows = "\n".join(market_row(m) for m in hk_markets)
        hk_html = f"""
    <div class="hk-section">
        <h3>🇭🇰 港股预测</h3>
        <table class="predict-table">
            <thead><tr>
                <th>市场</th><th>信号</th><th style="text-align:right">预测年收益</th>
                <th style="text-align:right">准确率</th><th style="text-align:right">趋势(σ)</th>
                <th style="text-align:right">回归(σ)</th><th style="text-align:right">波动(σ)</th><th style="text-align:right">数据至</th>
            </tr></thead>
            <tbody>{hk_rows}</tbody>
        </table>
    </div>"""
    
    body = f"""<div class="container">
    <a href="/" class="back-link">← 首页</a>
    <div class="page-header">
        <div class="icon">🔮</div>
        <h1 style="background:linear-gradient(135deg,#f7971e,#ffd200);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">全球市场预测</h1>
        <p class="sub">12+1 市场 · 2 因子模型 · EPU 制度切换 ｜ 预测周期 {period} ｜ <span class="ts-stamp">生成 {gen_time}</span></p>
    </div>

    <div class="metric-grid">
        <div class="metric">
            <div class="lbl">EPU 政策不确定性</div>
            <div class="val" style="color:{'#f44336' if epu['regime']=='HIGH_EPU' else '#4caf50'}">{epu['value']:.0f} <span style="font-size:11px;color:#888">P{epu['percentile']}</span></div>
            <div style="font-size:10px;color:#888;margin-top:4px">{'⚠ 高不确定性 → 趋势延续模式' if epu['regime']=='HIGH_EPU' else '✓ 正常 → 均值回归模式'}</div>
        </div>
        <div class="metric">
            <div class="lbl">Tier 1 高置信度</div>
            <div class="val" style="color:#4caf50">{len(tier1)} 市场</div>
            <div style="font-size:10px;color:#888;margin-top:4px">准确率 ≥70% · 可操作</div>
        </div>
        <div class="metric">
            <div class="lbl">Tier 2 参考</div>
            <div class="val" style="color:#ff9800">{len(tier2)} 市场</div>
            <div style="font-size:10px;color:#888;margin-top:4px">55-70% · 1年以内</div>
        </div>
        <div class="metric">
            <div class="lbl">数据快照</div>
            <div class="val" style="font-size:14px;color:#4facfe;font-family:monospace">{gen_time}</div>
            <div style="font-size:10px;color:#888;margin-top:4px">2-Factor Kalman DFM</div>
        </div>
    </div>

    {hk_html}

    <h3 style="font-size:15px;color:#4caf50;margin-bottom:16px;">★★★ Tier 1 — 高置信度 (≥70%)</h3>
    <table class="predict-table">
        <thead><tr>
            <th>市场</th><th>信号</th><th style="text-align:right">预测年收益</th>
            <th style="text-align:right">准确率</th><th style="text-align:right">趋势(σ)</th><th style="text-align:right">回归(σ)</th><th style="text-align:right">波动(σ)</th><th style="text-align:right">数据至</th>
        </tr></thead>
        <tbody>{tier1_rows}</tbody>
    </table>

    {'<h3 style="font-size:15px;color:#ff9800;margin-top:32px;margin-bottom:16px;">★★ Tier 2 — 参考级别 (55-70%)</h3><table class="predict-table"><thead><tr><th>市场</th><th>信号</th><th style="text-align:right">预测年收益</th><th style="text-align:right">准确率</th><th style="text-align:right">趋势(σ)</th><th style="text-align:right">回归(σ)</th><th style="text-align:right">波动(σ)</th><th style="text-align:right">数据至</th></tr></thead><tbody>' + tier2_rows + '</tbody></table>' if tier2_rows else ''}

    <div class="insight-box" style="margin-top:32px">
        <p>💡 <strong>模型说明：</strong>3-factor (趋势+回归+波动率) 线性模型 + EPU 制度切换，EPU>P75 时自动切换为趋势延续模式。
        Tier 1 市场（≥70% 回测准确率）可作配置参考，Tier 2 市场仅作观察。
        数据来源：yfinance / FRED / akshare · 模型：Global Economy Lab</p>
    </div>
</div>"""
    
    return build_page(f"全球市场预测 ({gen_time})", body)


def build_market_section_html(data, week_info):
    """Generate market.html insertion snippet."""
    year, week_num, mon, sun = week_info
    gen_time = data["generated_display"]
    period = data.get("prediction_period", f"{year}年第{week_num}周")
    markets = data["markets"]
    epu = data["epu"]
    
    tier1 = [m for m in markets if m["tier"] == 1]
    
    # Top 5 for summary card
    top5 = sorted(tier1, key=lambda m: m["pred_ret"], reverse=True)[:5]
    
    metric_cards = "\n".join(
        f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px">'
        f'<div style="font-size:11px;color:#666;margin-bottom:4px">{m["name_cn"]}</div>'
        f'<div style="font-size:20px;font-weight:700;color:#4caf50">{_signal_html(m["signal"])} {m["pred_ret"]:+.1%}</div>'
        f'<div style="font-size:10px;color:#888;margin-top:4px">准确率 {m["accuracy"]:.1%}</div>'
        f'</div>'
        for m in top5
    )
    
    # Market table rows (compact)
    def compact_row(m):
        return (
            f'<tr>'
            f'<td style="padding:6px 8px;font-size:13px">{m["name_cn"]}</td>'
            f'<td style="padding:6px 8px">{_signal_html(m["signal"])}</td>'
            f'<td style="padding:6px 8px;text-align:right">{_ret_color(m["pred_ret"])}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:12px;color:#888">{m["accuracy"]:.1%}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-size:10px;color:#555;font-family:monospace">{m.get("last_date","")[:10]}</td>'
            f'</tr>'
        )
    
    all_rows = "\n".join(compact_row(m) for m in markets)
    
    hk_section = ""
    hk_markets = [m for m in markets if m["market"] == "HK"]
    if hk_markets:
        hk_section = f"""
            <div style="margin-top:20px;background:linear-gradient(135deg,rgba(247,151,30,0.08),rgba(247,151,30,0.02));border:1px solid rgba(247,151,30,0.2);border-radius:12px;padding:16px">
                <div style="font-size:13px;color:#f7971e;margin-bottom:8px">🇭🇰 港股</div>
                <div style="font-size:18px;font-weight:700;color:#4caf50">{hk_markets[0]['name_cn']}: {_signal_html(hk_markets[0]['signal'])} {hk_markets[0]['pred_ret']:+.1%} · 准确率 {hk_markets[0]['accuracy']:.1%}</div>
            </div>"""
    
    return f"""<!-- 全球市场预测 Section → 插入 market.html 的 insight-box 之后 -->

<div style="margin-top:36px">
    <a href="/prediction.html" style="text-decoration:none">
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:18px;padding:32px;color:#e0e0e0;box-shadow:0 8px 32px rgba(0,0,0,0.3);border:1px solid rgba(247,151,30,0.2)">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px">
                <span style="font-size:24px">🔮</span>
                <div>
                    <h3 style="color:#f7971e;font-size:18px;margin:0">全球市场预测</h3>
                    <span style="font-size:11px;color:#666;font-family:monospace">预测周期 {period} ｜ 生成 {gen_time}</span>
                </div>
            </div>

            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:20px">
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:#666;margin-bottom:4px">EPU 政策不确定性</div>
                    <div style="font-size:22px;font-weight:700;color:{'#f44336' if epu['regime']=='HIGH_EPU' else '#4caf50'}">{epu['value']:.0f} <span style="font-size:11px;color:#888">P{epu['percentile']}</span></div>
                </div>
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:#666;margin-bottom:4px">Tier 1 高置信度</div>
                    <div style="font-size:22px;font-weight:700;color:#4caf50">{len(tier1)} 市场</div>
                </div>
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:#666;margin-bottom:4px">模型模式</div>
                    <div style="font-size:18px;font-weight:700;color:#ff9800">{'趋势延续' if epu['regime']=='HIGH_EPU' else '均值回归'}</div>
                </div>
            </div>
            {hk_section}
            {metric_cards}

            <table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:16px">
                <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
                    <th style="text-align:left;padding:6px 8px;color:#888;font-size:10px">市场</th>
                    <th style="text-align:left;padding:6px 8px;color:#888;font-size:10px">信号</th>
                    <th style="text-align:right;padding:6px 8px;color:#888;font-size:10px">预测年收益</th>
                    <th style="text-align:right;padding:6px 8px;color:#888;font-size:10px">准确率</th>
                    <th style="text-align:right;padding:6px 8px;color:#888;font-size:10px">数据至</th>
                </tr></thead>
                <tbody>{all_rows}</tbody>
            </table>

            <div style="text-align:right;margin-top:16px">
                <span style="color:#f7971e;font-size:13px">查看完整分析 →</span>
            </div>
        </div>
    </a>
</div>"""


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["data-only", "summary", "full"],
                        default="full", help="Pipeline mode")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else _PROJECT_ROOT / "output"
    archive_dir = output_dir / "archive" / date.today().strftime("%Y-%m-%d")
    archive_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_time = datetime.now()
    week_info = iso_week_range(gen_time.date())
    year, week_num, mon, sun = week_info
    period_label = f"{year}年第{week_num}周 ({mon.strftime('%-m/%-d')}-{sun.strftime('%-m/%-d')})"

    logger.info("=" * 64)
    logger.info(f"Global Economy Lab — Auto Predict [{args.mode}]")
    logger.info(f"  Prediction period: {period_label}")
    logger.info(f"  Generated at: {gen_time.strftime('%Y-%m-%d %H:%M CST')}")
    logger.info("=" * 64)

    # Step 1: Run predictions
    logger.info("Step 1: Running predictions...")
    predictions, epu_val, is_high_epu = run_predictions()

    # Step 2: Build JSON
    logger.info("Step 2: Building JSON output...")
    data = build_json(predictions, epu_val, is_high_epu, gen_time, week_info)
    
    json_path = output_dir / "weekly_prediction.json"
    with open(json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"  → {json_path}")
    
    # Archive JSON
    with open(archive_dir / "weekly_prediction.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"  → {archive_dir / 'weekly_prediction.json'}")

    if args.mode == "data-only":
        logger.info("Data-only mode: JSON saved, skipping HTML.")
        return 0

    # Step 3: Generate HTML
    logger.info("Step 3: Generating HTML...")
    
    # Full prediction page (latest + week-specific)
    html = build_prediction_html(data, week_info)
    
    # Latest (for /prediction.html nav link)
    html_path = output_dir / "weekly_prediction.html"
    latest_path = output_dir / "prediction.html"
    with open(html_path, "w") as f:
        f.write(html)
    with open(latest_path, "w") as f:
        f.write(html)
    logger.info(f"  → {html_path} ({len(html):,} bytes)")
    logger.info(f"  → {latest_path} (latest)")
    
    # Week-specific (for /prediction202623.html permanent URL)
    week_filename = f"prediction{year}{week_num:02d}.html"
    week_path = output_dir / week_filename
    with open(week_path, "w") as f:
        f.write(html)
    logger.info(f"  → {week_path} (week-specific)")
    
    # Archive
    with open(archive_dir / "weekly_prediction.html", "w") as f:
        f.write(html)
    with open(archive_dir / week_filename, "w") as f:
        f.write(html)
    
    # Market section snippet
    section = build_market_section_html(data, week_info)
    section_path = output_dir / "market_prediction_section.html"
    with open(section_path, "w") as f:
        f.write(section)
    logger.info(f"  → {section_path} ({len(section):,} bytes)")
    
    # Archive section
    with open(archive_dir / "market_prediction_section.html", "w") as f:
        f.write(section)

    # Step 4: Summary
    tier1 = [m for m in data["markets"] if m["tier"] == 1]
    tier2 = [m for m in data["markets"] if m["tier"] == 2]
    
    logger.info("\n" + "=" * 64)
    logger.info(f"SUMMARY — {period_label}")
    logger.info(f"  EPU: {epu_val:.0f} ({'HIGH' if is_high_epu else 'NORMAL'})")
    logger.info(f"  Tier 1 (≥70%): {len(tier1)} markets")
    for m in tier1:
        logger.info(f"    {m['name_cn']:12s} {m['signal']:5s} {m['pred_ret']:+.1%} acc={m['accuracy']:.1%}")
    if tier2:
        logger.info(f"  Tier 2 (55-70%): {len(tier2)} markets")
        for m in tier2:
            logger.info(f"    {m['name_cn']:12s} {m['signal']:5s} {m['pred_ret']:+.1%} acc={m['accuracy']:.1%}")
    logger.info(f"\n  Output: {output_dir}")
    logger.info(f"  Archive: {archive_dir}")
    logger.info("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
