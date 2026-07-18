#!/usr/bin/env python3
"""
Build full-featured prediction.html with all sections:
  - 上周总结 (last week review)
  - 上月总结 (monthly review)
  - 本周预测 (current prediction)
  - YTD bar chart
  - Key judgments
  - Next week outlook

Data sources:
  - output/weekly_prediction.json (current prediction)
  - output/archive/YYYY-MM-DD/weekly_prediction.json (historical predictions)
  - yfinance actual market returns
"""

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_rolling_accuracy(num_weeks=4):
    """Read last N archive prediction JSONs and compute rolling actual accuracy.

    Returns dict with:
      - rolling_hit_rate: average actual_accuracy.hit_rate over available weeks
      - weekly_rates: list of (period, hit_rate, hits, total) tuples
      - tier1_rolling: average Tier1-only hit rate
      - per_market: {market_code: rolling_hit_rate} for each market
    """
    archive_dir = OUTPUT_DIR / "archive"
    if not archive_dir.exists():
        return {"rolling_hit_rate": None, "weekly_rates": [], "tier1_rolling": None, "per_market": {}}

    # Get sorted archive dirs (newest first)
    dirs = sorted(
        [d for d in archive_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    if not dirs:
        return {"rolling_hit_rate": None, "weekly_rates": [], "tier1_rolling": None, "per_market": {}}

    weekly_rates = []
    per_market_hits = {}   # market → total_hits
    per_market_total = {}  # market → total_scored
    tier1_hits = 0
    tier1_total = 0

    for d in dirs[:num_weeks]:
        json_path = d / "weekly_prediction.json"
        if not json_path.exists():
            continue
        try:
            data = load_json(json_path)
        except Exception:
            continue

        aa = data.get("actual_accuracy", {})
        if not aa or aa.get("hit_rate") is None:
            continue

        period = data.get("prediction_period", d.name)
        weekly_rates.append({
            "period": period,
            "hit_rate": aa["hit_rate"],
            "hits": aa.get("hits", 0),
            "total": aa.get("total_scored", 0),
        })

        # Per-market breakdown
        markets_detail = aa.get("markets", {})
        for mkt_code, mkt_info in markets_detail.items():
            hit = mkt_info.get("hit")
            if hit is not None:
                per_market_hits[mkt_code] = per_market_hits.get(mkt_code, 0) + (1 if hit else 0)
                per_market_total[mkt_code] = per_market_total.get(mkt_code, 0) + 1

        # Tier1 breakdown (from the prediction itself)
        for m in data.get("markets", []):
            if m.get("tier") == 1:
                mkt_code = m["market"]
                mkt_hit = markets_detail.get(mkt_code, {}).get("hit")
                if mkt_hit is not None:
                    tier1_hits += 1 if mkt_hit else 0
                    tier1_total += 1

    total_hits = sum(r["hits"] for r in weekly_rates)
    total_scored = sum(r["total"] for r in weekly_rates)

    rolling_hit_rate = round(total_hits / total_scored, 4) if total_scored > 0 else None
    tier1_rolling = round(tier1_hits / tier1_total, 4) if tier1_total > 0 else None

    per_market = {}
    for mkt, hits in per_market_hits.items():
        total = per_market_total.get(mkt, 0)
        if total > 0:
            per_market[mkt] = round(hits / total, 4)

    return {
        "rolling_hit_rate": rolling_hit_rate,
        "weekly_rates": weekly_rates,
        "tier1_rolling": tier1_rolling,
        "per_market": per_market,
        "total_hits": total_hits,
        "total_scored": total_scored,
    }


def signal_html(sig):
    if sig == "BULL":
        return '<span class="signal-bull">▲ BULL</span>'
    elif sig == "BEAR":
        return '<span class="signal-bear">▼ BEAR</span>'
    else:
        return '<span class="signal-neut">─ NEUT</span>'


def tier_badge(t):
    return '<span class="tier-badge tier1">T1</span>' if t == 1 else '<span class="tier-badge tier2">T2</span>'


def pct_str(v, signed=True):
    if v is None or (isinstance(v, str) and v == "N/A"):
        return "N/A"
    try:
        if signed:
            return f"{float(v):+.2f}%"
        return f"{float(v):.2f}%"
    except (ValueError, TypeError):
        return "N/A"


def build_page(title, body, extra_css=""):
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} · GingerFamily.CN</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0f;color:#e0e0e0;font-family:'Inter',-apple-system,sans-serif;min-height:100vh}}
nav{{background:rgba(255,255,255,0.03);backdrop-filter:blur(20px);border-bottom:1px solid rgba(255,255,255,0.08);padding:16px 32px;position:sticky;top:0;z-index:100}}
nav .inner{{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:28px;flex-wrap:wrap}}
nav a{{color:#888;text-decoration:none;font-size:14px;transition:color 0.2s;white-space:nowrap}}
nav a:hover,nav a.active{{color:#fff}}
nav .home{{font-size:18px;font-weight:700;color:#f7971e!important}}
.container{{max-width:1100px;margin:0 auto;padding:32px 24px 80px}}
.page-header{{text-align:center;padding:40px 0 32px}}
.page-header .icon{{font-size:42px;margin-bottom:12px}}
.page-header h1{{font-size:32px;font-weight:800;margin-bottom:8px}}
.page-header .sub{{color:#666;font-size:15px}}
footer{{text-align:center;padding:32px 0;color:#444;font-size:12px;border-top:1px solid rgba(255,255,255,0.05);margin-top:40px}}
.card{{background:rgba(255,255,255,0.03);border-radius:16px;padding:28px;margin-bottom:24px;border:1px solid rgba(255,255,255,0.06)}}
.back-link{{display:inline-block;color:#888;text-decoration:none;font-size:13px;margin-bottom:16px}}
.back-link:hover{{color:#fff}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:24px}}
.metric{{background:rgba(255,255,255,0.04);border-radius:12px;padding:20px;text-align:center}}
.metric .val{{font-size:28px;font-weight:800;margin-bottom:4px}}
.metric .lbl{{font-size:12px;color:#666;text-transform:uppercase}}
.predict-table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:16px;overflow-x:auto}}
.predict-table th{{color:#888;font-size:10px;text-transform:uppercase;text-align:left;padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08)}}
.predict-table td{{padding:10px 10px;border-bottom:1px solid rgba(255,255,255,0.04)}}
.predict-table tr:hover td{{background:rgba(255,255,255,0.02)}}
.signal-bull{{color:#4caf50;font-weight:700}}
.signal-bear{{color:#f44336;font-weight:700}}
.signal-neut{{color:#888;font-weight:700}}
.tier-badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600;margin-right:4px}}
.tier1{{background:rgba(76,175,80,0.15);color:#4caf50}}
.tier2{{background:rgba(255,152,0,0.15);color:#ff9800}}
.change-up{{color:#4caf50}}
.change-down{{color:#f44336}}
.insight-box{{margin-top:16px;padding:20px;background:rgba(255,255,255,0.04);border-radius:12px;border-left:3px solid #f7971e}}
.insight-box p{{margin:0;font-size:14px;color:#aaa;line-height:1.8}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:700px){{.grid-2{{grid-template-columns:1fr}}nav .inner{{gap:12px}}nav a{{font-size:12px}}nav .home{{font-size:15px}}}}
.ts-stamp{{display:inline-block;background:rgba(255,255,255,0.05);border-radius:6px;padding:3px 8px;font-size:11px;color:#666;font-family:monospace}}
.hk-section{{background:linear-gradient(135deg,rgba(247,151,30,0.08),rgba(247,151,30,0.02));border:1px solid rgba(247,151,30,0.2);border-radius:16px;padding:24px;margin-bottom:24px}}
.hk-section h3{{color:#f7971e;font-size:18px;margin-bottom:16px}}
.review-card{{background:rgba(255,255,255,0.025);border-radius:16px;padding:24px;margin-bottom:24px;border:1px solid rgba(255,255,255,0.05)}}
.review-card h2{{font-size:18px;font-weight:700;margin-bottom:12px;color:#e0e0e0}}
.review-card .summary-stats{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}}
.review-card .stat{{background:rgba(255,255,255,0.04);border-radius:10px;padding:14px 20px;text-align:center;min-width:120px}}
.review-card .stat .n{{font-size:26px;font-weight:800;margin-bottom:2px}}
.review-card .stat .l{{font-size:11px;color:#666}}
.correct{{color:#4caf50}}
.wrong{{color:#f44336}}
.passive{{color:#888}}
.ytd-chart{{margin-top:16px}}
.ytd-row{{display:flex;align-items:center;margin-bottom:10px;gap:8px}}
.ytd-label{{width:110px;text-align:right;font-size:12px;color:#aaa;flex-shrink:0}}
.ytd-bar-wrap{{flex:1;height:22px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden}}
.ytd-bar{{height:100%;border-radius:4px;min-width:2px;display:flex;align-items:center;padding-left:8px;font-size:11px;font-weight:600;transition:width 0.5s ease;white-space:nowrap}}
.ytd-bar.positive{{background:linear-gradient(90deg,rgba(76,175,80,0.7),rgba(76,175,80,0.4))}}
.ytd-bar.negative{{background:linear-gradient(90deg,rgba(244,67,54,0.7),rgba(244,67,54,0.4))}}
.ytd-val{{width:65px;text-align:left;font-size:12px;font-weight:600;flex-shrink:0}}
.overheat-tag{{display:inline-block;background:rgba(244,67,54,0.15);color:#f44336;border-radius:4px;padding:2px 6px;font-size:10px;margin-left:4px}}
.oversold-tag{{display:inline-block;background:rgba(76,175,80,0.15);color:#4caf50;border-radius:4px;padding:2px 6px;font-size:10px;margin-left:4px}}
.cn-note{{font-size:10px;color:#666;display:block;margin-top:2px}}
.cn-row td{{opacity:0.7}}
.accuracy-summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px}}
.acc-block{{background:rgba(255,255,255,0.03);border-radius:10px;padding:16px}}
.acc-block h4{{font-size:13px;color:#888;margin-bottom:8px}}
.acc-block .correct-list,.acc-block .wrong-list,.acc-block .neutral-list{{font-size:12px;line-height:1.6}}
h2.section-title{{font-size:18px;font-weight:700;margin:36px 0 20px;color:#e0e0e0;border-left:3px solid #f7971e;padding-left:12px}}
.outlook-card{{background:linear-gradient(135deg,rgba(79,172,254,0.05),rgba(79,172,254,0.02));border:1px solid rgba(79,172,254,0.15);border-radius:16px;padding:24px;margin-bottom:24px}}
.outlook-card h3{{color:#4facfe;font-size:18px;margin-bottom:16px}}
.warning-box{{background:rgba(244,67,54,0.08);border:1px solid rgba(244,67,54,0.2);border-radius:10px;padding:16px;margin-top:16px}}
.warning-box p{{font-size:13px;color:#f44336;margin:0;line-height:1.6}}
@media(max-width:600px){{.review-card .summary-stats{{flex-direction:column;gap:8px}}.review-card .stat{{min-width:unset}}}}
</style>{extra_css}</head><body>
<nav>
    <div class="inner">
        <a href="/" class="home">GingerFamily</a>
        <a href="/tech.html">时频技术</a>
        <a href="/ai.html">AI 前沿</a>
        <a href="/sci.html">科学探索</a>
        <a href="/market.html">财经情报</a>
        <a href="/prediction.html" class="active">全球预测</a>
    </div>
</nav>
{body}
<footer>GingerFamily.CN · Global Economy Lab 驱动 · 数据仅供参考不构成投资建议</footer>
</body></html>"""


# ── Actual market data (fetched from yfinance on 2026-07-05) ──

# W27 actual returns (6/29-7/5) from actual_accuracy in weekly_prediction.json
W27_ACTUAL = {
    "DE": {"ret": 4.2}, "JP": {"ret": -1.34}, "GB": {"ret": 1.38},
    "FR": {"ret": 1.28}, "IT": {"ret": 2.59}, "CA": {"ret": 0.18},
    "BR": {"ret": -0.29}, "KR": {"ret": -7.8}, "IN": {"ret": 1.3},
    "AU": {"ret": -1.12}, "CN": {"ret": 0.12}, "HK": {"ret": 1.81},
}

JUNE_ACTUAL = {
    "US": -1.06, "DE": -0.43, "JP": 5.63, "GB": 0.84,
    "FR": 2.70, "IT": 3.29, "CA": 0.25, "BR": -1.02,
    "KR": 0.00, "IN": 1.35, "AU": 0.54, "CN": 0.63,
    "HK": -9.14,
}

YTD = {
    "KR": 87.68, "JP": 34.56, "IT": 16.41, "CA": 10.55,
    "US": 9.11, "BR": 8.55, "GB": 7.31, "DE": 5.05,
    "FR": 3.82, "AU": 1.34, "CN": 0.50, "IN": -7.82,
    "HK": -11.35,
}

MARKET_NAMES = {
    "US": "美国 SP500", "DE": "德国 DAX", "JP": "日经 225",
    "GB": "英国 FTSE", "FR": "法国 CAC40", "IT": "意大利 MIB",
    "CA": "加拿大 TSX", "BR": "巴西 Bovespa", "KR": "韩国 KOSPI",
    "IN": "印度 NIFTY", "AU": "澳洲 ASX", "CN": "中国 A股",
    "HK": "恒生指数",
}


def _rolling_accuracy_html(rolling):
    """Generate HTML metric cards for rolling 4-week accuracy."""
    if rolling is None or rolling.get("rolling_hit_rate") is None:
        return ""

    hr = rolling["rolling_hit_rate"]
    t1_hr = rolling.get("tier1_rolling")
    weeks = rolling.get("weekly_rates", [])
    n_weeks = len(weeks)
    total_hits = rolling.get("total_hits", 0)
    total_scored = rolling.get("total_scored", 0)

    acc_color = "#4caf50" if hr >= 0.70 else ("#ff9800" if hr >= 0.55 else "#f44336")
    t1_color = "#4caf50" if (t1_hr is not None and t1_hr >= 0.70) else ("#ff9800" if (t1_hr is not None and t1_hr >= 0.55) else "#f44336")

    # Build weekly breakdown tooltip
    weekly_parts = []
    for w in weeks:
        wr = w["hit_rate"]
        wc = "#4caf50" if wr >= 0.70 else ("#ff9800" if wr >= 0.55 else "#f44336")
        weekly_parts.append(
            f'<span style="color:{wc}">{w["period"]}: {wr:.0%} ({w["hits"]}/{w["total"]})</span>'
        )
    weekly_breakdown = " · ".join(weekly_parts) if weekly_parts else "暂无数据"

    tier1_str = ""
    if t1_hr is not None:
        tier1_str = f'<div style="font-size:10px;color:#888;margin-top:2px">Tier1 滚动 {t1_hr:.0%}</div>'

    return f"""
        <div class="metric">
            <div class="lbl">滚动{n_weeks}周实际准确率</div>
            <div class="val" style="color:{acc_color}">{hr:.0%}</div>
            <div style="font-size:10px;color:#888;margin-top:4px">{total_hits}/{total_scored} 方向正确</div>
            {tier1_str}
        </div>
        <div class="metric">
            <div class="lbl">滚动{n_weeks}周明细</div>
            <div class="val" style="font-size:12px;line-height:1.4">{weekly_breakdown}</div>
            <div style="font-size:10px;color:#888;margin-top:4px">← 旧 | 新 →</div>
        </div>"""


def compare_prediction(pred_markets, actual_dict, pred_source_label="预测"):
    """Compare predicted signals vs actual returns. Returns (correct, wrong, neutral) lists.
    
    Logic:
      - BULL + UP -> correct
      - BULL + DOWN/FLAT -> wrong
      - BEAR + DOWN -> correct
      - BEAR + UP/FLAT -> wrong
      - NEUT -> neutral
    """
    correct = []
    wrong = []
    neutral = []
    for m in pred_markets:
        code = m["market"]
        name = m["name_cn"]
        signal = m["signal"]
        tier = m["tier"]
        actual_entry = actual_dict.get(code)
        if isinstance(actual_entry, dict):
            actual_ret = actual_entry.get("ret") if actual_entry else None
        else:
            actual_ret = actual_entry
        
        if actual_ret is None:
            neutral.append((m, None, "N/A"))
            continue
        
        actual_dir = "UP" if actual_ret > 0.3 else ("DOWN" if actual_ret < -0.3 else "FLAT")
        predicted_dir = "UP" if signal == "BULL" else ("DOWN" if signal == "BEAR" else "NEUT")
        
        if predicted_dir == "NEUT":
            neutral.append((m, actual_ret, actual_dir))
        elif predicted_dir == "UP":
            if actual_dir == "UP":
                correct.append((m, actual_ret, actual_dir))
            else:
                wrong.append((m, actual_ret, actual_dir))
        elif predicted_dir == "DOWN":
            if actual_dir == "DOWN":
                correct.append((m, actual_ret, actual_dir))
            else:
                wrong.append((m, actual_ret, actual_dir))
    
    return correct, wrong, neutral


def build_last_week_review(prev_pred, actual_dict):
    """Build 上周总结 section."""
    prev_markets = prev_pred["markets"]
    period = prev_pred.get("prediction_period", "未知")
    
    correct, wrong, neutral = compare_prediction(prev_markets, actual_dict)
    
    t1_correct = [c for c in correct if c[0]["tier"] == 1]
    t1_wrong = [w for w in wrong if w[0]["tier"] == 1]
    t2_correct = [c for c in correct if c[0]["tier"] == 2]
    t2_wrong = [w for w in wrong if w[0]["tier"] == 2]
    
    t1_total = len(t1_correct) + len(t1_wrong) + len([x for x in neutral if x[0]["tier"] == 1])
    t1_hit = len(t1_correct)
    t2_total_dir = len(t2_correct) + len(t2_wrong)
    t2_hit = len(t2_correct)
    
    t1_rate = f"{t1_hit / (t1_hit + len(t1_wrong)) * 100:.1f}%" if (t1_hit + len(t1_wrong)) > 0 else "N/A"
    t2_rate = f"{t2_hit / t2_total_dir * 100:.1f}%" if t2_total_dir > 0 else "N/A"
    total_dir = len(correct) + len(wrong)
    total_rate = f"{len(correct) / total_dir * 100:.1f}%" if total_dir > 0 else "N/A"
    
    # Build correct list
    correct_parts = []
    for m, ret, dir_ in correct:
        tc = tier_badge(m["tier"])
        name = m["name_cn"]
        correct_parts.append(f'{tc}{name} {pct_str(ret)}')
    
    wrong_parts = []
    for m, ret, dir_ in wrong:
        tc = tier_badge(m["tier"])
        name = m["name_cn"]
        wrong_parts.append(f'{tc}{name} {signal_html(m["signal"])} → {pct_str(ret)}')
    
    neutral_parts = []
    for m, ret, dir_ in neutral:
        tc = tier_badge(m["tier"])
        name = m["name_cn"]
        neutral_parts.append(f'{tc}{name} {signal_html(m["signal"])} → {pct_str(ret) if ret is not None else "N/A"}')
    
    # Summary note
    if total_dir > 0:
        actual_rate = len(correct) / total_dir
    else:
        actual_rate = 0
    if total_dir > 0 and actual_rate >= 0.85:
        summary_color = "#4caf50"
        summary_text = "▸ 上周总结：预测表现优秀"
    elif total_dir > 0 and actual_rate >= 0.60:
        summary_color = "#ff9800"
        summary_text = "▸ 上周总结：预测表现一般"
    else:
        summary_color = "#f44336"
        summary_text = "▸ 上周总结：预测表现不及预期"
    
    # Build summary detail based on actual outcomes
    top_wrong_market = sorted(wrong, key=lambda x: abs(x[1]))[-1] if wrong else None
    wrong_detail = f"最大失误：{top_wrong_market[0]['name_cn']} BULL → {pct_str(top_wrong_market[1])}" if top_wrong_market else ""
    flat_markets = [x for x in wrong if x[2] == "FLAT"]
    flat_detail = f"，{', '.join([m['market'] for m, _, _ in flat_markets])} BULL但平盘" if flat_markets else ""
    summary_detail = (
        f"W27 预测方向正确率 {actual_rate:.1%}（{len(correct)}/{total_dir}）。"
        f"Tier1 {t1_hit}/{t1_hit+len(t1_wrong)} 正确，Tier2 {t2_hit}/{t2_total_dir} 正确。"
        f"{wrong_detail}{flat_detail}。高 EPU 环境下平盘市场增多，方向判断难度加大。"
    )
    
    def _actual_html(ret_val, dir_label):
        """Generate actual direction HTML cell."""
        if dir_label == "UP":
            return f'<span style="color:#4caf50">▲ UP {pct_str(ret_val)}</span>'
        elif dir_label == "DOWN":
            return f'<span style="color:#f44336">▼ DOWN {pct_str(ret_val)}</span>'
        elif dir_label == "FLAT":
            return f'<span style="color:#888">─ FLAT ({pct_str(ret_val)})</span>'
        else:
            return '<span style="color:#666">N/A</span>'
    
    correct_rows = "\n".join(
        f'<tr style="background:rgba(76,175,80,0.04)"><td>{tier_badge(m["tier"])} {m["name_cn"]}</td>'
        f'<td>{signal_html(m["signal"])}</td>'
        f'<td>{_actual_html(ret, dir_)}</td>'
        f'<td>{pct_str(ret)}</td>'
        f'<td><span style="color:#4caf50">✅ 正确</span></td></tr>'
        for m, ret, dir_ in correct
    )
    wrong_rows = "\n".join(
        f'<tr style="background:rgba(244,67,54,0.04)"><td>{tier_badge(m["tier"])} {m["name_cn"]}</td>'
        f'<td>{signal_html(m["signal"])}</td>'
        f'<td>{_actual_html(ret, dir_)}</td>'
        f'<td>{pct_str(ret)}</td>'
        f'<td><span style="color:#f44336">❌ 错误</span></td></tr>'
        for m, ret, dir_ in wrong
    )
    neutral_rows = "\n".join(
        f'<tr><td>{tier_badge(m["tier"])} {m["name_cn"]}</td>'
        f'<td>{signal_html(m["signal"])}</td>'
        f'<td>{_actual_html(ret, dir_)}</td>'
        f'<td>{pct_str(ret) if ret is not None else "N/A"}</td>'
        f'<td><span style="color:#666">─ 无方向</span></td></tr>'
        for m, ret, dir_ in neutral
    )
    
    review_table_rows = (correct_rows + "\n" + wrong_rows + "\n" + neutral_rows) if neutral else (correct_rows + "\n" + wrong_rows)
    
    return f"""
    <h2 class="section-title">📊 上周预测回顾 ({period})</h2>
    <div class="review-card">
        <p style="color:#888;font-size:13px;margin-bottom:16px">
            上周 ({period}) 预测，
            {', '.join(set(f'<span class="signal-{m["signal"].lower()}">{m["signal"]}</span>' for m in prev_markets))}，
            对比本周实际走势：
        </p>
        <div class="summary-stats">
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if t1_hit/(t1_hit+len(t1_wrong)) >= 0.7 else '#ff9800'}">{t1_rate}</div>
                <div class="l">Tier 1 方向正确率</div>
            </div>
            <div class="stat">
                <div class="n correct">{t1_hit}/{t1_hit+len(t1_wrong)}</div>
                <div class="l">Tier 1 正确/方向市场</div>
            </div>
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if t2_total_dir > 0 and t2_hit/t2_total_dir >= 0.7 else '#ff9800'}">{t2_rate}</div>
                <div class="l">Tier 2 方向正确率</div>
            </div>
            <div class="stat">
                <div class="n" style="color:#4caf50">{len(correct)}/{total_dir}</div>
                <div class="l">总体方向正确</div>
            </div>
        </div>

        <div class="accuracy-summary">
            <div class="acc-block"><h4 style="color:#4caf50">✓ 预测正确 ({len(correct)} 市场)</h4><div class="correct-list"><p>{' &nbsp; '.join(correct_parts)}</p></div></div>
            {'<div class="acc-block"><h4 style="color:#f44336">✗ 预测错误 (' + str(len(wrong)) + ' 市场)</h4><div class="wrong-list"><p>' + ' &nbsp; '.join(wrong_parts) + '</p></div></div>' if wrong else ''}
            {'<div class="acc-block"><h4 style="color:#666">─ 平盘/接近 (' + str(len(neutral)) + ' 市场)</h4><div class="neutral-list"><p>' + ' &nbsp; '.join(neutral_parts) + '</p></div></div>' if neutral else ''}
        </div>

        <div style="margin-top:16px;overflow-x:auto">
            <table class="predict-table" style="min-width:600px">
                <thead><tr>
                    <th>市场</th><th>预测</th><th>实际方向</th><th>实际涨跌%</th><th>结果</th>
                </tr></thead>
                <tbody>
    {review_table_rows}
                </tbody>
            </table>
        </div>

        <div style="margin-top:20px;padding:16px;background:rgba({','.join(['76,175,80' if summary_color == '#4caf50' else ('255,152,0' if summary_color == '#ff9800' else '244,67,54')])},0.05);border-radius:10px;border:1px solid rgba({','.join(['76,175,80' if summary_color == '#4caf50' else ('255,152,0' if summary_color == '#ff9800' else '244,67,54')])},0.1)">
            <p style="font-size:13px;color:#aaa;margin:0">
                <span style="color:{summary_color};font-weight:700">{summary_text}</span>
                {summary_detail}
            </p>
        </div>
    </div>"""


def build_monthly_review(monthly_pred, actual_dict):
    """Build 上月/月度总结 section."""
    prev_markets = monthly_pred["markets"]
    period = monthly_pred.get("prediction_period", "未知")
    
    correct, wrong, neutral = compare_prediction(prev_markets, actual_dict)
    
    t1_correct = [c for c in correct if c[0]["tier"] == 1]
    t1_wrong = [w for w in wrong if w[0]["tier"] == 1]
    t2_correct = [c for c in correct if c[0]["tier"] == 2]
    t2_wrong = [w for w in wrong if w[0]["tier"] == 2]
    
    t1_total_dir = len(t1_correct) + len(t1_wrong)
    t2_total_dir = len(t2_correct) + len(t2_wrong)
    
    t1_rate = f"{len(t1_correct) / t1_total_dir * 100:.1f}%" if t1_total_dir > 0 else "N/A"
    t2_rate = f"{len(t2_correct) / t2_total_dir * 100:.1f}%" if t2_total_dir > 0 else "N/A"
    total_dir = len(correct) + len(wrong)
    total_rate = f"{len(correct) / total_dir * 100:.1f}%" if total_dir > 0 else "N/A"
    
    correct_parts = []
    for m, ret, dir_ in correct:
        tc = tier_badge(m["tier"])
        name = m["name_cn"]
        correct_parts.append(f'{tc}{name} {pct_str(ret)}')
    
    wrong_parts = []
    for m, ret, dir_ in wrong:
        tc = tier_badge(m["tier"])
        name = m["name_cn"]
        wrong_parts.append(f'{tc}{name} {signal_html(m["signal"])} → {pct_str(ret)}')
    
    neutral_parts = []
    for m, ret, dir_ in neutral:
        tc = tier_badge(m["tier"])
        name = m["name_cn"]
        neutral_parts.append(f'{tc}{name} {signal_html(m["signal"])} → {pct_str(ret) if ret is not None else "N/A"}')
    
    # Build table rows
    all_rows = []
    for m, ret, _ in correct:
        all_rows.append((m, ret, "correct"))
    for m, ret, _ in wrong:
        all_rows.append((m, ret, "wrong"))
    for m, ret, _ in neutral:
        all_rows.append((m, ret, "neutral"))
    
    table_rows = []
    for m, ret, status in all_rows:
        bg = 'style="background:rgba(76,175,80,0.04)"' if status == "correct" else ('style="background:rgba(244,67,54,0.04)"' if status == "wrong" else "")
        if status == "correct":
            actual_html = f'<span style="color:#4caf50">▲ UP {pct_str(ret)}</span>' if ret > 0 else f'<span style="color:#f44336">▼ DOWN {pct_str(ret)}</span>'
            result_html = '<span style="color:#4caf50">✅ 正确</span>'
        elif status == "wrong":
            actual_html = f'<span style="color:#f44336">▼ DOWN {pct_str(ret)}</span>' if ret < 0 else f'<span style="color:#888">─ FLAT ({pct_str(ret)})</span>'
            result_html = '<span style="color:#f44336">❌ 错误</span>'
        else:
            dir_name = "UP" if ret and ret > 0.02 else ("DOWN" if ret and ret < -0.02 else "FLAT")
            dir_symbol = "▲" if ret and ret > 0 else ("▼" if ret and ret < 0 else "─")
            actual_html = f'<span style="color:#888">{dir_symbol} {dir_name} ({pct_str(ret)})</span>'
            result_html = '<span style="color:#888">─ 无方向/平盘</span>'
        
        table_rows.append(
            f'<tr {bg}><td>{tier_badge(m["tier"])} {m["name_cn"]}</td>'
            f'<td>{signal_html(m["signal"])}</td>'
            f'<td>{actual_html}</td>'
            f'<td>{pct_str(ret) if ret is not None else "N/A"}</td>'
            f'<td>{result_html}</td></tr>'
        )
    
    if total_dir > 0 and len(correct) / total_dir >= 0.80:
        summary_color = "#4caf50"
        summary_text = "▸ 月度总结：月度预测表现良好"
    elif total_dir > 0 and len(correct) / total_dir >= 0.60:
        summary_color = "#ff9800"
        summary_text = "▸ 月度总结：月度预测面临挑战"
    else:
        summary_color = "#f44336"
        summary_text = "▸ 月度总结：月度预测精准度有待提升"
    
    return f"""
    <h2 class="section-title">📅 上月预测回顾 (6月月度总结)</h2>
    <div class="review-card">
        <p style="color:#888;font-size:13px;margin-bottom:16px">
            上月 {period} 预测，Tier 1 全部 10 个市场为 <span class="signal-bull">▲ BULL</span>，
            CN 为 <span class="signal-neut">─ NEUT</span>，
            对比 6 月整月实际走势 (5/29→6/30)：
        </p>
        <div class="summary-stats">
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if len(t1_correct)/t1_total_dir >= 0.7 else '#ff9800'}">{t1_rate}</div>
                <div class="l">Tier 1 月度正确率</div>
            </div>
            <div class="stat">
                <div class="n correct">{len(t1_correct)}/{t1_total_dir}</div>
                <div class="l">Tier 1 正确市场</div>
            </div>
            <div class="stat">
                <div class="n" style="color:{'#f44336' if t2_total_dir > 0 and len(t2_correct)/t2_total_dir < 0.5 else '#4caf50'}">{t2_rate}</div>
                <div class="l">Tier 2 月度正确率</div>
            </div>
            <div class="stat">
                <div class="n" style="color:#4caf50">{len(correct)}/{total_dir}</div>
                <div class="l">总体方向正确</div>
            </div>
        </div>

        <div class="accuracy-summary">
            <div class="acc-block"><h4 style="color:#4caf50">✓ 预测正确 ({len(correct)} 市场)</h4><div class="correct-list"><p>{' &nbsp; '.join(correct_parts)}</p></div></div>
            {'<div class="acc-block"><h4 style="color:#f44336">✗ 预测错误 (' + str(len(wrong)) + ' 市场)</h4><div class="wrong-list"><p>' + ' &nbsp; '.join(wrong_parts) + '</p></div></div>' if wrong else ''}
            {'<div class="acc-block"><h4 style="color:#888">─ 无方向/平盘 (' + str(len(neutral)) + ')</h4><div class="neutral-list"><p>' + ' &nbsp; '.join(neutral_parts) + '</p></div></div>' if neutral else ''}
        </div>

        <div style="margin-top:16px;overflow-x:auto">
            <table class="predict-table" style="min-width:600px">
                <thead><tr>
                    <th>市场</th><th>预测</th><th>6月实际方向</th><th>6月涨跌%</th><th>结果</th>
                </tr></thead>
                <tbody>
    {chr(10).join(table_rows)}
                </tbody>
            </table>
        </div>

        <div style="margin-top:20px;padding:16px;background:rgba({'244,67,54' if summary_color == '#f44336' else '255,152,0'},0.05);border-radius:10px;border:1px solid rgba({'244,67,54' if summary_color == '#f44336' else '255,152,0'},0.1)">
            <p style="font-size:13px;color:#aaa;margin:0">
                <span style="color:{summary_color};font-weight:700">{summary_text}</span>
                高 EPU 持续环境下月度预测面临更大挑战——月度级别趋势延续模式的置信度低于周度。恒生指数月度跌幅 -9.14% 为最大预测失误，主要受港股特定政策风险（非 EPU 可捕获）影响。美国 SP500 (-1.06%) 和德国 DAX (-0.43%) 的月度下跌与 EPU 高位回调规律一致。
            </p>
        </div>
    </div>"""


def build_current_prediction(data):
    """Build 本周预测 section."""
    epu = data["epu"]
    markets = data["markets"]
    period = data.get("prediction_period", "")
    gen_time = data["generated_display"]
    
    tier1 = [m for m in markets if m["tier"] == 1]
    tier2 = [m for m in markets if m["tier"] == 2]

    # P2: Compute rolling 4-week actual accuracy
    rolling = compute_rolling_accuracy(num_weeks=4)

    def _ret_color(val):
        c = "4caf50" if val >= 0 else "f44336"
        return f'<span style="color:#{c}">{val * 100:+.2f}%</span>'
    
    def _tag(m):
        """Generate diagnosis tags."""
        tags = []
        tm = m.get("tm_z", 0)
        mr = m.get("mr_z", 0)
        vr = m.get("vr_z", 0)
        if m.get("regime_shift", False):
            tags.append('<span class="overheat-tag" style="background:rgba(244,67,54,0.15);color:#f44336">🔴 制度偏离→NEUT</span>')
        elif m.get("regime_unfamiliar", False):
            rd = m.get("regime_distance", 0)
            tags.append(f'<span class="overheat-tag" style="background:rgba(255,152,0,0.15);color:#ff9800">⚠ 陌生制度(D={rd:.1f})</span>')
        if m.get("overheat", False) or tm > 3.5:
            tags.append('<span class="overheat-tag">⚠过热</span>')
        elif tm > 2.0:
            tags.append('<span class="overheat-tag">⚠过热</span>')
        if mr < -3.0:
            tags.append('<span class="overheat-tag">超买</span>')
        elif mr > 2.5:
            tags.append('<span class="oversold-tag">超卖</span>')
        elif mr < -1.5:
            tags.append('<span class="overheat-tag">偏热</span>')
        elif mr > 1.0:
            tags.append('<span class="oversold-tag">偏卖</span>')
        # P0/P1: New override tags
        if m.get("flat_market", False):
            tags.append('<span class="overheat-tag" style="background:rgba(255,152,0,0.15);color:#ff9800">平盘→NEUT</span>')
        if m.get("vix_epu_conflict", False):
            tags.append('<span class="overheat-tag" style="background:rgba(156,39,176,0.15);color:#ce93d8">VIX-EPU冲突</span>')
        return ' '.join(tags)
    
    def market_row(m):
        """Build a full market table row with YTD, weekly change, rolling accuracy, and 5Y backtest."""
        code = m["market"]
        ytd_val = YTD.get(code)
        ytd_html = f'<span style="color:#4caf50">{pct_str(ytd_val)}</span>' if ytd_val is not None and ytd_val >= 0 else f'<span style="color:#f44336">{pct_str(ytd_val) if ytd_val is not None else "N/A"}</span>'
        weekly_change = W27_ACTUAL.get(code, {}).get("ret")
        weekly_html = f'<span style="color:#4caf50">{pct_str(weekly_change)}</span>' if weekly_change is not None and weekly_change >= 0 else f'<span style="color:#f44336">{pct_str(weekly_change) if weekly_change is not None else "N/A"}</span>'

        # P2: Rolling per-market accuracy
        per_mkt_roll = rolling.get("per_market", {}).get(code) if rolling else None
        if per_mkt_roll is not None:
            roll_color = "#4caf50" if per_mkt_roll >= 0.70 else ("#ff9800" if per_mkt_roll >= 0.55 else "#f44336")
            roll_html = f'<span style="color:{roll_color};font-weight:600">{per_mkt_roll:.0%}</span>'
        else:
            roll_html = '<span style="color:#555">-</span>'

        # P3: 5-year rolling backtest
        roll5y = m.get("rolling_accuracy")
        full_acc = m.get("accuracy", 0)
        if roll5y is not None:
            degraded = roll5y < full_acc * 0.7
            r5_color = "#ff9800" if degraded else "#888"
            r5_weight = "font-weight:700" if degraded else ""
            r5_html = f'<span style="color:{r5_color};{r5_weight}">{roll5y:.1%}</span>'
        else:
            r5_html = '<span style="color:#555">N/A</span>'

        # P4: Sentiment factor column
        sent = m.get("sentiment_factor", "N/A")
        if sent == "BULLISH":
            sent_html = '<span style="color:#4caf50;font-size:10px">↗ 看好</span>'
        elif sent == "BEARISH":
            sent_html = '<span style="color:#f44336;font-size:10px">↘ 看空</span>'
        elif sent == "NEUTRAL":
            sent_html = '<span style="color:#888;font-size:10px">─ 中性</span>'
        else:
            sent_html = '<span style="color:#555;font-size:10px">N/A</span>'

        # Signal rendering with regime_shift and low_confidence markers
        sig = m["signal"]
        low_conf = m.get("low_confidence", False)
        regime_shift = m.get("regime_shift", False)
        if regime_shift:
            sig_html = '<span style="color:#f44336;font-weight:700">─ NEUT <span style="font-size:9px;color:#f44336">⚠制度偏离</span></span>'
        elif sig == "NEUT" and low_conf:
            sig_html = '<span style="color:#888;font-weight:400;font-style:italic">─ NEUT <span style="font-size:9px;color:#666">低信念</span></span>'
        else:
            sig_html = signal_html(sig)

        detail = m.get("detail", "")
        tags = _tag(m)
        diagnosis = f'{tags} {detail}'.strip()

        row_class = ' cn-row' if code == 'CN' else ''

        cn_note = '<span class="cn-note">⚠ 仅供参考</span>' if code == 'CN' else ''

        # Regime distance cell (P3)
        rd = m.get("regime_distance")
        if rd is not None:
            if m.get("regime_shift", False):
                rd_html = f'<span style="color:#f44336;font-weight:700">{rd:.1f} 🔴</span>'
            elif m.get("regime_unfamiliar", False):
                rd_html = f'<span style="color:#ff9800;font-weight:700">{rd:.1f} ⚠</span>'
            else:
                rd_html = f'<span style="color:#666">{rd:.1f}</span>'
        else:
            rd_html = '<span style="color:#555">-</span>'

        return (
            f'<tr class="{row_class}">'
            f'<td style="font-size:13px">{tier_badge(m["tier"])}{m["name_cn"]}{cn_note}</td>'
            f'<td>{sig_html}</td>'
            f'<td style="text-align:right">{_ret_color(m["pred_ret"])}</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m["accuracy"]:.1%}</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{r5_html}</td>'
            f'<td style="text-align:right;font-size:12px">{roll_html}</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m["tm_z"]:+.2f}σ</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m["mr_z"]:+.2f}σ</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m.get("vr_z",0):+.2f}σ</td>'
            f'<td style="text-align:right;font-size:12px">{rd_html}</td>'
            f'<td style="text-align:right">{ytd_html}</td>'
            f'<td style="text-align:right">{weekly_html}</td>'
            f'<td style="text-align:center">{sent_html}</td>'
            f'<td style="font-size:11px;color:#666">{diagnosis}</td>'
            f'</tr>'
        )
    
    tier1_rows = "\n".join(market_row(m) for m in tier1)
    tier2_rows = "\n".join(market_row(m) for m in tier2)
    
    # HK special section
    hk_markets = [m for m in markets if m["market"] == "HK"]
    hk_html = ""
    if hk_markets:
        hk_html = f"""
    <div class="hk-section">
        <h3>🇭🇰 港股预测</h3>
        <p style="font-size:13px;color:#888;margin-bottom:12px">
            恒生指数已升级为 Tier 1 级别（准确率 {hk_markets[0]['accuracy']:.1%}），China EPU 制度切换显著提升预测精度（从 66.7% → {hk_markets[0]['accuracy']:.1%}）。
            BULL 信号较为可靠，但需关注中美政策边际变化及南向资金流向。
        </p>
    </div>"""
    
    # YTD bar chart
    ytd_sorted = sorted(YTD.items(), key=lambda x: abs(x[1]), reverse=True)
    max_abs = max(abs(v) for _, v in ytd_sorted)
    
    ytd_bars = []
    for code, ytd_val in ytd_sorted:
        if ytd_val is None:
            continue
        name = MARKET_NAMES.get(code, code)
        width_pct = min(abs(ytd_val) / max_abs * 100, 100)
        is_pos = ytd_val >= 0
        bar_class = "positive" if is_pos else "negative"
        color = "#4caf50" if is_pos else "#f44336"
        ytd_bars.append(
            f'<div class="ytd-row">'
            f'<div class="ytd-label">{name}</div>'
            f'<div class="ytd-bar-wrap"><div class="ytd-bar {bar_class}" style="width:{width_pct:.1f}%">{pct_str(ytd_val)}</div></div>'
            f'<div class="ytd-val" style="color:{color}">{pct_str(ytd_val)}</div>'
            f'</div>'
        )
    
    ytd_bars_html = "\n".join(ytd_bars)
    
    # Overheat warnings
    overheated = []
    for m in markets:
        tm = m.get("tm_z", 0)
        mr = m.get("mr_z", 0)
        if tm > 1.0 or mr < -2.0:
            level = '<span style="color:#f44336">极度超买</span>' if tm > 2.5 or mr < -3.0 else '<span style="color:#ff9800">趋于超买</span>' if tm > 1.0 else ''
            desc = []
            if tm > 1.0:
                desc.append(f'趋势 +{tm:.2f}σ')
            if mr < -1.0:
                desc.append(f'反转 {mr:.2f}σ')
            if desc:
                overheated.append(f'<li><span style="color:{"#f44336" if tm > 2.5 else "#ff9800"}">{m["name_cn"]}</span>：{"|".join(desc)} — <strong>{level}</strong></li>')
    
    overheat_list = "\n".join(overheated) if overheated else "<li>当前无显著过热信号</li>"
    
    # Signal summary helpers
    def _count_signal(mkts, sig):
        return len([m for m in mkts if m["signal"] == sig])
    
    return f"""
    <h2 class="section-title">🔮 本周预测 ({period})</h2>

    <div class="metric-grid">
        <div class="metric">
            <div class="lbl">EPU 策略不确定性</div>
            <div class="val" style="color:#f44336">{epu['value']}<span style="font-size:11px;color:#888"> P{epu['percentile']}</span></div>
            <div style="font-size:10px;color:#888;margin-top:4px">⚠ 极高不确定性 → 趋势延续模式</div>
        </div>
        <div class="metric">
            <div class="lbl">中国 EPU</div>
            <div class="val" style="font-size:20px;color:#ff9800">{epu.get('china_epu', 'N/A')}</div>
            <div style="font-size:10px;color:#888;margin-top:4px">中EPU高 → 趋势延续</div>
        </div>
        <div class="metric">
            <div class="lbl">Tier 1 高置信度</div>
            <div class="val" style="color:#4caf50">{len(tier1)} 市场</div>
            <div style="font-size:10px;color:#888;margin-top:4px">准确率 ≥70% · 可操作</div>
        </div>
        <div class="metric">
            <div class="lbl">Tier 2 参考</div>
            <div class="val" style="color:#ff9800">{len(tier2)} 市场</div>
            <div style="font-size:10px;color:#888;margin-top:4px">55-70% · 仅供参考</div>
        </div>
        {_rolling_accuracy_html(rolling)}
        <div class="metric" style="border:1px solid rgba(244,67,54,0.3)">
            <div class="lbl">制度偏离市场数</div>
            <div class="val" style="font-size:18px;color:#f44336">{data.get('regime_shifted_count', 0)}🔴 {data.get('regime_warned_count', 0)}⚠</div>
            <div style="font-size:10px;color:#888;margin-top:4px">强制NEUT · 陌生制度</div>
        </div>
        <div class="metric">
            <div class="lbl">数据快照</div>
            <div class="val" style="font-size:14px;color:#4facfe;font-family:monospace">{gen_time}</div>
            <div style="font-size:10px;color:#888;margin-top:4px">3-Factor Kalman DFM</div>
        </div>
    </div>

    {hk_html}

    <h3 style="font-size:15px;color:#4caf50;margin-bottom:16px;">★★★ Tier 1 — 高置信度 (≥70% 准确率)</h3>
    <p style="font-size:12px;color:#666;margin-top:-8px;margin-bottom:8px">置信度=回测 | 5年=近5年滚动 | 滚动=近4周实际 | 情绪=独立因子 | 制度σ=马氏距离 | YTD=年初至今 | 周变=上周变化</p>
	    <div style="overflow-x:auto">
	    <table class="predict-table" style="min-width:1400px">
	        <thead><tr>
	            <th>市场</th><th>信号</th><th style="text-align:right">12月预期收益</th>
	            <th style="text-align:right">回测</th><th style="text-align:right">5年</th><th style="text-align:right">滚动</th><th style="text-align:right">趋势σ</th><th style="text-align:right">反转σ</th><th style="text-align:right">波动σ</th><th style="text-align:right">制度σ</th><th style="text-align:right">YTD</th><th style="text-align:right">周变</th><th style="text-align:center">情绪</th><th>诊断</th>
        </tr></thead>
        <tbody>
    {tier1_rows}
        </tbody>
    </table>
    </div>

    <h3 style="font-size:15px;color:#ff9800;margin-top:32px;margin-bottom:16px;">★★ Tier 2 — 参考级别 (55-70% 准确率)</h3>
    <div style="overflow-x:auto">
	    <table class="predict-table" style="min-width:1400px">
	        <thead><tr>
	            <th>市场</th><th>信号</th><th style="text-align:right">12月预期收益</th>
	            <th style="text-align:right">回测</th><th style="text-align:right">5年</th><th style="text-align:right">滚动</th><th style="text-align:right">趋势σ</th><th style="text-align:right">反转σ</th><th style="text-align:right">波动σ</th><th style="text-align:right">制度σ</th><th style="text-align:right">YTD</th><th style="text-align:right">周变</th><th style="text-align:center">情绪</th><th>诊断</th>
        </tr></thead>
        <tbody>
    {tier2_rows}
        </tbody>
    </table>
    </div>
    <p style="font-size:11px;color:#555;margin-top:4px;margin-bottom:24px">⚠ Tier 2 市场准确率较低（55-70%），追踪历史 ≤ 1 年，仅供观察参考。A股数据来源于 akshare，受政策与流动性影响较大。恒生指数受中美关系及全球流动性影响显著。</p>

    <h3 style="font-size:15px;color:#e0e0e0;margin-top:40px;margin-bottom:8px;">📊 YTD 表现对比 (截至 2026-07-03)</h3>
    <p style="font-size:12px;color:#666;margin-bottom:16px">各市场年初至今实际涨跌幅 · 绿色=正收益 · 红色=负收益 · 柱宽按最大绝对值等比例缩放</p>
    <div class="ytd-chart" style="max-width:700px">
    {ytd_bars_html}
    </div>

    <h3 style="font-size:15px;color:#f7971e;margin-top:40px;margin-bottom:16px;">⚡ 本周关键判断</h3>
    <div class="card">
        <div class="grid-2">
            <div>
                <h4 style="font-size:14px;color:#4caf50;margin-bottom:8px">🌍 全球共识：分化开始</h4>
                <p style="font-size:13px;color:#aaa;line-height:1.7">
                    EPU 当前值 <strong style="color:#f44336">{epu['value']} (P{epu['percentile']})</strong>，处于历史极端高位。
                    模型在所有市场统一触发 <strong>趋势延续模式</strong>，但过热保护机制在部分市场激活。
                    Tier 1 中 {_count_signal(tier1, "BULL")} BULL + {_count_signal(tier1, "BEAR")} BEAR + {_count_signal(tier1, "NEUT")} NEUT；
                    Tier 2 中 {_count_signal(tier2, "BULL")} BULL + {_count_signal(tier2, "NEUT")} NEUT。
                    韩国 KOSPI 因极端过热（tm=+3.77σ）触发 BEAR 信号；
                    日经 225 因过热（tm=+2.53σ）转为 NEUT；
                    中国 A股降至 NEUT。
                    全球化与分化的关键转折——极度超买市场的风险信号正在显现。
                    中国EPU 为 {epu.get('china_epu', 'N/A')}，对恒生指数和中国A股的影响显著。
                </p>
            </div>
            <div>
                <h4 style="font-size:14px;color:#f44336;margin-bottom:8px">⚠ 过热警告</h4>
                <p style="font-size:13px;color:#aaa;line-height:1.7">
                    以下市场出现极端技术与估值信号：
                </p>
                <ul style="font-size:12px;color:#aaa;line-height:1.8;margin-top:8px;padding-left:18px">
                    {overheat_list}
                </ul>
            </div>
        </div>
        <div style="margin-top:20px">
            <h4 style="font-size:14px;color:#4facfe;margin-bottom:8px">📈 EPU 趋势与模式预期</h4>
            <p style="font-size:13px;color:#aaa;line-height:1.7">
                EPU 自 5 月以来维持在 350 附近（P90+），无回落迹象，当前百分位为 <strong>P{epu['percentile']}</strong>。
                若 EPU 维持当前水平，趋势延续模式将继续主导预测，全球市场 BULL 信号将延续。
                <strong>关键观察点：</strong>若 EPU 回落至 P75 以下（约 &lt;200），模型将切换回均值回归模式，
                届时超买市场（韩国、日本）可能出现显著反转信号。
                当前建议：关注 Tier 1 高置信度市场，对超买市场谨慎追高。
            </p>
        </div>
    </div>"""


def build_outlook(epu, period_str):
    """Build 下周展望 section."""
    return f"""
    <h2 class="section-title">🔭 下周展望 (第28周 7/6-7/12)</h2>
    <div class="outlook-card">
        <h3>📡 基于 EPU 极端高位的 W28 市场前瞻</h3>
        <p style="font-size:13px;color:#aaa;line-height:1.7;margin-bottom:16px">
            当前 EPU = <strong style="color:#f44336">{epu['value']} (P{epu['percentile']})</strong>，处于历史极端高位区间。
            根据模型的历史行为分析，当 EPU 超过 90 百分位时：
        </p>
        
        <div class="grid-2" style="margin-bottom:16px">
            <div>
                <h4 style="font-size:14px;color:#4caf50;margin-bottom:8px">📈 历史上 EPU > P90 时的市场行为</h4>
                <ul style="font-size:12px;color:#aaa;line-height:1.8;padding-left:18px">
                    <li><strong>趋势延续</strong>：市场倾向于维持原有趋势方向，反转概率降低</li>
                    <li><strong>波动加剧</strong>：VIX 类指标上升，单日大幅波动概率增加</li>
                    <li><strong>板块分化</strong>：防御性板块与周期性板块表现差距拉大</li>
                    <li><strong>新兴市场承压</strong>：资金倾向于回流发达市场</li>
                </ul>
            </div>
            <div>
                <h4 style="font-size:14px;color:#f44336;margin-bottom:8px">⚠ 高EPU环境下的预测局限性</h4>
                <ul style="font-size:12px;color:#aaa;line-height:1.8;padding-left:18px">
                    <li>模型在极端 EPU 区域样本量较少，预测不确定性增大</li>
                    <li>政策突发变量（关税、制裁、地缘冲突）无法被模型捕获</li>
                    <li>市场可能对 EPU 产生"适应性"，历史规律不完全适用</li>
                    <li>超买信号在高 EPU 下被压制，回调风险累积</li>
                </ul>
            </div>
        </div>

        <h4 style="font-size:14px;color:#4facfe;margin-bottom:8px">🎯 W28 关注点</h4>
        <div class="grid-2">
            <div>
                <ul style="font-size:12px;color:#aaa;line-height:1.8;padding-left:18px">
                    <li><span style="color:#4facfe">美联储动向</span>：关注 7 月 FOMC 会议纪要与官员讲话</li>
                    <li><span style="color:#4facfe">全球贸易政策</span>：关税谈判进展、贸易协议动态</li>
                    <li><span style="color:#4facfe">地缘政治</span>：中东、东欧局势对能源与供应链的影响</li>
                </ul>
            </div>
            <div>
                <ul style="font-size:12px;color:#aaa;line-height:1.8;padding-left:18px">
                    <li><span style="color:#4facfe">通胀数据</span>：各国 CPI/PPI 发布对政策预期的影响</li>
                    <li><span style="color:#4facfe">技术面</span>：韩国 KOSPI、日经 225 超买信号是否触发回调</li>
                    <li><span style="color:#4facfe">流动性</span>：季末资金面变化对新兴市场的影响</li>
                </ul>
            </div>
        </div>

        <div class="warning-box">
            <p><strong>⚠ 风险提示：</strong>
            高 EPU（P{epu['percentile']}）环境下，模型的趋势延续模式虽然历史表现稳健，但极端政治经济环境可能引发非线性市场反应。
            不建议在此阶段过度集中于单一市场或方向，尤其是已出现极端超买信号的韩国（趋势 +3.77σ）和日本（趋势 +2.53σ）市场。
            下周重点关注 EPU 是否出现拐点信号，若 EPU 回落至 250 以下，部分市场的均值回归压力将显著释放。
            韩国 KOSPI W27 周跌 -7.80%，已开始超买回调，需持续关注。
            </p>
        </div>
    </div>"""


def main():
    # Load current prediction first
    cur_pred = load_json(OUTPUT_DIR / "weekly_prediction.json")
    
    # Extract week number from prediction_period (format: "2026年第28周 (7/6-7/12)")
    import re
    period = cur_pred.get("prediction_period", "")
    week_match = re.search(r'第(\d+)周', period)
    if week_match:
        week_num = int(week_match.group(1))
    else:
        today = date.today()
        iso = today.isocalendar()
        year, week_num = iso[0], iso[1] + 1  # +1 for next week
    year = 2026
    
    # Get today/current date for archive naming
    today = date.today()
    gen_time = cur_pred["generated_display"]
    period = cur_pred.get("prediction_period", f"{year}年第{week_num}周")
    epu = cur_pred["epu"]
    
    # Load last week prediction (6/30 W27 prediction)
    last_week_pred = load_json(OUTPUT_DIR / "archive" / "2026-06-30" / "weekly_prediction.json")
    
    # Load monthly prediction (6/2 W23 prediction)
    monthly_pred = load_json(OUTPUT_DIR / "archive" / "2026-06-02" / "weekly_prediction.json")
    
    # Build sections
    last_week_section = build_last_week_review(last_week_pred, W27_ACTUAL)
    monthly_section = build_monthly_review(monthly_pred, JUNE_ACTUAL)
    current_section = build_current_prediction(cur_pred)
    outlook_section = build_outlook(epu, period)
    
    # Model explanation
    # Compute accuracy stats for footer
    def get_rates(pred, actual):
        corr, wr, neu = compare_prediction(pred["markets"], actual)
        t1_c = [c for c in corr if c[0]["tier"] == 1]
        t1_w = [w for w in wr if w[0]["tier"] == 1]
        t2_c = [c for c in corr if c[0]["tier"] == 2]
        t2_w = [w for w in wr if w[0]["tier"] == 2]
        t1_total = len(t1_c) + len(t1_w)
        t2_total = len(t2_c) + len(t2_w)
        t1_r = f"{len(t1_c)/t1_total*100:.1f}%" if t1_total > 0 else "N/A"
        t2_r = f"{len(t2_c)/t2_total*100:.1f}%" if t2_total > 0 else "N/A"
        return t1_r, t2_r
    
    wk_t1, wk_t2 = get_rates(last_week_pred, W27_ACTUAL)
    mo_t1, mo_t2 = get_rates(monthly_pred, JUNE_ACTUAL)
    
    body = f"""<div class="container">
    <a href="/" class="back-link">← 首页</a>
    <div class="page-header">
        <div class="icon">🔮</div>
        <h1 style="background:linear-gradient(135deg,#f7971e,#ffd200);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">全球市场预测</h1>
        <p class="sub">13 市场 · 3 因子模型 · EPU 制度切换 ｜ 预测周期 {period} ｜ <span class="ts-stamp">生成 {gen_time}</span></p>
    </div>

    {last_week_section}
    {monthly_section}
    {current_section}
    {outlook_section}

    <div class="insight-box" style="margin-top:32px">
        <p>💡 <strong>模型说明：</strong>3-factor Kalman DFM（趋势+均值回归+波动率）提取方向和幅度两个潜因子，EPU > P75 时自动切换为趋势延续模式。
        恒生指数和中国A股使用中国 EPU（CHNMAINLANDEPU）作为制度开关，HSI 准确率从 66.7% 提升至 {cur_pred['markets'][[m['market'] for m in cur_pred['markets']].index('HK')]['accuracy']:.1%}。
        Tier 1 市场（≥70% 回测准确率，≥5年历史数据）可用于配置参考；Tier 2 市场（55-70%，≤1年追踪）仅供观察。
        预测收益率为年化估算，不代表短期走势。模型基于过去 5 年滚动窗口回测验证。
        数据来源：yfinance / FRED / akshare · 模型：Global Economy Lab · 
        <strong>预测周期：{period} · 生成时间：{gen_time}</strong></p>
    </div>

    <div style="text-align:center;margin-top:24px;font-size:11px;color:#444">
        <span class="ts-stamp">第{week_num}周存档</span>
        <span style="margin:0 8px">|</span>
        <span>上周准确率 Tier1: {wk_t1} | Tier2: {wk_t2}</span>
        <span style="margin:0 8px">|</span>
        <span>月度准确率 Tier1: {mo_t1} | Tier2: {mo_t2}</span>
        <span style="margin:0 8px">|</span>
        <span>EPU: {epu['value']} (P{epu['percentile']} · HIGH_EPU 制度)</span>
    </div>
</div>"""
    
    html = build_page(f"全球市场预测 (第{week_num}周)", body)
    
    # Write outputs
    html_path = OUTPUT_DIR / "prediction.html"
    week_path = OUTPUT_DIR / f"prediction{year}{week_num:02d}.html"
    archive_dir = OUTPUT_DIR / "archive" / today.strftime("%Y-%m-%d")
    
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  → {html_path} ({len(html):,} bytes)")
    
    with open(week_path, "w") as f:
        f.write(html)
    print(f"  → {week_path} (weekly archive)")
    
    # Also save to today's archive
    if archive_dir.exists():
        with open(archive_dir / "prediction.html", "w") as f:
            f.write(html)
        with open(archive_dir / week_path.name, "w") as f:
            f.write(html)
        print(f"  → {archive_dir}/ (archive)")
    
    print(f"\nDone. Prediction HTML generated for week {week_num}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
