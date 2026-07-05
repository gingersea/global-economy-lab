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

W27_ACTUAL = {
    "US": {"ret": 1.76, "start_date": "6/26", "end_date": "7/2"},
    "DE": {"ret": 4.49, "start_date": "6/26", "end_date": "7/3"},
    "JP": {"ret": 0.55, "start_date": "6/26", "end_date": "7/3"},
    "GB": {"ret": 1.63, "start_date": "6/26", "end_date": "7/3"},
    "FR": {"ret": 1.47, "start_date": "6/26", "end_date": "7/3"},
    "IT": {"ret": 3.03, "start_date": "6/26", "end_date": "7/3"},
    "CA": {"ret": 0.76, "start_date": "6/26", "end_date": "7/3"},
    "BR": {"ret": 0.56, "start_date": "6/26", "end_date": "7/3"},
    "KR": {"ret": -3.84, "start_date": "6/26", "end_date": "7/3"},
    "IN": {"ret": 0.89, "start_date": "6/25", "end_date": "7/3"},
    "AU": {"ret": 0.92, "start_date": "6/26", "end_date": "7/3"},
    "CN": {"ret": 0.41, "start_date": "6/26", "end_date": "7/3"},
    "HK": {"ret": 2.99, "start_date": "6/26", "end_date": "7/3"},
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


def compare_prediction(pred_markets, actual_dict, pred_source_label="预测"):
    """Compare predicted signals vs actual returns. Returns (correct, wrong, neutral) lists."""
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
        
        actual_dir = "UP" if actual_ret > 0.02 else ("DOWN" if actual_ret < -0.02 else "FLAT")
        predicted_dir = "UP" if signal == "BULL" else ("DOWN" if signal == "BEAR" else "NEUT")
        
        if predicted_dir == "NEUT":
            neutral.append((m, actual_ret, actual_dir))
        elif predicted_dir == "UP" and actual_ret > -0.03:
            if actual_ret > 0.02:
                correct.append((m, actual_ret, actual_dir))
            else:
                neutral.append((m, actual_ret, actual_dir))
        elif predicted_dir == "DOWN" and actual_ret < 0.03:
            if actual_ret < -0.02:
                correct.append((m, actual_ret, actual_dir))
            else:
                neutral.append((m, actual_ret, actual_dir))
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
    if total_dir > 0 and len(correct) / total_dir >= 0.85:
        summary_color = "#4caf50"
        summary_text = "▸ 上周总结：预测表现优秀"
    elif total_dir > 0 and len(correct) / total_dir >= 0.70:
        summary_color = "#ff9800"
        summary_text = "▸ 上周总结：预测表现良好"
    else:
        summary_color = "#f44336"
        summary_text = "▸ 上周总结：预测准确率需要关注"
    
    correct_rows = "\n".join(
        f'<tr><td>{tier_badge(m["tier"])} {m["name_cn"]}</td>'
        f'<td>{signal_html(m["signal"])}</td>'
        f'<td><span style="color:#4caf50">▲ UP {pct_str(ret)}</span></td>'
        f'<td>{pct_str(ret)}</td>'
        f'<td><span style="color:#4caf50">✅ 正确</span></td></tr>'
        for m, ret, _ in correct
    )
    wrong_rows = "\n".join(
        f'<tr style="background:rgba(244,67,54,0.04)"><td>{tier_badge(m["tier"])} {m["name_cn"]}</td>'
        f'<td>{signal_html(m["signal"])}</td>'
        f'<td><span style="color:#f44336">▼ DOWN {pct_str(ret)}</span></td>'
        f'<td>{pct_str(ret)}</td>'
        f'<td><span style="color:#f44336">❌ 错误</span></td></tr>'
        for m, ret, _ in wrong
    )
    neutral_rows = "\n".join(
        f'<tr><td>{tier_badge(m["tier"])} {m["name_cn"]}</td>'
        f'<td>{signal_html(m["signal"])}</td>'
        f'<td><span style="color:#666">'
        f'{"▲ UP" if ret is not None and ret > 0 else ("▼ DOWN" if ret is not None else "N/A")} {pct_str(ret)}</span></td>'
        f'<td>{pct_str(ret) if ret is not None else "N/A"}</td>'
        f'<td><span style="color:#666">─ 平盘/接近</span></td></tr>'
        for m, ret, _ in neutral
    )
    
    review_table_rows = (correct_rows + "\n" + wrong_rows + "\n" + neutral_rows) if neutral else (correct_rows + "\n" + wrong_rows)
    
    return f"""
    <h2 class="section-title">📊 上周预测回顾 ({period})</h2>
    <div class="review-card">
        <p style="color:#888;font-size:13px;margin-bottom:16px">
            上周 ({period}) 预测全部市场为 <span class="signal-bull">▲ BULL</span>，
            对比本周 ({period}) 实际走势 (6/26→7/3)：
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
                EPU 高企（P93.5）环境下趋势延续模式表现稳定，模型在多数市场保持方向判断准确。韩国 KOSPI 为唯一 Tier 1 方向错误市场（周内震荡-3.84%，极热后的回调）。
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
    
    def _ret_color(val):
        c = "4caf50" if val >= 0 else "f44336"
        return f'<span style="color:#{c}">{val * 100:+.2f}%</span>'
    
    def _tag(m):
        """Generate diagnosis tags."""
        tags = []
        tm = m.get("tm_z", 0)
        mr = m.get("mr_z", 0)
        vr = m.get("vr_z", 0)
        if tm > 2.0:
            tags.append('<span class="overheat-tag">⚠过热</span>')
        elif tm < -1.5:
            tags.append('<span class="oversold-tag">弱趋势</span>')
        if mr < -2.5:
            tags.append('<span class="overheat-tag">超买</span>')
        elif mr > 2.5:
            tags.append('<span class="oversold-tag">超卖</span>')
        return ' '.join(tags)
    
    def market_row(m):
        """Build a full market table row with YTD and weekly change."""
        code = m["market"]
        ytd_val = YTD.get(code)
        ytd_html = f'<span style="color:#4caf50">{pct_str(ytd_val)}</span>' if ytd_val is not None and ytd_val >= 0 else f'<span style="color:#f44336">{pct_str(ytd_val) if ytd_val is not None else "N/A"}</span>'
        weekly_change = W27_ACTUAL.get(code, {}).get("ret")
        weekly_html = f'<span style="color:#4caf50">{pct_str(weekly_change)}</span>' if weekly_change is not None and weekly_change >= 0 else f'<span style="color:#f44336">{pct_str(weekly_change) if weekly_change is not None else "N/A"}</span>'
        
        detail = m.get("detail", "")
        tags = _tag(m)
        diagnosis = f'{tags} {detail}'.strip()
        
        row_class = ' cn-row' if code == 'CN' else ''
        
        cn_note = '<span class="cn-note">⚠ 仅供参考</span>' if code == 'CN' else ''
        
        return (
            f'<tr class="{row_class}">'
            f'<td style="font-size:13px">{tier_badge(m["tier"])}{m["name_cn"]}{cn_note}</td>'
            f'<td>{signal_html(m["signal"])}</td>'
            f'<td style="text-align:right">{_ret_color(m["pred_ret"])}</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m["accuracy"]:.1%}</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m["tm_z"]:+.2f}σ</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m["mr_z"]:+.2f}σ</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m.get("vr_z",0):+.2f}σ</td>'
            f'<td style="text-align:right">{ytd_html}</td>'
            f'<td style="text-align:right">{weekly_html}</td>'
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
        <div class="metric">
            <div class="lbl">数据快照</div>
            <div class="val" style="font-size:14px;color:#4facfe;font-family:monospace">{gen_time}</div>
            <div style="font-size:10px;color:#888;margin-top:4px">3-Factor Kalman DFM</div>
        </div>
    </div>

    {hk_html}

    <h3 style="font-size:15px;color:#4caf50;margin-bottom:16px;">★★★ Tier 1 — 高置信度 (≥70% 准确率)</h3>
    <p style="font-size:12px;color:#666;margin-top:-8px;margin-bottom:8px">预测未来12个月收益方向 · YTD = 年初至今实际涨幅 · 周变 = 本周({period.split(' ')[1] if ' ' in period else '当前周'})变化 · 数据至 = 最近交易日</p>
    <div style="overflow-x:auto">
    <table class="predict-table" style="min-width:1000px">
        <thead><tr>
            <th>市场</th><th>信号</th><th style="text-align:right">12月预期收益</th>
            <th style="text-align:right">置信度</th><th style="text-align:right">趋势σ</th><th style="text-align:right">反转σ</th><th style="text-align:right">波动σ</th><th style="text-align:right">YTD</th><th style="text-align:right">周变</th><th>诊断 · 数据至</th>
        </tr></thead>
        <tbody>
    {tier1_rows}
        </tbody>
    </table>
    </div>

    <h3 style="font-size:15px;color:#ff9800;margin-top:32px;margin-bottom:16px;">★★ Tier 2 — 参考级别 (55-70% 准确率)</h3>
    <div style="overflow-x:auto">
    <table class="predict-table" style="min-width:1000px">
        <thead><tr>
            <th>市场</th><th>信号</th><th style="text-align:right">12月预期收益</th>
            <th style="text-align:right">置信度</th><th style="text-align:right">趋势σ</th><th style="text-align:right">反转σ</th><th style="text-align:right">波动σ</th><th style="text-align:right">YTD</th><th style="text-align:right">周变</th><th>诊断 · 数据至</th>
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
                <h4 style="font-size:14px;color:#4caf50;margin-bottom:8px">🌍 全球共识：全面看多</h4>
                <p style="font-size:13px;color:#aaa;line-height:1.7">
                    EPU 当前值 <strong style="color:#f44336">{epu['value']} (P{epu['percentile']})</strong>，处于历史极端高位。
                    模型在所有市场统一触发 <strong>趋势延续模式</strong>。
                    Tier 1 全部 {len(tier1)} 个市场均发出 BULL 信号，Tier 2 中 BR 为 BULL、CN 为 NEUT。
                    EPU 阈值效应下的典型表现：高不确定性环境掩盖了微弱反转信号，趋势因子权重提升。
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
            韩国 KOSPI W27 周跌 -3.84%，可能已开始超买回调，需持续关注。
            </p>
        </div>
    </div>"""


def main():
    # Set week info
    today = date.today()
    iso = today.isocalendar()
    year, week_num = iso[0], iso[1]
    mon = today - timedelta(days=today.weekday())
    sun = mon + timedelta(days=6)
    
    # Load current prediction
    cur_pred = load_json(OUTPUT_DIR / "weekly_prediction.json")
    gen_time = cur_pred["generated_display"]
    period = cur_pred.get("prediction_period", f"{year}年第{week_num}周 ({mon.month}/{mon.day}-{sun.month}/{sun.day})")
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
