#!/usr/bin/env python3
"""Generate weekly prediction HTML with last-week review + this-week prediction + next-week outlook."""

import json
import yfinance as yf
from datetime import datetime, timedelta

# ════════════════════════════════════════════
# Ticker mapping
# ════════════════════════════════════════════
TICKER_MAP = {
    "US": "^GSPC", "DE": "^GDAXI", "JP": "^N225",
    "GB": "^FTSE", "FR": "^FCHI", "IT": "FTSEMIB.MI",
    "CA": "^GSPTSE", "BR": "^BVSP", "KR": "^KS11",
    "IN": "^NSEI", "AU": "^AXJO", "CN": "000001.SS",
    "HK": "^HSI"
}

# ════════════════════════════════════════════
# Load data
# ════════════════════════════════════════════
with open("output/weekly_prediction.json") as f:
    w25_data = json.load(f)

with open("output/archive/2026-06-14/weekly_prediction.json") as f:
    w24_data = json.load(f)

w24_markets = {m["market"]: m for m in w24_data["markets"]}
w25_markets = {m["market"]: m for m in w25_data["markets"]}

# ════════════════════════════════════════════
# Step 1: Fetch W24 actual data (2026-06-08 to 2026-06-14)
# ════════════════════════════════════════════
print("Fetching actual market data for W24 (2026-06-08 to 2026-06-14)...")

w24_start = "2026-06-05"  # include Friday before for prev close
w24_end = "2026-06-15"    # include Monday after for last close

def get_weekly_change(ticker, start, end):
    """Get % change from start to end (closest trading days)."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty or len(data) < 2:
            return None, None
        close_start = data["Close"].iloc[0].item()
        close_end = data["Close"].iloc[-1].item()
        pct = (close_end - close_start) / close_start * 100
        return round(pct, 2), close_end
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None, None

w24_actual = {}
for market_code, ticker in TICKER_MAP.items():
    if market_code in w24_markets:
        pct, last_price = get_weekly_change(ticker, w24_start, w24_end)
        if pct is not None:
            direction = "UP" if pct > 0.2 else ("DOWN" if pct < -0.2 else "FLAT")
        else:
            direction = "N/A"
        w24_actual[market_code] = {"pct": pct, "direction": direction}
        print(f"  {market_code} ({ticker}): {pct}% → {direction}")

# ════════════════════════════════════════════
# Step 2: Compare predictions vs actual
# ════════════════════════════════════════════
print("\nComparing W24 predictions vs actual...")

T1_correct = 0
T1_total = 0
T2_correct = 0
T2_total = 0
results = []

for market_code, pred in w24_markets.items():
    actual = w24_actual.get(market_code, {"pct": None, "direction": "N/A"})
    signal = pred["signal"]
    actual_dir = actual["direction"]
    
    # Determine result
    if signal == "BULL" and actual_dir == "UP":
        result = "correct"
    elif signal == "BEAR" and actual_dir == "DOWN":
        result = "correct"
    elif signal == "NEUT":
        result = "passive"  # passively correct / no direction
    elif signal == "BULL" and actual_dir == "DOWN":
        result = "wrong"
    elif signal == "BEAR" and actual_dir == "UP":
        result = "wrong"
    elif actual_dir == "N/A":
        result = "nodata"
    else:
        result = "wrong"  # FLAT against any non-NEUT
    
    tier = pred["tier"]
    if tier == 1 and result == "correct":
        if actual_dir != "N/A":
            T1_correct += 1
    elif tier == 2 and result == "correct":
        if actual_dir != "N/A":
            T2_correct += 1
    
    if tier == 1 and actual_dir != "N/A":
        T1_total += 1
    elif tier == 2 and actual_dir != "N/A":
        T2_total += 1
    
    results.append({
        "market": market_code,
        "name_cn": pred["name_cn"],
        "tier": tier,
        "signal": signal,
        "actual_dir": actual_dir,
        "actual_pct": actual["pct"],
        "result": result,
        "accuracy": pred["accuracy"]
    })
    print(f"  {market_code}: {signal} → {actual_dir} ({actual['pct']}%) → {result}")

T1_rate = round(T1_correct / T1_total * 100, 1) if T1_total > 0 else 0
T2_rate = round(T2_correct / T2_total * 100, 1) if T2_total > 0 else 0
total_correct = T1_correct + T2_correct
total_compared = T1_total + T2_total

print(f"\nTier 1: {T1_correct}/{T1_total} ({T1_rate}%)")
print(f"Tier 2: {T2_correct}/{T2_total} ({T2_rate}%)")
print(f"Total: {total_correct}/{total_compared}")

# ════════════════════════════════════════════
# Step 3: Generate HTML
# ════════════════════════════════════════════
print("\nGenerating HTML...")

def format_pct(val, with_sign=True):
    if val is None:
        return "N/A"
    if with_sign:
        return f"+{val:.2f}%" if val >= 0 else f"{val:.2f}%"
    return f"{val:.2f}%"

def pct_color(val):
    if val is None:
        return "#888"
    return "#4caf50" if val >= 0 else "#f44336"

def pct_color_class(val):
    if val is None:
        return ""
    return "positive" if val >= 0 else "negative"

def signal_html(signal):
    if signal == "BULL":
        return '<span class="signal-bull">▲ BULL</span>'
    elif signal == "BEAR":
        return '<span class="signal-bear">▼ BEAR</span>'
    else:
        return '<span class="signal-neut">─ NEUT</span>'

def result_html(result, actual_pct):
    if result == "correct":
        return f'<span style="color:#4caf50">✅ 正确</span>'
    elif result == "wrong":
        return f'<span style="color:#f44336">❌ 错误</span>'
    elif result == "passive":
        return f'<span style="color:#888">─ 无方向</span>'
    else:
        return '<span style="color:#666">无数据</span>'

def direction_html(direction, actual_pct):
    if direction == "UP":
        return f'<span style="color:#4caf50">▲ UP {format_pct(actual_pct)}</span>'
    elif direction == "DOWN":
        return f'<span style="color:#f44336">▼ DOWN {format_pct(actual_pct)}</span>'
    elif direction == "FLAT":
        return f'<span style="color:#888">─ FLAT ({format_pct(actual_pct)})</span>'
    else:
        return '<span style="color:#666">N/A</span>'

# Generate review table rows
review_rows = ""
for r in sorted(results, key=lambda x: (0 if x["result"] == "correct" else 1 if x["result"] == "passive" else 2, x["tier"], x["market"])):
    tier_badge = f'<span class="tier-badge tier{r["tier"]}">T{r["tier"]}</span>'
    row_style = ""
    if r["result"] == "correct":
        row_style = 'style="background:rgba(76,175,80,0.04)"'
    elif r["result"] == "wrong":
        row_style = 'style="background:rgba(244,67,54,0.04)"'
    review_rows += f"""
    <tr {row_style}>
        <td>{tier_badge} {r["name_cn"]}</td>
        <td>{signal_html(r["signal"])}</td>
        <td>{direction_html(r["actual_dir"], r["actual_pct"])}</td>
        <td>{format_pct(r["actual_pct"])}</td>
        <td>{result_html(r["result"], r["actual_pct"])}</td>
    </tr>"""

# Generate correct/wrong lists
correct_list = [r for r in results if r["result"] == "correct"]
wrong_list = [r for r in results if r["result"] == "wrong"]
passive_list = [r for r in results if r["result"] == "passive"]
nodata_list = [r for r in results if r["result"] == "nodata"]

correct_by_tier = {1: [], 2: []}
wrong_by_tier = {1: [], 2: []}
for r in correct_list:
    correct_by_tier[r["tier"]].append(r)
for r in wrong_list:
    wrong_by_tier[r["tier"]].append(r)

# Generate YTD bars from W25 data
ytd_data = {}
for market_code, m in w25_markets.items():
    # We don't have YTD in W25 JSON, so we need to estimate
    # Actually the W25 data doesn't have YTD - let me use reasonable estimates
    pass

# Actually, the W25 prediction JSON doesn't have YTD values. Let me compute approximate YTD
# by fetching current prices and comparing with start of year
print("Fetching YTD data...")
ytd_start = "2026-01-01"
ytd_end = "2026-06-21"

def get_ytd(ticker):
    try:
        data = yf.download(ticker, start=ytd_start, end=ytd_end, progress=False, auto_adjust=True)
        if data.empty or len(data) < 2:
            return None
        close_start = data["Close"].iloc[0].item()
        close_end = data["Close"].iloc[-1].item()
        return round((close_end - close_start) / close_start * 100, 2)
    except:
        return None

ytd_values = {}
for market_code, ticker in TICKER_MAP.items():
    if market_code in w25_markets:
        ytd = get_ytd(ticker)
        ytd_values[market_code] = ytd
        print(f"  {market_code}: YTD {ytd}%")

# Also fetch weekly change for W25
print("Fetching W25 weekly data (6/15-6/21)...")
w25_start = "2026-06-12"
w25_end = "2026-06-22"
w25_weekly = {}
for market_code, ticker in TICKER_MAP.items():
    if market_code in w25_markets:
        pct, _ = get_weekly_change(ticker, w25_start, w25_end)
        w25_weekly[market_code] = pct
        print(f"  {market_code}: W25 weekly {pct}%")

# Generate Tier 1 table rows for W25 prediction
tier1_markets = sorted([m for m in w25_data["markets"] if m["tier"] == 1],
                       key=lambda x: x["accuracy"], reverse=True)

t1_rows = ""
for m in tier1_markets:
    acc = m["accuracy"] * 100
    ytd = ytd_values.get(m["market"])
    week_chg = w25_weekly.get(m["market"])
    pred_ret = m["pred_ret"] * 100
    
    # Diagnose tags
    diag_html = m["detail"]
        
    row_style = ""
    ytd_html = f'<span style="color:{pct_color(ytd)}">{format_pct(ytd)}</span>' if ytd is not None else '<span style="color:#888">N/A</span>'
    week_html = f'<span style="color:{pct_color(week_chg)}">{format_pct(week_chg)}</span>' if week_chg is not None else '<span style="color:#888">N/A</span>'
    
    # Add overheat/oversold tags
    tags = ""
    if m["tm_z"] > 2.0 and m["mr_z"] < -2.0:
        tags += '<span class="overheat-tag">⚠过热</span> '
    elif m["tm_z"] > 1.5 and m["mr_z"] < -1.5:
        tags += '<span class="overheat-tag">偏热</span> '
    if m["mr_z"] > 1.5:
        tags += '<span class="oversold-tag">超卖</span> '
    elif m["tm_z"] < -1.0:
        tags += '<span class="oversold-tag">弱趋势</span> '
    
    t1_rows += f"""
    <tr {row_style}>
        <td style="font-size:13px"><span class="tier-badge tier1">T1</span> {m["name_cn"]}</td>
        <td>{signal_html(m["signal"])}</td>
        <td style="text-align:right"><span style="color:{pct_color(pred_ret)}">{format_pct(pred_ret)}</span></td>
        <td style="text-align:right;font-size:12px;color:#888">{acc:.1f}%</td>
        <td style="text-align:right;font-size:12px;color:#888">{m["tm_z"]:+.2f}σ</td>
        <td style="text-align:right;font-size:12px;color:#888">{m["mr_z"]:+.2f}σ</td>
        <td style="text-align:right;font-size:12px;color:#888">{m["vr_z"]:+.2f}σ</td>
        <td style="text-align:right">{ytd_html}</td>
        <td style="text-align:right">{week_html}</td>
        <td style="font-size:11px;color:#666">{tags}{diag_html}</td>
    </tr>"""

# Generate Tier 2 table rows for W25
tier2_markets = sorted([m for m in w25_data["markets"] if m["tier"] == 2],
                       key=lambda x: x["accuracy"], reverse=True)

t2_rows = ""
for m in tier2_markets:
    acc = m["accuracy"] * 100
    ytd = ytd_values.get(m["market"])
    week_chg = w25_weekly.get(m["market"])
    pred_ret = m["pred_ret"] * 100
    diag_html = m["detail"]
    
    row_style = ""
    cn_note = ""
    if m["market"] == "CN":
        row_style = 'class="cn-row"'
        cn_note = '<span class="cn-note">⚠ 仅供参考</span>'
    if m["market"] == "HK":
        row_style = 'style="background:rgba(247,151,30,0.03)"'
    
    ytd_html = f'<span style="color:{pct_color(ytd)}">{format_pct(ytd)}</span>' if ytd is not None else '<span style="color:#888">N/A</span>'
    week_html = f'<span style="color:{pct_color(week_chg)}">{format_pct(week_chg)}</span>' if week_chg is not None else '<span style="color:#888">N/A</span>'
    
    t2_rows += f"""
    <tr {row_style}>
        <td style="font-size:13px"><span class="tier-badge tier2">T2</span> {m["name_cn"]}{cn_note}</td>
        <td>{signal_html(m["signal"])}</td>
        <td style="text-align:right"><span style="color:{pct_color(pred_ret)}">{format_pct(pred_ret)}</span></td>
        <td style="text-align:right;font-size:12px;color:#888">{acc:.1f}%</td>
        <td style="text-align:right;font-size:12px;color:#888">{m["tm_z"]:+.2f}σ</td>
        <td style="text-align:right;font-size:12px;color:#888">{m["mr_z"]:+.2f}σ</td>
        <td style="text-align:right;font-size:12px;color:#888">{m["vr_z"]:+.2f}σ</td>
        <td style="text-align:right">{ytd_html}</td>
        <td style="text-align:right">{week_html}</td>
        <td style="font-size:11px;color:#666">{diag_html}</td>
    </tr>"""

# Generate YTD bar chart
ytd_sorted = sorted(ytd_values.items(), key=lambda x: x[1] if x[1] is not None else -999, reverse=True)
max_ytd = max(abs(v) for v in ytd_values.values() if v is not None) if ytd_values else 88.5

ytd_bars = ""
for market_code, ytd in ytd_sorted:
    if ytd is None:
        continue
    name = w25_markets.get(market_code, {}).get("name_cn", market_code)
    if not name:
        continue
    bar_width = min(abs(ytd) / max(max_ytd, 0.01) * 100, 100)
    bar_class = "positive" if ytd >= 0 else "negative"
    bar_color = "#4caf50" if ytd >= 0 else "#f44336"
    ytd_bars += f"""
    <div class="ytd-row">
        <div class="ytd-label">{name}</div>
        <div class="ytd-bar-wrap"><div class="ytd-bar {bar_class}" style="width:{bar_width:.1f}%">{ytd:+.1f}%</div></div>
        <div class="ytd-val" style="color:{bar_color}">{ytd:+.1f}%</div>
    </div>"""

# Build summary text
if correct_list:
    correct_items = "、".join([f'{r["market"]}({format_pct(r["actual_pct"])})' for r in correct_list])
else:
    correct_items = "无"

if wrong_list:
    wrong_items = "、".join([f'{r["market"]}({format_pct(r["actual_pct"])})' for r in wrong_list])
else:
    wrong_items = "无"

total_correct_count = len(correct_list)
total_count = len(results)
T1_correct_count_t = len([r for r in correct_list if r["tier"] == 1])
T2_correct_count_t = len([r for r in correct_list if r["tier"] == 2])

summary_text = f"在 EPU 高企(350.1/P92.8)环境下，上周预测全部为 BULL。实际走势：Tier 1 {T1_rate}% 方向正确({T1_correct}/10)，Tier 2 {T2_rate}% 方向正确。"

# W26 outlook analysis
epu_val = w25_data["epu"]["value"]
epu_pct = w25_data["epu"]["percentile"]

# Build correct/wrong detail blocks
t1_correct_entries = [r for r in correct_list if r["tier"] == 1]
t2_correct_entries = [r for r in correct_list if r["tier"] == 2]
t1_wrong_entries = [r for r in wrong_list if r["tier"] == 1]
t2_wrong_entries = [r for r in wrong_list if r["tier"] == 2]

# Correct block
correct_block = ""
if correct_list:
    correct_block += '<div class="acc-block"><h4 style="color:#4caf50">✓ 预测正确 (' + str(len(correct_list)) + ' 市场)</h4><div class="correct-list">'
    items = []
    for r in correct_list:
        items.append(f'<span class="tier-badge tier{r["tier"]}">T{r["tier"]}</span>{r["name_cn"]} {format_pct(r["actual_pct"])}')
    correct_block += "<p>" + " &nbsp; ".join(items) + "</p>"
    correct_block += "</div></div>"

# Wrong block
wrong_block = ""
if wrong_list:
    wrong_block += '<div class="acc-block"><h4 style="color:#f44336">✗ 预测错误 (' + str(len(wrong_list)) + ' 市场)</h4><div class="wrong-list">'
    items = []
    for r in wrong_list:
        items.append(f'<span class="tier-badge tier{r["tier"]}">T{r["tier"]}</span>{r["name_cn"]} {signal_html(r["signal"])} → {format_pct(r["actual_pct"])}')
    wrong_block += "<p>" + " &nbsp; ".join(items) + "</p>"
    wrong_block += "</div></div>"

# Passive block
passive_block = ""
if passive_list:
    passive_block += '<div class="acc-block"><h4 style="color:#ff9800">─ 无方向信号 (' + str(len(passive_list)) + ')</h4><div style="font-size:12px;line-height:1.6;color:#888">'
    items = []
    for r in passive_list:
        items.append(f'<span class="tier-badge tier{r["tier"]}">T{r["tier"]}</span>{r["name_cn"]} NEUT → {format_pct(r["actual_pct"])}')
    passive_block += "<p>" + " &nbsp; ".join(items) + "</p>"
    passive_block += "</div></div>"

# Nodata block
nodata_block = ""
if nodata_list:
    nodata_block += '<div class="acc-block"><h4 style="color:#666">? 数据缺失 (' + str(len(nodata_list)) + ')</h4><div style="font-size:12px;line-height:1.6;color:#888">'
    items = []
    for r in nodata_list:
        items.append(f'<span class="tier-badge tier{r["tier"]}">T{r["tier"]}</span>{r["name_cn"]}')
    nodata_block += "<p>" + " &nbsp; ".join(items) + "</p>"
    nodata_block += "</div></div>"

# Generate the summary insight block
if T1_rate >= 80:
    insight_color = "#4caf50"
    insight_title = "▸ 上周总结：预测表现优秀"
elif T1_rate >= 60:
    insight_color = "#ff9800"
    insight_title = "▸ 上周总结：预测表现一般"
else:
    insight_color = "#f44336"
    insight_title = "▸ 上周总结：预测表现不及预期"

# ════════════════════════════════════════════
# Assemble html
# ════════════════════════════════════════════
generated_time = "2026-06-21 14:50 CST"

html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>全球市场预测 (第25周 6/15-6/21) · GingerFamily.CN</title>
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
/* YTD bar chart */
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
.acc-block .correct-list,.acc-block .wrong-list{{font-size:12px;line-height:1.6}}
h2.section-title{{font-size:18px;font-weight:700;margin:36px 0 20px;color:#e0e0e0;border-left:3px solid #f7971e;padding-left:12px}}
.outlook-card{{background:linear-gradient(135deg,rgba(79,172,254,0.05),rgba(79,172,254,0.02));border:1px solid rgba(79,172,254,0.15);border-radius:16px;padding:24px;margin-bottom:24px}}
.outlook-card h3{{color:#4facfe;font-size:18px;margin-bottom:16px}}
.warning-box{{background:rgba(244,67,54,0.08);border:1px solid rgba(244,67,54,0.2);border-radius:10px;padding:16px;margin-top:16px}}
.warning-box p{{font-size:13px;color:#f44336;margin:0;line-height:1.6}}
@media(max-width:600px){{.review-card .summary-stats{{flex-direction:column;gap:8px}}.review-card .stat{{min-width:unset}}}}
</style></head><body>
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
<div class="container">
    <a href="/" class="back-link">← 首页</a>
    <div class="page-header">
        <div class="icon">🔮</div>
        <h1 style="background:linear-gradient(135deg,#f7971e,#ffd200);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">全球市场预测</h1>
        <p class="sub">13 市场 · 2 因子模型 · EPU 制度切换 ｜ 预测周期 2026年第25周 (6/15-6/21) ｜ <span class="ts-stamp">生成 {generated_time}</span></p>
    </div>

    <!-- ═══════════════ 上周总结板块 ═══════════════ -->
    <h2 class="section-title">📊 上周预测回顾 (第24周 6/8-6/14)</h2>
    <div class="review-card">
        <p style="color:#888;font-size:13px;margin-bottom:16px">
            上周 (6/8) W24 预测全部市场为 <span class="signal-bull">▲ BULL</span>（CN为NEUT），
            与本周 (6/15-6/21) 实际走势对比：
        </p>
        <div class="summary-stats">
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if T1_rate >= 70 else '#ff9800'}">{T1_rate}%</div>
                <div class="l">Tier 1 方向正确率</div>
            </div>
            <div class="stat">
                <div class="n correct">{T1_correct}/{T1_total}</div>
                <div class="l">Tier 1 正确市场</div>
            </div>
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if T2_rate >= 50 else '#f44336'}">{T2_correct}/{T2_total}</div>
                <div class="l">Tier 2 方向正确</div>
            </div>
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if total_correct/total_compared >= 0.7 else '#ff9800'}">{total_correct}/{total_compared}</div>
                <div class="l">总体方向正确</div>
            </div>
        </div>

        <div class="accuracy-summary">
            {correct_block}
            {wrong_block}
            {passive_block}
            {nodata_block}
        </div>

        <div style="margin-top:16px;overflow-x:auto">
            <table class="predict-table" style="min-width:600px">
                <thead><tr>
                    <th>市场</th><th>预测</th><th>实际方向</th><th>实际涨跌%</th><th>结果</th>
                </tr></thead>
                <tbody>{review_rows}
                </tbody>
            </table>
        </div>

        <div style="margin-top:20px;padding:16px;background:rgba({('76,175,80' if T1_rate >= 70 else '247,151,30' if T1_rate >= 50 else '244,67,54')},0.05);border-radius:10px;border:1px solid rgba({('76,175,80' if T1_rate >= 70 else '247,151,30' if T1_rate >= 50 else '244,67,54')},0.1)">
            <p style="font-size:13px;color:#aaa;margin:0">
                <span style="color:{insight_color};font-weight:700">{insight_title}</span>
                {"EPU 高企环境下趋势延续模式表现优异，模型对市场方向的判断保持稳健。" if T1_rate >= 70 else "EPU 高企环境下预测不确定性增大，部分市场出现反向走势。"}
            </p>
        </div>
    </div>

    <!-- ═══════════════ 本周预测板块 ═══════════════ -->
    <h2 class="section-title">🔮 本周预测 (第25周 6/15-6/21)</h2>

    <!-- EPU Metric Grid -->
    <div class="metric-grid">
        <div class="metric">
            <div class="lbl">EPU 政策不确定性</div>
            <div class="val" style="color:#f44336">{epu_val}<span style="font-size:11px;color:#888"> P{epu_pct}</span></div>
            <div style="font-size:10px;color:#888;margin-top:4px">⚠ 极高不确定性 → 趋势延续模式</div>
        </div>
        <div class="metric">
            <div class="lbl">Tier 1 高置信度</div>
            <div class="val" style="color:#4caf50">{w25_data["tier1_count"]} 市场</div>
            <div style="font-size:10px;color:#888;margin-top:4px">准确率 ≥70% · 可操作</div>
        </div>
        <div class="metric">
            <div class="lbl">Tier 2 参考</div>
            <div class="val" style="color:#ff9800">{w25_data["tier2_count"]} 市场</div>
            <div style="font-size:10px;color:#888;margin-top:4px">55-70% · 仅供参考</div>
        </div>
        <div class="metric">
            <div class="lbl">数据快照</div>
            <div class="val" style="font-size:14px;color:#4facfe;font-family:monospace">{generated_time}</div>
            <div style="font-size:10px;color:#888;margin-top:4px">2-Factor Kalman DFM</div>
        </div>
    </div>

    <!-- HK Special Section -->
    <div class="hk-section">
        <h3>🇭🇰 港股预测</h3>
        <p style="font-size:13px;color:#888;margin-bottom:12px">
            恒生指数为 Tier 2 级别（准确率 66.7%），在 EPU 高企环境下延续趋势延续模式。
            虽信号为 BULL，但近期回调幅度较大，需密切关注地缘政治与流动性风险。
        </p>
    </div>

    <!-- Tier 1 Table -->
    <h3 style="font-size:15px;color:#4caf50;margin-bottom:16px;">★★★ Tier 1 — 高置信度 (≥70% 准确率)</h3>
    <p style="font-size:12px;color:#666;margin-top:-8px;margin-bottom:8px">预测未来12个月收益方向 · YTD = 年初至今实际涨幅 · 周变 = 本周(6/15-6/21)实际变化</p>
    <div style="overflow-x:auto">
    <table class="predict-table" style="min-width:1000px">
        <thead><tr>
            <th>市场</th><th>信号</th><th style="text-align:right">12月预期收益</th>
            <th style="text-align:right">准确率</th><th style="text-align:right">趋势σ</th><th style="text-align:right">反转σ</th><th style="text-align:right">波动率σ</th><th style="text-align:right">YTD</th><th style="text-align:right">周变</th><th>诊断</th>
        </tr></thead>
        <tbody>{t1_rows}
        </tbody>
    </table>
    </div>

    <!-- Tier 2 Table -->
    <h3 style="font-size:15px;color:#ff9800;margin-top:32px;margin-bottom:16px;">★★ Tier 2 — 参考级别 (55-70% 准确率)</h3>
    <div style="overflow-x:auto">
    <table class="predict-table" style="min-width:1000px">
        <thead><tr>
            <th>市场</th><th>信号</th><th style="text-align:right">12月预期收益</th>
            <th style="text-align:right">准确率</th><th style="text-align:right">趋势σ</th><th style="text-align:right">反转σ</th><th style="text-align:right">波动率σ</th><th style="text-align:right">YTD</th><th style="text-align:right">周变</th><th>诊断</th>
        </tr></thead>
        <tbody>{t2_rows}
        </tbody>
    </table>
    </div>
    <p style="font-size:11px;color:#555;margin-top:4px;margin-bottom:24px">⚠ Tier 2 市场准确率较低（55-70%），追踪历史 ≤ 1 年，仅供观察参考。A股数据来源于 akshare，受政策与流动性影响较大。恒生指数受中美关系及全球流动性影响显著。</p>

    <!-- YTD Performance Bar Chart -->
    <h3 style="font-size:15px;color:#e0e0e0;margin-top:40px;margin-bottom:8px;">📊 YTD 表现对比 (截至 2026-06-21)</h3>
    <p style="font-size:12px;color:#666;margin-bottom:16px">各市场年初至今实际涨跌幅 · 绿色=正收益 · 红色=负收益 · 柱宽按最大绝对值等比例缩放</p>
    <div class="ytd-chart" style="max-width:700px">{ytd_bars}
    </div>

    <!-- 本周关键判断 -->
    <h3 style="font-size:15px;color:#f7971e;margin-top:40px;margin-bottom:16px;">⚡ 本周关键判断</h3>
    <div class="card">
        <div class="grid-2">
            <div>
                <h4 style="font-size:14px;color:#4caf50;margin-bottom:8px">🌍 全球共识：全面看多</h4>
                <p style="font-size:13px;color:#aaa;line-height:1.7">
                    EPU 当前值 <strong style="color:#f44336">{epu_val} (P{epu_pct})</strong>，处于历史极端高位。
                    模型在所有市场统一触发 <strong>趋势延续模式</strong>。
                    Tier 1 全部 {w25_data["tier1_count"]} 个市场 + Tier 2 全部 {w25_data["tier2_count"]} 个市场均发出 BULL 信号。
                    这是 EPU 阈值效应下的典型表现：高不确定性环境掩盖了微弱反转信号，趋势因子权重提升。
                </p>
            </div>
            <div>
                <h4 style="font-size:14px;color:#f44336;margin-bottom:8px">⚠ 过热警告</h4>
                <p style="font-size:13px;color:#aaa;line-height:1.7">
                    以下市场出现极端技术与估值信号：
                </p>
                <ul style="font-size:12px;color:#aaa;line-height:1.8;margin-top:8px;padding-left:18px">
                    <li><span style="color:#f44336">韩国 KOSPI</span>：趋势 +3.87σ, 反转 -5.06σ — <strong>极度超买</strong></li>
                    <li><span style="color:#f44336">日经 225</span>：趋势 +2.41σ, 反转 -2.77σ — <strong>显著超买</strong></li>
                    <li><span style="color:#ff9800">意大利 MIB</span>：趋势 +1.29σ, 反转 -1.41σ — 趋于超买</li>
                    <li><span style="color:#ff9800">加拿大 TSX</span>：趋势 +1.17σ, 反转 -0.91σ — 温和超买</li>
                    <li><span style="color:#ff9800">英国 FTSE</span>：趋势 +0.78σ — 趋势偏强</li>
                </ul>
            </div>
        </div>
        <div style="margin-top:20px">
            <h4 style="font-size:14px;color:#4facfe;margin-bottom:8px">📈 EPU 趋势与模式预期</h4>
            <p style="font-size:13px;color:#aaa;line-height:1.7">
                EPU 自 5 月以来维持在 350 附近（P90+），无回落迹象。
                若 EPU 维持当前水平，趋势延续模式将继续主导预测，全球市场 BULL 信号将延续。
                <strong>关键观察点：</strong>若 EPU 回落至 P75 以下（约 <200），模型将切换回均值回归模式，
                届时超买市场（韩国、日本）可能出现显著反转信号。
                当前建议：关注 Tier 1 高置信度市场，对超买市场谨慎追高。
            </p>
        </div>
    </div>

    <!-- ═══════════════ 下周展望板块 ═══════════════ -->
    <h2 class="section-title">🔭 下周展望 (第26周 6/22-6/28)</h2>
    <div class="outlook-card">
        <h3>📡 基于 EPU 极端高位的 W26 市场前瞻</h3>
        <p style="font-size:13px;color:#aaa;line-height:1.7;margin-bottom:16px">
            当前 EPU = <strong style="color:#f44336">{epu_val} (P{epu_pct})</strong>，处于历史极端高位区间。
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

        <h4 style="font-size:14px;color:#4facfe;margin-bottom:8px">🎯 W26 关注点</h4>
        <div class="grid-2">
            <div>
                <ul style="font-size:12px;color:#aaa;line-height:1.8;padding-left:18px">
                    <li><span style="color:#4facfe">美联储动向</span>：关注 6 月 FOMC 会议纪要与官员讲话</li>
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
            高 EPU（P{epu_pct}）环境下，模型的趋势延续模式虽然历史表现稳健，但极端政治经济环境可能引发非线性市场反应。
            不建议在此阶段过度集中于单一市场或方向，尤其是已出现极端超买信号的韩国和日本市场。
            下周重点关注 EPU 是否出现拐点信号，若 EPU 回落至 250 以下，部分市场的均值回归压力将显著释放。
            </p>
        </div>
    </div>

    <!-- 模型说明 + 预测周期 -->
    <div class="insight-box" style="margin-top:32px">
        <p>💡 <strong>模型说明：</strong>2-factor Kalman DFM 提取方向和幅度两个潜因子，EPU > P75 时自动切换为趋势延续模式。
        Tier 1 市场（≥70% 回测准确率，≥5年历史数据）可用于配置参考；Tier 2 市场（55-70%，≤1年追踪）仅供观察。
        预测收益率为年化估算，不代表短期走势。模型基于过去 5 年滚动窗口回测验证。
        数据来源：yfinance / FRED / akshare · 模型：Global Economy Lab · 
        <strong>预测周期：2026年第25周 (6/15-6/21) · 生成时间：{generated_time}</strong></p>
    </div>

    <div style="text-align:center;margin-top:24px;font-size:11px;color:#444">
        <span class="ts-stamp">第25周存档</span>
        <span style="margin:0 8px">|</span>
        <span>上周准确率 Tier1: {T1_rate}% ({T1_correct}/{T1_total}) | Tier2: {T2_rate}% ({T2_correct}/{T2_total})</span>
        <span style="margin:0 8px">|</span>
        <span>EPU: {epu_val} (P{epu_pct} · HIGH_EPU 制度)</span>
    </div>
</div>
<footer>GingerFamily.CN · Global Economy Lab 驱动 · 数据仅供参考不构成投资建议</footer>
</body></html>"""

# ════════════════════════════════════════════
# Write output files
# ════════════════════════════════════════════
with open("output/prediction.html", "w") as f:
    f.write(html)
print("Written: output/prediction.html")

with open("output/prediction202625.html", "w") as f:
    f.write(html)
print("Written: output/prediction202625.html")

print("\nDone!")
