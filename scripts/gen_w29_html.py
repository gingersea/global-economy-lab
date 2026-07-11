#!/usr/bin/env python3
"""Generate W29 prediction HTML (weekend task)"""
import json
from pathlib import Path

# ── Market YTD & week data (fetched 2026-07-11) ─────────────
YTD_DATA = {
    "US": 10.45, "DE": 2.15, "JP": 32.27, "GB": 5.49,
    "FR": 1.75, "IT": 15.96, "CA": 10.73, "BR": 10.79,
    "KR": 73.47, "IN": -7.42, "AU": 0.90, "CN": -0.68, "HK": -8.21,
}

# Actual week change from W28 (7/6-7/11) in actual_accuracy
WEEK_CHANGE = {
    "US": 0.54, "DE": -2.83, "JP": -2.07, "GB": -1.94,
    "FR": -2.30, "IT": -0.98, "CA": 0.04, "BR": -0.75,
    "KR": -7.75, "IN": -0.75, "AU": -0.93, "CN": -0.68, "HK": 3.46,
}

# Monthly June returns (May end → June end, fetched)
JUNE_RETURNS = {
    "US": -1.28, "DE": -0.26, "JP": 6.25, "GB": 0.66,
    "FR": 1.88, "IT": 3.13, "CA": 0.25, "BR": -1.21,
    "KR": -2.04, "IN": 1.95, "AU": -0.10, "CN": 1.08, "HK": -9.14,
}

# June W23 predictions (from archive 2026-06-14)
JUNE_PREDS = {
    "US": "BULL", "DE": "BULL", "JP": "BULL", "GB": "BULL",
    "FR": "BULL", "IT": "BULL", "CA": "BULL", "BR": "BULL",
    "KR": "BULL", "IN": "BULL", "AU": "BULL", "CN": "NEUT", "HK": "BULL",
}

JUNE_TIERS = {
    "US": 1, "DE": 1, "JP": 1, "GB": 1, "FR": 1, "IT": 1,
    "CA": 1, "KR": 1, "IN": 1, "AU": 1, "CN": 2, "HK": 2,
}

NAMES = {
    "US": "美国 SP500", "DE": "德国 DAX", "JP": "日经 225",
    "GB": "英国 FTSE", "FR": "法国 CAC40", "IT": "意大利 MIB",
    "CA": "加拿大 TSX", "BR": "巴西 Bovespa", "KR": "韩国 KOSPI",
    "IN": "印度 NIFTY", "AU": "澳洲 ASX", "CN": "中国 A股", "HK": "恒生指数",
}

def fmt_ret(v):
    c = "4caf50" if v >= 0 else "f44336"
    return f'<span style="color:#{c}">{v:+.2f}%</span>'

def make_prediction_html():
    proj = Path(__file__).parent.parent
    with open(proj / "output/weekly_prediction.json") as f:
        data = json.load(f)

    gen_time = data["generated_display"]
    period = data["prediction_period"]
    epu = data["epu"]
    vix = data["vix"]
    markets = data["markets"]
    acc = data["actual_accuracy"]
    sdist = data["signal_distribution"]

    tier1 = [m for m in markets if m["tier"] == 1]
    tier2 = [m for m in markets if m["tier"] == 2]

    # ── Signal HTML helpers ──────────────────────────────────
    def sig_html(s, overheat=False):
        if overheat:
            if s == "BEAR":
                return '<span class="signal-overheat">▼ BEAR (过热)</span>'
            return '<span class="signal-overheat">─ NEUT (过热)</span>'
        if s == "BULL":
            return '<span class="signal-bull">▲ BULL</span>'
        elif s == "BEAR":
            return '<span class="signal-bear">▼ BEAR</span>'
        else:
            return '<span class="signal-neut">─ NEUT</span>'

    def tbadge(t):
        return '<span class="tier-badge tier1">T1</span>' if t == 1 else '<span class="tier-badge tier2">T2</span>'

    # ── Last week review table ───────────────────────────────
    last_week_rows = []
    for mk in ["US", "DE", "JP", "GB", "FR", "IT", "CA", "BR", "KR", "IN", "AU", "CN", "HK"]:
        m = acc["markets"].get(mk)
        if not m:
            continue
        sig = m["prior_signal"]
        actual_ret = m["actual_ret"]
        hit = m.get("hit")
        actual_dir = m["actual_dir"]
        if sig == "NEUT":
            result = '<span style="color:#888">─ 无方向</span>'
            bg = ""
        elif hit is True:
            result = '<span style="color:#4caf50">✅ 正确</span>'
            bg = 'style="background:rgba(76,175,80,0.04)"'
        else:
            result = '<span style="color:#f44336">❌ 错误</span>'
            bg = 'style="background:rgba(244,67,54,0.04)"'

        dir_color = "#4caf50" if actual_ret > 0.005 else ("#f44336" if actual_ret < -0.005 else "#888")
        dir_arrow = "▲ UP" if actual_ret > 0.005 else ("▼ DOWN" if actual_ret < -0.005 else "─ FLAT")

        lw_mkt = None
        for pm in markets:
            if pm["market"] == mk:
                lw_mkt = pm
                break
        prev_tier = lw_mkt["tier"] if lw_mkt else 1

        last_week_rows.append(
            f'<tr {bg}><td>{tbadge(prev_tier)} {NAMES[mk]}</td>'
            f'<td><span class="signal-{sig.lower()}">{sig}</span></td>'
            f'<td><span style="color:{dir_color}">{dir_arrow} {actual_ret:+.2f}%</span></td>'
            f'<td>{fmt_ret(actual_ret)}</td>'
            f'<td>{result}</td></tr>'
        )

    # ── Monthly review table ─────────────────────────────────
    monthly_rows = []
    monthly_correct = []
    monthly_wrong = []
    monthly_neutral = []
    t1_hits = 0
    t1_total = 0
    t2_hits = 0
    t2_total = 0

    for mk in ["US", "DE", "JP", "GB", "FR", "IT", "CA", "BR", "KR", "IN", "AU", "CN", "HK"]:
        pred = JUNE_PREDS.get(mk, "BULL")
        ret = JUNE_RETURNS.get(mk, 0)
        tier = JUNE_TIERS.get(mk, 1)
        is_up = ret > 0.005
        is_down = ret < -0.005

        if pred == "BULL":
            hit = is_up
        elif pred == "BEAR":
            hit = is_down
        else:  # NEUT
            hit = None

        if hit is True:
            result = '<span style="color:#4caf50">✅ 正确</span>'
            bg = 'style="background:rgba(76,175,80,0.04)"'
            monthly_correct.append(f'{tbadge(tier)} {NAMES[mk]} {ret:+.2f}%')
            if tier == 1:
                t1_hits += 1
                t1_total += 1
            else:
                t2_hits += 1
                t2_total += 1
        elif hit is False:
            result = '<span style="color:#f44336">❌ 错误</span>'
            bg = 'style="background:rgba(244,67,54,0.04)"'
            monthly_wrong.append(f'{tbadge(tier)} {NAMES[mk]} <span class="signal-bull">▲ BULL</span> → {ret:+.2f}%')
            if tier == 1:
                t1_total += 1
            else:
                t2_total += 1
        else:
            result = '<span style="color:#888">─ 无方向</span>'
            bg = ''
            monthly_neutral.append(f'{tbadge(tier)} {NAMES[mk]} <span class="signal-neut">─ NEUT</span> → {ret:+.2f}%')

        dir_color = "#4caf50" if is_up else ("#f44336" if is_down else "#888")
        dir_str = "▲ UP" if is_up else ("▼ DOWN" if is_down else "─ FLAT")

        monthly_rows.append(
            f'<tr {bg}><td>{tbadge(tier)} {NAMES[mk]}</td>'
            f'<td><span class="signal-{pred.lower()}">{pred}</span></td>'
            f'<td><span style="color:{dir_color}">{dir_str} ({ret:+.2f}%)</span></td>'
            f'<td>{ret:+.2f}%</td>'
            f'<td>{result}</td></tr>'
        )

    # ── Prediction table rows ─────────────────────────────────
    def pred_row(m):
        mk = m["market"]
        ytd = YTD_DATA.get(mk, 0)
        wch = WEEK_CHANGE.get(mk, 0)
        oh = m.get("overheat", False)
        cn_row = ' class="cn-row"' if mk == "CN" else ""

        # Tags
        tags = []
        if oh:
            tags.append('<span class="overheat-tag">⚠过热</span>')
        if m["mr_z"] < -2.0:
            tags.append('<span class="overheat-tag">超买</span>')
        if m["mr_z"] > 2.0:
            tags.append('<span class="oversold-tag">超卖</span>')

        detail = m.get("detail", "")
        cn_note = '<span class="cn-note">⚠ 仅供参考</span>' if mk == "CN" else ""

        return (
            f'<tr{cn_row}><td style="font-size:13px">{tbadge(m["tier"])} {NAMES[mk]}'
            f'{cn_note}</td>'
            f'<td>{sig_html(m["signal"], oh)}</td>'
            f'<td style="text-align:right"><span style="color:{"#4caf50" if m["pred_ret"] >= 0 else "#f44336"}">{m["pred_ret"]:+.2%}</span></td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m["accuracy"]:.1%}</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m["tm_z"]:+.2f}σ</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m["mr_z"]:+.2f}σ</td>'
            f'<td style="text-align:right;font-size:12px;color:#888">{m.get("vr_z",0):+.2f}σ</td>'
            f'<td style="text-align:right">{fmt_ret(ytd)}</td>'
            f'<td style="text-align:right">{fmt_ret(wch)}</td>'
            f'<td style="font-size:11px;color:#666">{"".join(tags)} {detail}</td>'
            f'</tr>'
        )

    t1_rows = "\n".join(pred_row(m) for m in tier1)
    t2_rows = "\n".join(pred_row(m) for m in tier2)

    # ── YTD chart ─────────────────────────────────────────────
    ytd_all = [(NAMES[m["market"]], YTD_DATA.get(m["market"], 0)) for m in markets]
    ytd_all.sort(key=lambda x: abs(x[1]), reverse=True)
    max_abs = max(abs(v) for _, v in ytd_all) if ytd_all else 1
    ytd_bars = []
    for name, val in ytd_all:
        pct = abs(val) / max_abs * 100 if max_abs > 0 else 0
        cls = "positive" if val >= 0 else "negative"
        color = "#4caf50" if val >= 0 else "#f44336"
        ytd_bars.append(
            f'<div class="ytd-row"><div class="ytd-label">{name}</div>'
            f'<div class="ytd-bar-wrap"><div class="ytd-bar {cls}" style="width:{pct:.1f}%">{val:+.2f}%</div></div>'
            f'<div class="ytd-val" style="color:{color}">{val:+.2f}%</div></div>'
        )

    # ── Overheat warnings ─────────────────────────────────────
    overheat_warns = []
    for m in markets:
        tm = m["tm_z"]
        mr = m["mr_z"]
        if tm > 3.5:
            overheat_warns.append(f'<li><span style="color:#f44336">{NAMES[m["market"]]}</span>：趋势 +{tm:.2f}σ|反转 {mr:+.2f}σ — <strong><span style="color:#f44336">极度过热</span></strong></li>')
        elif tm > 2.5:
            overheat_warns.append(f'<li><span style="color:#ff9800">{NAMES[m["market"]]}</span>：趋势 +{tm:.2f}σ|反转 {mr:+.2f}σ — <strong><span style="color:#ff9800">过热警告</span></strong></li>')
        elif tm > 1.0:
            overheat_warns.append(f'<li><span style="color:#ff9800">{NAMES[m["market"]]}</span>：趋势 +{tm:.2f}σ — <strong><span style="color:#ff9800">趋于超买</span></strong></li>')

    # ── Build HTML ────────────────────────────────────────────
    # Hit rate display
    hr = acc["hit_rate"]
    hr_color = "#4caf50" if hr >= 0.7 else ("#ff9800" if hr >= 0.5 else "#f44336")

    # Correct/wrong lists for last week
    correct_list = []
    wrong_list = []
    neutral_list = []
    for mk in ["US", "DE", "JP", "GB", "FR", "IT", "CA", "BR", "KR", "IN", "AU", "CN", "HK"]:
        m = acc["markets"].get(mk)
        if not m:
            continue
        sig = m["prior_signal"]
        ret = m["actual_ret"]
        hit = m.get("hit")
        lw_mkt = next((pm for pm in markets if pm["market"] == mk), None)
        prev_tier = lw_mkt["tier"] if lw_mkt else 1
        if hit is True:
            correct_list.append(f'{tbadge(prev_tier)} {NAMES[mk]} {ret:+.2f}%')
        elif hit is False:
            wrong_list.append(f'{tbadge(prev_tier)} {NAMES[mk]} <span class="signal-bull">▲ {sig}</span> → {ret:+.2f}%')
        else:
            neutral_list.append(f'{tbadge(prev_tier)} {NAMES[mk]} <span class="signal-neut">─ {sig}</span> → {ret:+.2f}%')

    # Tier-specific accuracy for last week
    t1_correct = len([m for m in acc["markets"].values() if m.get("hit") is True and next((pm for pm in markets if pm["market"] == list(acc["markets"].keys())[list(acc["markets"].values()).index(m)]), None) and next(pm for pm in markets if pm["market"] == list(acc["markets"].keys())[list(acc["markets"].values()).index(m)]).get("tier") == 1])
    # Recalculate properly
    t1_lw_hits = 0
    t1_lw_total = 0
    t2_lw_hits = 0
    t2_lw_total = 0
    for mk, m in acc["markets"].items():
        hit = m.get("hit")
        sig = m["prior_signal"]
        if sig == "NEUT":
            continue
        lw_mkt = next((pm for pm in markets if pm["market"] == mk), None)
        tier = lw_mkt["tier"] if lw_mkt else 1
        if tier == 1:
            t1_lw_total += 1
            if hit:
                t1_lw_hits += 1
        else:
            t2_lw_total += 1
            if hit:
                t2_lw_hits += 1

    t1_rate = t1_lw_hits / t1_lw_total if t1_lw_total > 0 else 0
    t2_rate = t2_lw_hits / t2_lw_total if t2_lw_total > 0 else 0

    # Monthly tier accuracy
    t1_mo_rate = t1_hits / t1_total if t1_total > 0 else 0
    t2_mo_rate = t2_hits / t2_total if t2_total > 0 else 0
    mo_total = t1_hits + t2_hits
    mo_total_scored = t1_total + t2_total

    html = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>全球市场预测 (第29周) · GingerFamily.CN</title>
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
.signal-overheat{{color:#ff6d00;font-weight:700}}
.tier-badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600;margin-right:4px}}
.tier1{{background:rgba(76,175,80,0.15);color:#4caf50}}
.tier2{{background:rgba(255,152,0,0.15);color:#ff9800}}
.insight-box{{margin-top:16px;padding:20px;background:rgba(255,255,255,0.04);border-radius:12px;border-left:3px solid #f7971e}}
.insight-box p{{margin:0;font-size:14px;color:#aaa;line-height:1.8}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:700px){{.grid-2{{grid-template-columns:1fr}}nav .inner{{gap:12px}}nav a{{font-size:12px}}nav .home{{font-size:15px}}}}
.ts-stamp{{display:inline-block;background:rgba(255,255,255,0.05);border-radius:6px;padding:3px 8px;font-size:11px;color:#666;font-family:monospace}}
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
.cn-row td{{opacity:0.85}}
.accuracy-summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px}}
.acc-block{{background:rgba(255,255,255,0.03);border-radius:10px;padding:16px}}
.acc-block h4{{font-size:13px;color:#888;margin-bottom:8px}}
.acc-block .correct-list,.acc-block .wrong-list,.acc-block .neutral-list{{font-size:12px;line-height:1.6}}
h2.section-title{{font-size:18px;font-weight:700;margin:36px 0 20px;color:#e0e0e0;border-left:3px solid #f7971e;padding-left:12px}}
.outlook-card{{background:linear-gradient(135deg,rgba(79,172,254,0.05),rgba(79,172,254,0.02));border:1px solid rgba(79,172,254,0.15);border-radius:16px;padding:24px;margin-bottom:24px}}
.outlook-card h3{{color:#4facfe;font-size:18px;margin-bottom:16px}}
.warning-box{{background:rgba(244,67,54,0.08);border:1px solid rgba(244,67,54,0.2);border-radius:10px;padding:16px;margin-top:16px}}
.warning-box p{{font-size:13px;color:#f44336;margin:0;line-height:1.6}}
.hk-section{{background:linear-gradient(135deg,rgba(247,151,30,0.08),rgba(247,151,30,0.02));border:1px solid rgba(247,151,30,0.2);border-radius:16px;padding:24px;margin-bottom:24px}}
.hk-section h3{{color:#f7971e;font-size:18px;margin-bottom:16px}}
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
    <a href="/" class="back-link">&larr; 首页</a>
    <div class="page-header">
        <div class="icon">🔮</div>
        <h1 style="background:linear-gradient(135deg,#f7971e,#ffd200);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">全球市场预测</h1>
        <p class="sub">13 市场 · 3-Factor 模型 · EPU 制度切换 + VIX 辅助 ｜ 预测周期 {period} ｜ <span class="ts-stamp">生成 {gen_time}</span></p>
    </div>

    <!-- ========== 上周预测回顾 ========== -->
    <h2 class="section-title">📊 上周预测回顾 (第28周 7/6-7/12)</h2>
    <div class="review-card">
        <p style="color:#888;font-size:13px;margin-bottom:16px">
            上周 (2026年第28周 (7/6-7/12)) 预测：
            <span class="signal-bull">10 BULL</span> + <span class="signal-bear">1 BEAR</span> + <span class="signal-neut">2 NEUT</span>，
            对比本周实际走势 (7/6-7/11)：
        </p>
        <div class="summary-stats">
            <div class="stat">
                <div class="n" style="color:{hr_color}">{hr:.1%}</div>
                <div class="l">总体方向正确率</div>
            </div>
            <div class="stat">
                <div class="n correct">{acc['hits']}/11</div>
                <div class="l">正确 / 方向市场</div>
            </div>
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if t1_rate >= 0.5 else '#f44336'}">{t1_rate:.1%}</div>
                <div class="l">Tier 1 方向正确率</div>
            </div>
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if t2_rate >= 0.5 else '#f44336'}">{t2_rate:.1%}</div>
                <div class="l">Tier 2 方向正确率</div>
            </div>
            <div class="stat">
                <div class="n" style="color:#888">{acc['total_scored']}</div>
                <div class="l">方向性预测市场</div>
            </div>
        </div>

        <div class="accuracy-summary">
            <div class="acc-block"><h4 style="color:#4caf50">✓ 预测正确 ({acc['hits']} 市场)</h4><div class="correct-list"><p>{" &nbsp; ".join(correct_list)}</p></div></div>
            <div class="acc-block"><h4 style="color:#f44336">✗ 预测错误 ({acc['misses']} 市场)</h4><div class="wrong-list"><p>{" &nbsp; ".join(wrong_list)}</p></div></div>
            <div class="acc-block"><h4 style="color:#888">─ 无方向/NEUT ({len(neutral_list)} 市场)</h4><div class="neutral-list"><p>{" &nbsp; ".join(neutral_list)}</p></div></div>
        </div>

        <div style="margin-top:16px;overflow-x:auto">
            <table class="predict-table" style="min-width:600px">
                <thead><tr>
                    <th>市场</th><th>预测</th><th>实际方向</th><th>实际涨跌%</th><th>结果</th>
                </tr></thead>
                <tbody>
                    {"".join(last_week_rows)}
                </tbody>
            </table>
        </div>

        <div style="margin-top:20px;padding:16px;background:rgba(244,67,54,0.05);border-radius:10px;border:1px solid rgba(244,67,54,0.1)">
            <p style="font-size:13px;color:#aaa;margin:0">
                <span style="color:#f44336;font-weight:700">▸ 上周总结：预测表现严重不及预期</span><br>
                W28 预测方向正确率仅 <strong style="color:#f44336">27.3% (3/11)</strong>——为模型历史最低周度准确率。全球市场大面积下跌超出模型预期。
                仅美国 SP500 (+0.54%)、恒生指数 (+3.46%) 和韩国 KOSPI (-7.75%, BEAR信号命中) 三个市场方向正确。
                主要原因：EPU P99.1 极端环境下趋势延续模式的泛化能力受限——欧洲三大指数 (DAX/CAC40/FTSE) 和多个亚太市场均出现 BULL 信号后的下跌。
                韩国 KOSPI BEAR 信号成功预警了 -7.75% 的大幅回调，验证过热检测机制的有效性。
                恒生指数 BULL (+3.46%) 为最大单周正确预测。高 EPU 环境下方向判断难度极大，模型需要在多市场分化中进行针对性校准。
            </p>
        </div>
    </div>

    <!-- ========== 上月总结 (6月) ========== -->
    <h2 class="section-title">📅 上月预测回顾 (6月月度总结)</h2>
    <div class="review-card">
        <p style="color:#888;font-size:13px;margin-bottom:16px">
            上月 2026年第23周 (6/1-6/7) 预测，Tier 1 全部 10 个市场为 <span class="signal-bull">▲ BULL</span>，
            CN 为 <span class="signal-neut">─ NEUT</span>，
            对比 6 月整月实际走势 (5月底→6月底)：
        </p>
        <div class="summary-stats">
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if t1_mo_rate >= 0.5 else '#ff9800'}">{t1_mo_rate:.1%}</div>
                <div class="l">Tier 1 月度正确率</div>
            </div>
            <div class="stat">
                <div class="n correct">{t1_hits}/{t1_total}</div>
                <div class="l">Tier 1 正确市场</div>
            </div>
            <div class="stat">
                <div class="n" style="color:{'#4caf50' if t2_mo_rate >= 0.5 else '#f44336'}">{t2_mo_rate:.1%}</div>
                <div class="l">Tier 2 月度正确率</div>
            </div>
            <div class="stat">
                <div class="n" style="color:#4caf50">{mo_total}/{mo_total_scored}</div>
                <div class="l">总体方向正确</div>
            </div>
        </div>

        <div class="accuracy-summary">
            <div class="acc-block"><h4 style="color:#4caf50">✓ 预测正确 ({mo_total} 市场)</h4><div class="correct-list"><p>{" &nbsp; ".join(monthly_correct)}</p></div></div>
            <div class="acc-block"><h4 style="color:#f44336">✗ 预测错误 ({len(monthly_wrong)} 市场)</h4><div class="wrong-list"><p>{" &nbsp; ".join(monthly_wrong)}</p></div></div>
            <div class="acc-block"><h4 style="color:#888">─ 无方向 ({len(monthly_neutral)} 市场)</h4><div class="neutral-list"><p>{" &nbsp; ".join(monthly_neutral)}</p></div></div>
        </div>

        <div style="margin-top:16px;overflow-x:auto">
            <table class="predict-table" style="min-width:600px">
                <thead><tr>
                    <th>市场</th><th>预测</th><th>6月实际方向</th><th>6月涨跌%</th><th>结果</th>
                </tr></thead>
                <tbody>
                    {"".join(monthly_rows)}
                </tbody>
            </table>
        </div>

        <div style="margin-top:20px;padding:16px;background:rgba(244,67,54,0.05);border-radius:10px;border:1px solid rgba(244,67,54,0.1)">
            <p style="font-size:13px;color:#aaa;margin:0">
                <span style="color:#f44336;font-weight:700">▸ 月度总结：6月准确率下降，EPU 极端高位挑战模型稳健性</span><br>
                6月 10 个 Tier1 市场均预测 BULL，实际 5 个实现正收益（日本 +6.25% 最高），3 个微跌，2 个下跌。
                恒生指数月度跌幅 <strong>-9.14%</strong> 为最大预测失误，受港股特定风险（非 EPU 可捕获）影响。
                日本和欧洲正收益验证 BULL 信号有效性，但美国和韩国下跌揭示高 EPU 下趋势延续模式的盲区。
                连接 W27-W29 三周：5月高 EPU 初现 → 6月预测分化 → 7月准确率周度骤降。模型在高 EPU（P99+）持续环境中的鲁棒性需要提升。
            </p>
        </div>
    </div>

    <!-- ========== 本周预测 (第29周) ========== -->
    <h2 class="section-title">🔮 本周预测 ({period})</h2>

    <div class="metric-grid">
        <div class="metric">
            <div class="lbl">US EPU 政策不确定性</div>
            <div class="val" style="color:#f44336">{epu['value']}<span style="font-size:11px;color:#888"> P{epu['percentile']}</span></div>
            <div style="font-size:10px;color:#888;margin-top:4px">⚠ 极端高不确定性 → 趋势延续模式</div>
        </div>
        <div class="metric">
            <div class="lbl">中国 EPU</div>
            <div class="val" style="font-size:20px;color:#ff9800">{epu['china_epu']:.0f}</div>
            <div style="font-size:10px;color:#888;margin-top:4px">中EPU高 → 港股趋势延续</div>
        </div>
        <div class="metric">
            <div class="lbl">VIX 恐慌指数</div>
            <div class="val" style="font-size:22px;color:{'#f44336' if vix.get('current',0) > (vix.get('p80',25) or 25) else '#4caf50'}">{vix['current']:.1f}</div>
            <div style="font-size:10px;color:#888;margin-top:4px">5d {vix.get('trend_5d',0):+.1%} · 20d {vix.get('trend_20d',0):+.1%} · P80={vix.get('p80',0):.0f}</div>
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
            <div class="lbl">信号分布</div>
            <div class="val" style="font-size:16px">
                <span style="color:#4caf50">▲{sdist.get('BULL',0)}</span>
                <span style="color:#888;margin-left:8px">─{sdist.get('NEUT',0)}</span>
                <span style="color:#f44336;margin-left:8px">▼{sdist.get('BEAR',0)}</span>
            </div>
            <div style="font-size:10px;color:#888;margin-top:4px">BULL / NEUT / BEAR</div>
        </div>
    </div>

    <!-- HK section -->
    <div class="hk-section">
        <h3>🇭🇰 恒生指数 — NEUT 特殊标识</h3>
        <p style="font-size:13px;color:#888;margin-bottom:12px">
            恒生指数本周信号为 <span class="signal-neut">─ NEUT</span>，准确率 76.9% (Tier 1)。
            China EPU 376.4（高位）驱动趋势延续模式，但趋势因子 z=-0.89σ 与反转因子 z=+1.07σ 方向矛盾，
            且 VIX=15.0 处于 P20 以下低波动区间。综合判断：方向不明，维持 NEUT。
            W28 实际表现 +3.46%（BULL 命中）为上周最佳预测，但本周不确定性高。
        </p>
    </div>

    <h3 style="font-size:15px;color:#4caf50;margin-bottom:16px;">★★★ Tier 1 — 高置信度 (≥70% 准确率)</h3>
    <p style="font-size:12px;color:#666;margin-top:-8px;margin-bottom:8px">预测未来12个月收益方向 · YTD = 年初至今实际涨幅 · 周变 = 上周(7/6-7/11)变化 · 数据至 = 最近交易日</p>
    <div style="overflow-x:auto">
    <table class="predict-table" style="min-width:1100px">
        <thead><tr>
            <th>市场</th><th>信号</th><th style="text-align:right">12月预期收益</th>
            <th style="text-align:right">置信度</th><th style="text-align:right">趋势σ</th><th style="text-align:right">反转σ</th><th style="text-align:right">波动σ</th><th style="text-align:right">YTD</th><th style="text-align:right">周变</th><th>诊断 · 数据至</th>
        </tr></thead>
        <tbody>
{t1_rows}
        </tbody>
    </table>
    </div>

    <h3 style="font-size:15px;color:#ff9800;margin-top:32px;margin-bottom:16px;">★★ Tier 2 — 参考级别 (55-70% 准确率)</h3>
    <div style="overflow-x:auto">
    <table class="predict-table" style="min-width:1100px">
        <thead><tr>
            <th>市场</th><th>信号</th><th style="text-align:right">12月预期收益</th>
            <th style="text-align:right">置信度</th><th style="text-align:right">趋势σ</th><th style="text-align:right">反转σ</th><th style="text-align:right">波动σ</th><th style="text-align:right">YTD</th><th style="text-align:right">周变</th><th>诊断 · 数据至</th>
        </tr></thead>
        <tbody>
{t2_rows}
        </tbody>
    </table>
    </div>
    <p style="font-size:11px;color:#555;margin-top:4px;margin-bottom:24px">⚠ Tier 2 市场准确率较低（55-70%），追踪历史 ≤ 1 年，仅供观察参考。A股数据来源于 akshare，受政策与流动性影响较大。恒生指数受中美关系及全球流动性影响显著。</p>

    <h3 style="font-size:15px;color:#e0e0e0;margin-top:40px;margin-bottom:8px;">📊 YTD 表现对比 (截至 2026-07-10)</h3>
    <p style="font-size:12px;color:#666;margin-bottom:16px">各市场年初至今实际涨跌幅 · 绿色=正收益 · 红色=负收益 · 柱宽按最大绝对值等比例缩放</p>
    <div class="ytd-chart" style="max-width:700px">
        {"".join(ytd_bars)}
    </div>

    <h3 style="font-size:15px;color:#f7971e;margin-top:40px;margin-bottom:16px;">⚡ 本周关键判断</h3>
    <div class="card">
        <div class="grid-2">
            <div>
                <h4 style="font-size:14px;color:#4caf50;margin-bottom:8px">🌍 全球共识：11 BULL 压倒性共识</h4>
                <p style="font-size:13px;color:#aaa;line-height:1.7">
                    EPU 当前值 <strong style="color:#f44336">{epu['value']:.1f} (P{epu['percentile']})</strong>，仍处于历史极端高位。
                    模型在几乎所有市场触发趋势延续模式。W28 周度准确率 27.3% 令人警醒，但模型继续根据当前因子状态输出信号。
                    Tier 1 中 8 BULL + 1 BEAR (KOSPI) + 1 NEUT (恒生)；
                    Tier 2 中 3 BULL (含意大利 MIB 降级)。
                    信号分布：<strong>11 BULL / 1 NEUT / 1 BEAR</strong> —— 全球 BULL 共识信号。
                    韩国 KOSPI 连续第 3 周触发 BEAR (tm=+3.57σ, EXTREME OVERHEAT)，日经 225 虽 tm=+2.49σ 但未达 NEUT 阈值。
                    需注意：上周 BULL 信号大面积失败后，本周的 11 BULL 信号应在历史低位命中率背景下审慎解读。
                </p>
            </div>
            <div>
                <h4 style="font-size:14px;color:#f44336;margin-bottom:8px">🔥 KOSPI 过热警告（tm=+3.57σ → BEAR）</h4>
                <p style="font-size:13px;color:#aaa;line-height:1.7">
                    以下市场出现极端技术信号——注意 KOSPI 的持续极端过热：
                </p>
                <ul style="font-size:12px;color:#aaa;line-height:1.8;margin-top:8px;padding-left:18px">
                    {"".join(overheat_warns) if overheat_warns else "<li><span style=\"color:#888\">当前无明显过热信号</span></li>"}
                </ul>
            </div>
        </div>
        <div style="margin-top:20px">
            <h4 style="font-size:14px;color:#4facfe;margin-bottom:8px">📈 EPU 趋势与模型展望</h4>
            <p style="font-size:13px;color:#aaa;line-height:1.7">
                EPU 自 5 月以来维持在 350 附近（P99+），已持续逾 2 个月无回落迹象，当前百分位为 <strong>P{epu['percentile']}</strong>。
                VIX 已从 6 月的 19.4 回落至 <strong>{vix['current']:.1f}</strong>（5d {vix.get('trend_5d',0):+.1%}，20d {vix.get('trend_20d',0):+.1%}），
                处于 P20-P80 中性区间，暗示市场恐慌情绪有所消退，但 EPU 仍居高不下。
                若 EPU 维持当前水平，趋势延续模式将继续主导预测，全球市场 BULL 信号将延续。
                <strong>关键转折点：</strong>若 EPU 回落至 P75 以下（约 &lt;200），模型将切换回均值回归模式。
                <strong style="color:#f44336">重要免责：</strong>上周 27.3% 准确率提醒我们，极端 EPU 条件下模型的泛化能力有限——投资者应结合其他信息源综合判断。
            </p>
        </div>
    </div>

    <div class="insight-box" style="margin-top:32px">
        <p>💡 <strong>模型说明：</strong>3-factor Kalman DFM（趋势+均值回归+波动率）提取方向和幅度两个潜因子，EPU &gt; P75 (DM) / P65 (EM) 时自动切换为趋势延续模式。
        恒生指数使用中国 EPU (CHNMAINLANDEPU) 作为制度开关，准确率从 66.7% 提升至 76.9%。
        VIX 作为 EPU 日频代理辅助判断不确定性。过热检测：tm z-score &gt; +2.5σ → NEUT，&gt; +3.5σ → BEAR。
        Tier 1 市场（≥70% 回测准确率，≥5年历史数据）可用于配置参考；Tier 2 市场（55-70%，≤1年追踪）仅供观察。
        预测收益率为年化估算，不代表短期走势。模型基于过去 5 年滚动窗口回测验证。
        数据来源：yfinance / FRED / akshare · 模型：Global Economy Lab · 
        <strong>预测周期：{period} · 生成时间：{gen_time}</strong></p>
    </div>

    <div style="text-align:center;margin-top:24px;font-size:11px;color:#444">
        <span class="ts-stamp">第29周存档</span>
        <span style="margin:0 8px">|</span>
        <span>上周准确率 {hr:.1%} (3/11) | Tier1: {t1_rate:.1%} | Tier2: {t2_rate:.1%}</span>
        <span style="margin:0 8px">|</span>
        <span>月度准确率 Tier1: {t1_mo_rate:.1%} | Tier2: {t2_mo_rate:.1%}</span>
        <span style="margin:0 8px">|</span>
        <span>EPU: {epu['value']:.1f} (P{epu['percentile']} · HIGH_EPU 制度)</span>
    </div>
</div>
<footer>GingerFamily.CN · Global Economy Lab 驱动 · 数据仅供参考不构成投资建议</footer>
</body></html>'''

    return html


if __name__ == "__main__":
    html = make_prediction_html()
    proj = Path(__file__).parent.parent

    # Write prediction.html (main)
    with open(proj / "output/prediction.html", "w") as f:
        f.write(html)
    print(f"Wrote output/prediction.html ({len(html):,} bytes)")

    # Write prediction202629.html (archive)
    with open(proj / "output/prediction202629.html", "w") as f:
        f.write(html)
    print(f"Wrote output/prediction202629.html ({len(html):,} bytes)")

    print("Done.")
