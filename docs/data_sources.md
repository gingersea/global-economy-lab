# 数据源清单 (Data Source Catalog)

本项目所有数据源均**不强依赖任何 API Key**。下表按 8 大类目整理免费数据源、接入方式与历史长度。

> **总原则**：FRED 公共 CSV 端点 (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`) 完全免费、不需要 Key，可替代 `fredapi` 几乎所有调用。`MacroEconomicFetcher` 与 `BondsFetcher` 默认走 `fred_csv` 后端，并将 `fredapi` 作为可选 fallback。

---

## 1. 美国宏观 (Macro / US)

| 指标 | FRED Series | 主后端 | Fallback | 频率 | 起始 |
|---|---|---|---|---|---|
| 实际 GDP | `GDPC1` | `fred_csv` | `fredapi` | 季度 | 1947 |
| CPI | `CPIAUCSL` | `fred_csv` | `fredapi` | 月度 | 1947 |
| 核心 CPI | `CPILFESL` | `fred_csv` | `fredapi` | 月度 | 1957 |
| 失业率 | `UNRATE` | `fred_csv` | — | 月度 | 1948 |
| 工业产出 | `INDPRO` | `fred_csv` | — | 月度 | 1919 |
| ISM 制造业 PMI | `NAPM` | `fred_csv` | `fredapi` | 月度 | 1948 |
| 联邦基金利率 | `FEDFUNDS` | `fred_csv` | `fredapi` | 月度 | 1954 |
| NFCI 金融条件 | `NFCI` | `fred_csv` | — | 周度 | 1971 |

辅助来源：BLS Public Data API（无需 Key，匿名限频 25 次/天）、BEA、Federal Reserve H.15。

## 2. 美国债券 / 利率 (Bond / US)

| 指标 | 主后端 | Fallback | 频率 | 起始 |
|---|---|---|---|---|
| 美债 2Y (DGS2) | `fred_csv` | `treasury_gov`, `fredapi` | 日 | 1976 |
| 美债 10Y (DGS10) | `fred_csv` | `treasury_gov`, `fredapi` | 日 | 1962 |
| 美债 30Y (DGS30) | `fred_csv` | `treasury_gov`, `fredapi` | 日 | 1977 |

`treasury_gov` 后端：[美国财政部 Daily Treasury Par Yield Curve CSV](https://home.treasury.gov/resource-center/data-chart-center/interest-rates) (1990 至今，公开)。

## 3. 中国宏观 (Macro / CN)

| 指标 | 后端 | 备注 |
|---|---|---|
| 中国 CPI YoY | `akshare` (`ak.macro_china_cpi_yearly`) | 已封装 |
| 中国 PPI / GDP / M2 / 社融 / 利率 | `akshare` | 扩展即可 |
| 备份 | 国家统计局 / 东方财富 / 同花顺（均经由 akshare） | — |

## 4. 全球宏观 (Macro / Global, EU, JP, …)

| 指标 | 后端 | URL | Key |
|---|---|---|---|
| 欧元区 HICP | `ecb_sdw` (`ICP/M.U2.N.000000.4.ANR`) | https://sdw-wsrest.ecb.europa.eu/ | 无 |
| 欧元区国债收益率 | `ecb_sdw` (`YC/B.U2.EUR…`) | 同上 | 无 |
| 全球 GDP (US$) | `worldbank` (`WLD:NY.GDP.MKTP.CD`) | https://api.worldbank.org/v2/ | 无 |
| OECD 综合领先指标 | `oecd_sdmx`（计划） | https://stats.oecd.org/SDMX-JSON/ | 无 |
| IMF WEO / IFS | IMF SDMX（计划） | https://dataservices.imf.org/ | 无 |

## 5. 股票 / 指数 / ETF (Equity)

| 指标 | 主后端 | Fallback |
|---|---|---|
| S&P 500 / NASDAQ-100 | `yfinance` | `stooq` |
| 沪深 300 / 中证 500 | `yfinance`, `akshare` | — |
| 恒生 / 国企 | `yfinance` | `stooq` |
| 日经 225 / 欧股 | `yfinance` | `stooq` |

## 6. 商品 (Commodity)

| 指标 | 主后端 | Fallback |
|---|---|---|
| 黄金 / 白银 | `yfinance` | `stooq`, LBMA 公开 CSV |
| WTI / Brent 原油 | `yfinance` | EIA Open Data CSV |
| 综合商品指数 | `yfinance` | World Bank Pink Sheet |

## 7. 汇率 (FX)

| 指标 | 主后端 | Fallback |
|---|---|---|
| DXY | `yfinance` | `stooq` |
| EUR/USD | `yfinance` | `ecb_sdw` (`EXR/D.USD.EUR.SP00.A`), `stooq` |
| USD/CNY | `yfinance` | `akshare`（外管局） |

## 8. 情绪 / 风险 (Sentiment / Risk)

| 指标 | 主后端 | 备注 |
|---|---|---|
| VIX / MOVE / SKEW | `yfinance` | — |
| NFCI 金融条件 | `fred_csv` | 见 §1 |
| AAII 散户情绪 | 计划：AAII 公开 CSV | 每周 |
| GDELT 事件/情绪 | 计划：GDELT CSV | 全球 |

---

## 后端目录 (`src/data_fetcher/backends/`)

| 模块 | 端点 | 是否需要 Key |
|---|---|---|
| `fred_csv.py` | `fred.stlouisfed.org/graph/fredgraph.csv` | 否 |
| `treasury_gov.py` | `home.treasury.gov/.../daily-treasury-rates.csv` | 否 |
| `worldbank.py` | `api.worldbank.org/v2/` | 否 |
| `ecb_sdw.py` | `sdw-wsrest.ecb.europa.eu` | 否 |
| `stooq.py` | `stooq.com/q/d/l/` | 否 |

所有后端共享 `_http.py` 中的工具：浏览器风格 User-Agent、`tenacity` 指数退避、每主机限流。

## 数据落地结构

```
data/
├── raw/                       # 原始下载（按后端组织，可选）
├── processed/<fetcher>_<hash>.parquet   # 标准缓存
├── snapshots/<YYYYMMDD>/      # 不可变快照（复现性）
└── _meta/manifest.jsonl       # 每次抓取的元数据（来源/行数/md5/时间）
```

## 许可证 / 引用

- **FRED**：Federal Reserve Bank of St. Louis 数据，使用前请阅读其 Terms of Use。
- **US Treasury**：公共领域 (public domain)。
- **ECB SDW**：可自由再分发，需注明来源。
- **World Bank**：CC-BY 4.0。
- **Stooq**：免费个人/研究用途。
- **akshare** / **yfinance**：开源 Python 库，分别从公开网页与 Yahoo Finance 公开 API 拉取。
