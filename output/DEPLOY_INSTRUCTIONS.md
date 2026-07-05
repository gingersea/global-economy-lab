请执行以下部署任务：

## 任务：将全球市场预测文件部署到 gingerfamily.cn

### 背景
- gingerfamily.cn 服务器：106.54.5.160，通过 FRP (frps 在 gingerfamily 上) 管理
- 网站文件路径：/var/www/gingerfamily/ 或 nginx 默认的网站根目录
- 需要先找到 SSH 访问方式（可能通过 FRP 隧道或其他方式）

### 需要部署的文件

1. **`/data/global-economy-lab/output/weekly_prediction.html`**
   - 上传到 gingerfamily.cn 的网站根目录，重命名为 `prediction.html`
   - 这是完整的全球市场预测页面

2. **`/data/global-economy-lab/output/market_prediction_section.html`**
   - 该文件是一个 HTML 片段，需要插入到 `market.html` 中
   - 插入位置：在 `market.html` 的 insight-box div 之后
   - 如果 market.html 中已有旧的预测 section，先替换掉

### 验证
- 部署完成后，访问 https://gingerfamily.cn/prediction.html 应该能看到预测页面
- market.html 底部应该有预测卡片

### 注意
- 所有预测都已标记时间戳（2026-06-01 06:34 CST）
- 每个市场行都有数据日期标签（2026-05-29）
- 预测数据来源：Global Economy Lab 2-factor 模型
