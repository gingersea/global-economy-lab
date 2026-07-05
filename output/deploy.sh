#!/bin/bash
# Deploy prediction files to gingerfamily.cn nginx
# Run this ON gingerfamily.cn server
set -e

NGINX_ROOT="/var/www/gingerfamily"
# Auto-detect nginx root
if [ ! -d "$NGINX_ROOT" ]; then
    NGINX_ROOT=$(grep -r "root" /etc/nginx/sites-enabled/ 2>/dev/null | grep -v "#" | awk '{print $2}' | sed 's/;//' | head -1)
fi
if [ -z "$NGINX_ROOT" ] || [ ! -d "$NGINX_ROOT" ]; then
    NGINX_ROOT="/var/www/html"
fi
echo "Nginx root: $NGINX_ROOT"

FRP_SRC="http://127.0.0.1:8084"

# 1. Deploy prediction.html
echo "Downloading prediction.html..."
wget -q -O "$NGINX_ROOT/prediction.html" "$FRP_SRC/weekly_prediction.html"
echo "  ✓ prediction.html deployed ($(wc -c < "$NGINX_ROOT/prediction.html") bytes)"

# 2. Download the section snippet
echo "Downloading market section..."
wget -q -O /tmp/market_section.html "$FRP_SRC/market_prediction_section.html"

# 3. Insert into market.html
MARKET_HTML="$NGINX_ROOT/market.html"
if [ -f "$MARKET_HTML" ]; then
    # Backup
    cp "$MARKET_HTML" "$MARKET_HTML.bak.$(date +%Y%m%d_%H%M%S)"
    
    # Remove old prediction section if exists (between prediction markers or at the end)
    sed -i '/<!-- 全球市场预测 Section/,/<\/div>$/d' "$MARKET_HTML" 2>/dev/null || true
    
    # Insert before closing </div> of the container
    SECTION=$(cat /tmp/market_section.html)
    # Insert before the last </div> before </div> before footer
    python3 -c "
import re
with open('$MARKET_HTML', 'r') as f:
    html = f.read()
# Insert before the closing container div and footer
insert_pos = html.rfind('</div>\n</div>\n<footer>')
if insert_pos == -1:
    insert_pos = html.rfind('</div>\n<footer>')
if insert_pos == -1:
    insert_pos = html.rfind('<footer>')
if insert_pos > 0:
    html = html[:insert_pos] + '$SECTION' + '\n' + html[insert_pos:]
with open('$MARKET_HTML', 'w') as f:
    f.write(html)
print('  ✓ market.html updated')
"
    echo "  ✓ market.html updated"
else
    echo "  ⚠ market.html not found at $MARKET_HTML"
fi

# 4. Verify
echo ""
echo "=== Deployment Complete ==="
echo "Verify: https://gingerfamily.cn/prediction.html"
echo "Verify: https://gingerfamily.cn/market.html"
ls -la "$NGINX_ROOT"/prediction.html "$NGINX_ROOT"/market.html

# 5. Reload nginx
nginx -t && systemctl reload nginx && echo "  ✓ nginx reloaded"
