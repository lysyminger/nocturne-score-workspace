# Nocturne 部署指南

本文给出一个通用的 Ubuntu + systemd + HTTPS 反向代理方案。示例使用：

- 应用用户：`nocturne`
- 应用根目录：`/opt/nocturne`
- 本地监听：`127.0.0.1:8765`
- 持久数据：`/opt/nocturne/shared/data`

请根据自己的服务器和域名调整，不要把数据库、上传内容或 `.env` 提交到 Git。

## 1. 准备系统依赖

安装 Node.js 22+、Python 3.11+、Git、FFmpeg/FFprobe 和 Tesseract OCR，并确认：

```bash
node --version
npm --version
python3 --version
git --version
ffmpeg -version
ffprobe -version
tesseract --version
```

Audiveris 只在需要识别清晰印刷五线谱 PDF 时安装。

## 2. 创建服务用户和目录

```bash
sudo useradd --system --create-home --home-dir /opt/nocturne --shell /usr/sbin/nologin nocturne
sudo mkdir -p /opt/nocturne/releases /opt/nocturne/shared/data
sudo chown -R nocturne:nocturne /opt/nocturne
```

## 3. 安装首个版本

```bash
sudo -u nocturne git clone https://github.com/lysyminger/nocturne-score-workspace.git /opt/nocturne/releases/initial
sudo -u nocturne python3 -m venv /opt/nocturne/venv
sudo -u nocturne /opt/nocturne/venv/bin/python -m pip install --upgrade pip
sudo -u nocturne /opt/nocturne/venv/bin/python -m pip install -r /opt/nocturne/releases/initial/backend/requirements.txt

cd /opt/nocturne/releases/initial
sudo -u nocturne npm ci
sudo -u nocturne npm run build

sudo -u nocturne ln -sfn /opt/nocturne/releases/initial /opt/nocturne/current
```

源码版本与 `shared/data` 分离，回滚代码时不会删除数据库和用户文件。

## 4. 安装 systemd 服务

```bash
sudo cp /opt/nocturne/current/deploy/nocturne.service /etc/systemd/system/nocturne.service
sudo systemctl daemon-reload
sudo systemctl enable --now nocturne.service
sudo systemctl status nocturne.service --no-pager
curl --fail http://127.0.0.1:8765/api/health
```

日志：

```bash
sudo journalctl -u nocturne -n 100 --no-pager
sudo journalctl -u nocturne -f
```

## 5. 配置可信 HTTPS

移动端 Web Audio、Secure Cookie 和公网登录都应使用 HTTPS。推荐让应用只监听 `127.0.0.1:8765`，再由 Caddy、Nginx 或其他反向代理终止 TLS。

Caddy 示例：

```caddyfile
score.example.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:8765
}
```

HTTPS 正常后启用 Secure Cookie：

```bash
sudo systemctl edit nocturne.service
```

写入：

```ini
[Service]
Environment=APP_SECURE_COOKIES=1
```

然后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart nocturne.service
```

不要在纯 HTTP 部署中设置 `APP_SECURE_COOKIES=1`，否则浏览器不会发送登录 Cookie。

如需指定工具路径，可在同一个 systemd override 中增加：

```ini
[Service]
Environment=TESSERACT_BIN=/usr/bin/tesseract
Environment=AUDIVERIS_BIN=/opt/audiveris/bin/Audiveris
Environment=NOCTURNE_VISION_MODEL=http://127.0.0.1:8892
```

`NOCTURNE_VISION_MODEL` 只应指向可信的内网或 SSH 隧道端点。本机 GPU 方案可运行 `training/start_server_linked_ocr.ps1` 建立仅绑定服务器回环地址的反向隧道；模型服务离线时，网页仍可使用传统 OCR。

## 6. 发布新版本

```bash
release=$(date -u +%Y%m%d-%H%M%S)
sudo -u nocturne git clone --depth 1 https://github.com/lysyminger/nocturne-score-workspace.git "/opt/nocturne/releases/$release"

cd "/opt/nocturne/releases/$release"
sudo -u nocturne npm ci
sudo -u nocturne npm run build
sudo -u nocturne /opt/nocturne/venv/bin/python -m pip install -r backend/requirements.txt
sudo -u nocturne /opt/nocturne/venv/bin/python -m pytest backend/tests -q

previous=$(readlink -f /opt/nocturne/current)
sudo -u nocturne ln -sfn "/opt/nocturne/releases/$release" /opt/nocturne/current
sudo systemctl restart nocturne.service
curl --fail http://127.0.0.1:8765/api/health
```

只有健康检查成功后才清理旧 release。不要删除 `/opt/nocturne/shared/data`。

## 7. 回滚

将 `current` 指回上一个已验证版本：

```bash
sudo -u nocturne ln -sfn /opt/nocturne/releases/<previous-release> /opt/nocturne/current
sudo systemctl restart nocturne.service
curl --fail http://127.0.0.1:8765/api/health
```

## 8. 备份

至少备份：

```text
/opt/nocturne/shared/data/nocturne.db
/opt/nocturne/shared/data/projects/
```

备份前应暂停写入或使用 SQLite 的在线备份方式，避免只复制到一半的数据库文件。

## 上公网前的边界

当前实现适合本地或小规模私人部署。公网开放前至少应增加：

- HTTPS、可信域名和安全响应头；
- 注册限制、邮件验证和找回密码；
- CSRF、速率限制、上传配额和病毒扫描；
- 独立任务队列、进程隔离与资源上限；
- 数据库和对象存储备份；
- 来源署名、用户授权、举报和下架流程；
- 日志脱敏、监控和安全更新流程。

不要直接把 `8765` 端口映射到公网。
