# Nocturne · 夜谱

> 从滚动视频谱到可校对、可播放的在线六线 TAB 草稿。

![Status](https://img.shields.io/badge/status-alpha-8b82ff)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![Node.js](https://img.shields.io/badge/Node.js-22%2B-339933)

Nocturne 是一个本地优先的音乐练习工作台。它把 Bilibili 视频中的固定区域切成带时间证据的谱面帧，生成可阅读 PDF，并将清晰的六线 TAB 转成可以继续试听、校正和保存的 MusicXML 草稿。

当前版本适合个人研究、练习和限定模板验证，不承诺把任意视频一键转换成出版级乐谱。识别结果必须结合原帧和原声人工校对。

完整的产品与算法构想见 [PRODUCT_IDEA.md](./PRODUCT_IDEA.md)，识别边界见 [RECOGNITION.md](./RECOGNITION.md)。

## 能做什么

- 注册、登录和 30 天 HttpOnly 会话；
- 不依赖视频直接新建 8 小节空白六线谱，并继续追加小节；
- 输入 Bilibili 完整链接、BV 号或 AV 号建立私人项目；
- 通过 `yt-dlp` 自动读取公开视频元数据，将视频封面缓存到私人项目并显示在首页曲库；
- 在用户确认有权处理后下载视频；
- 手动选择视频起止时间、拖拽框选谱面区域并设置抽帧间隔；
- 用 FFmpeg 裁剪候选帧，记录视频秒数和源帧编号；
- 自动裁掉空白、跳过相邻重复画面并生成预览 PDF；
- 用 OpenCV + Tesseract 识别固定位置、清晰数字排版的六线 TAB；
- 对重复出现的小节投票，每小节保留质量最佳的来源裁图并重排 PDF；
- 导出并保存可播放的 MusicXML 草稿；
- 在同一个编辑工作台导出完整 PDF、Guitar Pro 7 `.gp`、标准 MIDI 和合成 WAV；
- 用 alphaTab 渲染谱面，用 FluidSynth + SoundFont 合成试听；
- 直接在渲染谱面上单选、连续多选或离散多选音符，批量修改品位、换弦和常用技巧；
- 用 `Space` 统一播放/暂停，以 `Ctrl/Cmd+E` 打开选区命令面板，并支持撤销和保存快捷键；
- 把原视频或当前小节原帧放进可拖动、可放大的非模态参考窗，按同步点跟随谱面播放；
- 在六线网格中让每根弦、每个十六分位置独立获得光标，用数字键或触控输入 `0–36` 品；
- 同一时间格逐弦组成和弦，单独删除当前弦，并用 `Ctrl/Cmd+Z`、`Ctrl/Cmd+Shift+Z` 撤销或重做；
- 手动添加连音、滑音、击弦、勾弦、推弦、颤音、泛音、闷音、延音和死音；
- 分析视频原声或上传音频的速度、起音和 A/B/C 段落候选；
- 保存音频时间、小节号和谱面位置同步点，并在练习时插值跟随滚动；
- 分别调节原声/上传音频和识别谱合成音的音量；
- 使用 SQLite 保存用户、项目、文件、识别诊断和同步数据。

## 目前做不到什么

- 从混合音乐中可靠分离并还原完整吉他、贝斯、钢琴、鼓和人声谱；
- 自动判断所有推弦、击勾弦、滑音、泛音和复杂节奏；
- 像 Guitar Pro 一样编辑变拍号、三连音、多声部和任意复杂结构；
- 直接生成 MP3、OGG 或 Musepack `.mpc` 压缩音频（当前先导出标准 PCM WAV）；
- 把音频段落自动命名为确定的主歌、副歌或桥段；
- 提供公开社区、评论、收藏、审核和版权投诉系统；
- 替代 Guitar Pro、MuseScore 或专业人工制谱。

## 工作流程

```text
Bilibili 链接 / BV / AV
        ↓
读取标题、作者、时长和封面
        ↓
用户确认处理权限并获取视频
        ↓
选择时间范围 + 拖拽谱面 ROI + 设置抽帧间隔
        ↓
候选帧 + 秒数/源帧证据 + 预览 PDF
        ↓
六线 TAB 识别 + 重复小节投票 + 重排 PDF
        ↓
MusicXML + alphaTab/FluidSynth 编辑与试听
        ↓
谱面多选、原帧/视频对照、键盘改品位、批量补技巧
        ↓
音频分析、同步点和滚动跟练
```

## 技术结构

```text
React 19 + Vite + TypeScript
├─ 登录与私人曲库
├─ 视频选段、ROI 框选和候选帧
├─ alphaTab 乐谱渲染、音符命中边界、光标和滚动
├─ FluidSynth WebAssembly + SoundFont 播放
├─ GP8 风格选区、快捷命令和可从空白开始的六线网格编辑器
└─ 音频混音、分析与同步点

FastAPI + SQLite
├─ Scrypt 密码哈希和 HttpOnly 会话
├─ 用户隔离的项目与文件接口
├─ yt-dlp Bilibili 适配器
├─ FFprobe/FFmpeg 视频和音频处理
├─ Pillow 多图与小节裁图 PDF
├─ OpenCV + Tesseract 六线 TAB 识别
├─ NumPy 轻量节拍、起音和段落分析
└─ 可选 Audiveris 五线谱 PDF 识别
```

## 环境要求

- Node.js 22 或更高版本；
- Python 3.11 或更高版本；
- FFmpeg 与 FFprobe；
- Tesseract OCR；
- 可选：Audiveris，用于清晰印刷五线谱 PDF。

开始前确认这些命令可以执行：

```text
node --version
npm --version
python --version
ffmpeg -version
ffprobe -version
tesseract --version
```

## 快速开始

### Windows PowerShell

```powershell
git clone https://github.com/lysyminger/nocturne-score-workspace.git
cd nocturne-score-workspace

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

npm ci
npm run dev
```

打开 <http://localhost:5173>。开发服务器会把 `/api` 代理到 <http://127.0.0.1:8765>。

### macOS / Linux

```bash
git clone https://github.com/lysyminger/nocturne-score-workspace.git
cd nocturne-score-workspace

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

npm ci
npm run dev
```

如果系统的 Python 命令不是 `python`，请在激活虚拟环境后确认 `python --version` 指向 `.venv`。

## 生产构建

```bash
npm run build
npm start
```

打开 <http://127.0.0.1:8765>。FastAPI 会在 API 路由之后托管 `dist/` 中的前端文件。

Ubuntu/systemd、HTTPS 反向代理、更新和回滚步骤见 [DEPLOYMENT.md](./DEPLOYMENT.md)。不要把开发端口直接暴露到公网。

## 配置

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `APP_DATA_DIR` | 仓库下的 `data/` | SQLite、上传文件、下载视频和识别输出目录 |
| `APP_SECURE_COOKIES` | 未启用 | HTTPS 部署时设置为 `1`，让会话 Cookie 只通过 HTTPS 发送 |
| `TESSERACT_BIN` | 自动搜索 PATH | 指定 Tesseract 可执行文件 |
| `AUDIVERIS_BIN` | 自动搜索 PATH | 指定可选的 Audiveris 可执行文件 |

`data/`、`.env`、虚拟环境、依赖、构建结果和本地部署包均被 Git 忽略。

## 音色与移动端播放

alphaTab 负责解析、排版、光标和播放事件；`js-synthesizer` 封装的 FluidSynth 负责 SoundFont/MIDI 合成。

- HTTPS 或 localhost 会优先使用 AudioWorklet；
- 普通 HTTP 会退回较大的 ScriptProcessor 缓冲；
- iPhone/iPad 等移动浏览器对 Web Audio 用户手势和安全来源要求更严格；
- 正式部署建议使用可信 HTTPS，并在用户首次点击播放时解锁音频；
- 如果只需要最稳定的移动播放，可在后续增加服务端预渲染 M4A/MP3 作为备用。

构建时，alphaTab Vite 插件会从固定版本依赖中提供 Bravura 字体和 SONiVOX SoundFont；`scripts/copy-audio-engine.mjs` 会复制 FluidSynth 浏览器运行时及其许可证。生成的二进制和运行文件不进入源码提交。

进入“编辑与播放”后，PDF、GP7、MIDI 和 WAV 都从谱面上方的统一“导出”菜单获取。WAV 在浏览器内离线合成，不会把乐谱上传给第三方服务。为避免手机或平板内存耗尽，单次 WAV 导出目前限制为 6 分钟。

## 谱面直接编辑与快捷键

这里借鉴 Guitar Pro 8 的“先建立选区，再对选区执行命令”逻辑，但没有把 alphaTab 描述成完整制谱器。alphaTab 负责解析、音符命中检测、渲染和播放；选区、命令事务、撤销、保存与识别结果映射由本项目实现。

| 操作 | 结果 |
| --- | --- |
| 单击音符 | 只选择当前音符 |
| `Ctrl/Cmd + 单击` | 添加或移除任意离散音符 |
| `Shift + 单击` | 在同一谱表、同一声部内从锚点连续选择 |
| 拖过音符 | 扩展连续选区 |
| 顶部“每行 3 / 4 小节” | 固定谱面每一行的小节数，并记住本机选择 |
| 顶部 `1/1–1/16` | 把所选节拍直接改成指定时值 |
| `+` / `=`、`-` / `_` | 缩短、延长所选节拍，方向与 Guitar Pro 8 一致 |
| `←/→`、`↑/↓` | 在相邻节拍或当前和弦的音之间移动选择 |
| `Shift + 方向键` | 扩展连续选择范围 |
| `0–9` | 给全部所选六线谱音符输入品位，支持两位数 |
| `Alt + ↑/↓` | 在相邻弦保持音高换把位；自然泛音会安全跳过 |
| `B/V/N/M/I/X` | 推弦、颤音、泛音、闷音、延音、死音 |
| `L/S/H/P` | 对同一弦上的两个音应用连音、滑音、击弦、勾弦 |
| `Delete` | 删除所选音符 |
| `Ctrl/Cmd + E` | 打开选区命令面板 |
| `Ctrl/Cmd + Z` | 撤销最近修改 |
| `Ctrl/Cmd + S` | 保存当前校对版本 |
| `Space` | 播放或暂停 |

识别生成的谱会把受影响小节写回识别诊断并重新生成 MusicXML；用户导入的结构化谱则保存为兼容性更明确的 GP7 `.gp`。当前不生成或伪装成 GP8 原生文件。浮动视频同步至少需要两个“小节—视频秒数”同步点；没有同步点时仍可自由播放视频或查看大图。

六线网格编辑器使用独立光标：`↑/↓` 只切换 1～6 弦，`←/→` 每次移动一个十六分位置，`0–9` 只写入当前弦，`Delete/Backspace` 也只删除当前弦。同一列可逐弦组成和弦；在已有延音内部输入时，会在游标位置切开前一个节拍。手机和平板可用可见的品位输入框与 44px 方向键完成相同操作。

技巧键采用本项目容易记忆的分拆映射，并非逐键复制 Guitar Pro 8。识别草稿当前每个音符只能保存一个 `technique`，所以对同一音符连续应用多个效果时以后一次为准；导入的结构化谱不受这个单字段限制。统一菜单中的 PDF 是原视频谱图裁切合成版，数值校正会更新可播放谱，但不会改写已经生成的像素 PDF。

识别结果和手动空白谱共用逐小节的十六分网格：鼠标拖动可选择连续时值，弦按钮可指定输入弦，数字键可直接打品位，`R` 把选区改为休止，`Alt + ←/→` 移动事件，`Alt + ↑/↓` 换弦。识别项目点击“重试本小节识别”只生成未保存提案，不会直接覆盖已经校正的版本。

交互参考：[Guitar Pro 8 用户手册](https://support.guitar-pro.com/hc/en-us/articles/5018404823069-GP8-Guitar-Pro-8-User-Guide)、[Guitar Pro 8 快捷键](https://support.guitar-pro.com/hc/en-us/articles/360001646978-GP8-List-of-keyboard-shortcuts)、[Guitar Pro 命令面板](https://www.guitar-pro.com/blog/p/54190-unlock-guitar-pro-tips-you-should-know)、[Guitar Pro 8 Audio Track](https://support.guitar-pro.com/hc/en-us/articles/7460696563357-GP8-How-to-use-the-Audio-Track)、[alphaTab 音符事件](https://alphatab.net/docs/reference/api/notemousedown)、[alphaTab 音视频同步](https://alphatab.net/docs/guides/audio-video-sync)。

## 乐谱识别边界

视频 TAB 路线当前假设：

- 谱面区域基本固定；
- 画面存在六根清晰水平谱线；支持浅底深色谱、黑色/半透明底上的浅色谱，以及上五线谱、下六线谱的固定联合布局；
- 品位数字清晰，透视、旋转和遮挡较少；
- 默认标准六弦调弦和 4/4 拍；自动节奏会尝试区分单梁八分与双梁十六分，校对器统一使用十六分网格；
- 休止符主要根据相邻事件之间的空隙推导，复杂连休止、三连音和被遮挡的符干仍需要人工校正；
- 重复小节可用于投票纠错，而不是直接全部删除。

联合布局目前由下方六线谱提供弦与品位，上方五线谱用于小节号、布局判断和完整 PDF 裁切；复杂五线谱节奏仍要在十六分网格中校正。识别结果会保存覆盖置信度、数字覆盖率、布局/极性、缺失小节、来源帧和解析警告。没有可靠小节号时不会再自动从 0 开始编号，系统也不会把低覆盖结果包装成确定的成品谱。

印刷五线谱使用可选 Audiveris：

```powershell
$env:AUDIVERIS_BIN = 'D:\Tools\Audiveris\bin\Audiveris.bat'
npm run dev
```

Audiveris 同样需要人工校正，并不负责视频六线 TAB 的专用识别流程。

## 测试

激活 Python 虚拟环境后运行：

```bash
npm run test:api
npm run build
npm run check
```

也可以上传 [examples/demo-guitar.musicxml](./examples/demo-guitar.musicxml)，验证谱面渲染、FluidSynth 播放和光标跟随。

## 数据、安全与内容权利

- 所有项目文件都通过所属用户校验后读取，不直接公开挂载存储目录；
- 密码使用随机盐 Scrypt 哈希，不保存明文；
- 会话 Cookie 为 HttpOnly、SameSite=Lax；
- 视频下载前必须由用户确认其有权处理内容；
- 适配器不读取浏览器 Cookie，不绕过私人、付费或权限受限内容；
- 免费或非商业用途不代表自动拥有复制、改编或传播权；
- 当前后台任务运行在 Web 进程内，仅适合本地或小规模原型；
- 公网部署前仍需补充限流、CSRF、邮件验证、找回密码、对象存储、任务队列、备份和内容治理。

请勿把真实数据库、会话、视频、音频或他人受版权保护的谱面提交到公开仓库。

## 第三方组件与许可证

第三方组件继续遵循各自许可证，概要见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

本仓库当前只是“源码公开”，尚未声明覆盖原创代码的项目级开源许可证。在添加明确许可证前，默认版权规则仍然适用；如需复制、修改或再分发原创部分，请先联系仓库所有者确认授权。

## 参与项目

欢迎通过 Issue 提交可复现的错误、浏览器兼容性信息和已获授权的识别失败样例。涉及大范围架构、识别模型或社区功能的改动，建议先开 Issue 讨论边界，再提交 Pull Request。
