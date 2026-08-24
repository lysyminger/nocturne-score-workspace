# 本地乐谱识别训练

这个目录专门使用本机 RTX 4060 训练，不影响网站后端的 `.venv`。

## 环境分工

- `.venv-paddle`：PaddlePaddle GPU，共同驱动 PaddleOCR 与 PaddleDetection。
- `vendor/PaddleOCR`：PaddleOCR 官方训练源码，安装脚本自动获取。
- `vendor/PaddleDetection`：PaddleDetection 2.9 官方训练源码，用于自训练乐谱符号检测器。
- `data/`：本地训练集，不提交 Git。
- `runs/`：训练日志与检查点，不提交 Git。
- `models/`：最终模型，不提交 Git。

## 安装与验证

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\training\install.ps1
powershell -ExecutionPolicy Bypass -File .\training\verify.ps1
```

## 符号检测数据格式

使用标准 COCO detection 格式：

```text
training/data/symbols/
├─ images/
│  ├─ train/
│  └─ val/
└─ annotations/
   ├─ train.json
   └─ val.json
```

标注框包括音头、符干、单梁/双梁、八分/十六分符尾、休止符、附点、连音弧、滑音线、推弦箭头和泛音标记。品位数字不要混进这个检测模型，由 PaddleOCR 单独识别。

开始训练：

```powershell
powershell -ExecutionPolicy Bypass -File .\training\train_detector.ps1 `
  -Epochs 40 `
  -BatchSize 2
```

RTX 4060 8GB 建议从 `BatchSize=2` 开始。脚本使用 PP-YOLOE+ CRN-S 并默认启用 AMP 混合精度，训练结果写入 `training/runs/symbol-detector/`。因为滑音方向有语义，配置中不会进行水平翻转。

## PaddleOCR 数据

先把私人 GP 文件渲染为图片和坐标真值，再生成品位数字 OCR 小图：

```powershell
node .\training\scripts\build_gp_corpus.mjs --source "D:\document\谱子" --profile tab
.\training\.venv-paddle\Scripts\python.exe .\training\scripts\extract_fret_crops.py
powershell -ExecutionPolicy Bypass -File .\training\train_fret_ocr.ps1 -Epochs 8 -BatchSize 256
```

续训、独立测试和导出网页/后端可加载的推理模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\training\train_fret_ocr.ps1 -Epochs 12 -BatchSize 256 -Resume
powershell -ExecutionPolicy Bypass -File .\training\evaluate_fret_ocr.ps1 -Split test
powershell -ExecutionPolicy Bypass -File .\training\export_fret_ocr.ps1
```

GP 的音符、品位、弦号、时值与技巧来自原始结构化数据；alphaTab 渲染坐标用于生成音符框。纯数字 OCR 会排除鬼音、延音目标、死音和泛音，避免裁图内容与标签不一致。PDF 没有这种真值，只渲染页面并自动切谱表：

```powershell
.\training\.venv-paddle\Scripts\python.exe .\training\scripts\build_pdf_corpus.py --source "D:\document\谱子"
```

PDF 切片属于弱标注，必须抽样复核后才能进入符号检测训练集。PaddleOCR 的原始训练说明仍以 `training/vendor/PaddleOCR/docs/` 中的官方文档为准。

## 服务器真实视频切片

服务器切片同步到 `training/data/real-video/` 后，先作为独立真实域测试/校对集，不要直接混入训练集：

```powershell
powershell -ExecutionPolicy Bypass -File .\training\prepare_real_video.ps1 -Stride 4
```

脚本会复用网站现有的六线谱定位逻辑，按完整品位 token（例如 `13`，不是拆开的 `1`、`3`）保存原始背景和归一化候选，并用已导出的品位模型给出建议值。打开 `training/data/real-video-token-review/review.html` 可分页核验；相同字体会先聚类，一次确认可传播到同簇候选。导出的 `verified-labels.tsv` 才能作为真实标签。预测值只是待确认建议，不能把高置信预测直接当真值，否则会把旧模型错误重新训练进去。

真实域划分必须按视频/BV 号进行：同一视频的不同时间、不同裁切范围和重复项目只能位于同一个 split。至少完整保留一个视频作为固定测试集，训练后才能知道黑底、透明叠加和五线谱+六线谱是否真正提升。

## 数据边界

- 原视频、PDF、账号数据和私人项目文件只保存在本地 `training/data/`。
- Git 只保存脚本、类别规范和配置，不保存用户素材或模型权重。
- 训练集、验证集、测试集按歌曲划分，避免同一视频相邻帧泄漏到不同集合。
