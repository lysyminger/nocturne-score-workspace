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

PaddleOCR 的训练配置与标签格式以 `training/vendor/PaddleOCR/doc/doc_ch/` 中的官方文档为准。正式训练前先把六线谱数字裁成小图，并分别建立训练集和验证集，不要直接用测试视频参与训练集。

## 数据边界

- 原视频、PDF、账号数据和私人项目文件只保存在本地 `training/data/`。
- Git 只保存脚本、类别规范和配置，不保存用户素材或模型权重。
- 训练集、验证集、测试集按歌曲划分，避免同一视频相邻帧泄漏到不同集合。
