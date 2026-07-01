## Usage

#### 1) Compiling LibMR

Compile LibMR and python interface to LibMR using following commands.
For pythong interfaces to work, you would require Cython to be pre-installed
on your machine
```bash
cd /home/u2023311329/jupyterlab/CROSR-main/libMR/
# Spider-Sense v2 — 快速上手

这是 Spider-Sense v2（基于 CROSR 思路改造的网络入侵检测系统）的代码仓库副本，已精简为可直接运行的桌面/本地服务形态。

简短说明：后端为 Flask API，前端为 React 单页应用；检测引擎使用 DHRNet-1D + Weibull/OpenMax（纯 Python 实现），并支持可选的 LLM Agent、RAG 和 XAI。

重要：仓库中保留了模型目录（`models/`）和处理后的数据目录（`processed_*/`），这些可能很大，建议不要直接推送到远程仓库（可用 release 或外部存储）。

快速启动

1) 创建并激活环境（推荐 Conda）

```bash
conda create -n openmax python=3.10 -y
conda activate openmax
pip install -r requirements.txt
```

或使用 venv：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) （仅在修改前端时）构建前端：

```bash
cd web-frontend
npm install
npm run build
cd ..
```

3) 启动服务（开发）：

```bash
python main.py --no-browser
# Spider-Sense v2 — 项目简介

简要：Spider-Sense v2 是基于 CROSR 思路的开放集网络入侵检测系统（DHRNet-1D + Weibull/OpenMax），扩展了多智能体辩论、RAG 知识检索与 XAI 可解释模块。后端为 Flask API，前端为 React 单页应用，项目设计为本地服务/桌面可执行原型。

状态：核心后端与前端已实现，可在本地运行。LLM Agent 为可选功能，需配置第三方兼容 OpenAI Chat API 的服务与 API Key。

快速启动（最小步骤）

```bash
conda create -n openmax python=3.10 -y
conda activate openmax
pip install -r requirements.txt
python main.py --no-browser
# 打开 http://127.0.0.1:5000
```

如需修改前端：

```bash
cd web-frontend
npm install
npm run build
cd ..
```

LLM 模式（可选）

在启动时传入 API Key，或在前端 `LLM 配置` 面板填写：

```bash
python main.py --llm-api-key "YOUR_KEY" --llm-model "deepseek-v4-flash" --llm-api-base "https://api.deepseek.com/v1"
```

项目结构（重要目录）

```
CROSR-main/
├─ main.py
├─ backend/                # Flask API 与业务逻辑
├─ web-frontend/           # React 前端
├─ models/                 # 本地模型权重（请不要直接提交到 Git）
│  ├─ weibull_om/
│  │  ├─ model.pth
│  │  └─ detector.pkl
│  └─ (Transformer 检测器已移除)
├─ knowledge/              # RAG 知识库（攻击模式与 MITRE）
├─ processed_cicids/       # 处理后的数据样本
└─ requirements.txt
```

可对外提供的模型与数据（建议通过 release 或外部存储发送）

- `models/weibull_om/model.pth` —— DHRNet-1D 权重（用于推理）
- `models/weibull_om/detector.pkl` —— OpenMax/Weibull 拟合器
`save_models/*/best.pth` 或 `save_models/cicids_1d/best.pth` —— 训练检查点（按数据集选取）
- `save_models/*/best.pth` 或 `save_models/cicids_1d/best.pth` —— 训练检查点（按数据集选取）
- 若需要重建 RAG/实验数据：`saved_features/`、`saved_MAVs/`、`saved_distance_scores/` 可打包提供

说明：这些文件通常较大，不建议纳入源码仓库。建议通过 GitHub Release、共享驱动或配置 Git LFS 传输。

日志与排错要点

- 健康检查：GET `/api/health`
- 常见问题：LLM 调用失败通常为 API Key 或权限问题，后端会在日志输出 `[LLM DEBUG]` 详情。
- 打包注意：生成 Windows `.exe` 必须在 Windows 环境用 PyInstaller 构建。

联系与后续

如需我为接手人打包一份包含模型的 release（或生成 Git LFS 配置），我可以准备上传脚本与说明。