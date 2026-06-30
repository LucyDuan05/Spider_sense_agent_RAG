# Spider-Sense v2 — 当前状态说明

## 1. 当前项目进度

本项目目前已经完成以下关键工作：

- `backend/` 核心后端已构建完毕，包含 Flask API、检测引擎、RAG、XAI、Agent 逻辑。
- `web-frontend/` React 前端已经实现可配置界面，包括 LLM 配置、实时检测、RAG/XAI/Agent 分析、捕获注入演示。
- `main.py` 已经可作为统一入口启动本地服务。
- LLM 运行时配置已支持：通过命令行参数或前端配置页面动态设置 `LLM_API_KEY`、`LLM_MODEL`、`LLM_API_BASE`。
- 已经清理掉大量历史遗留文件，保留当前可运行的项目核心资源。

## 2. 运行方式

### 2.1 环境准备

```bash
cd /home/mihu/CROSR-main
conda activate openmax
```

### 2.2 构建前端（仅在前端修改后需要）

```bash
cd web-frontend
npm install
npm run build
cd ..
```

### 2.3 启动服务

```bash
python main.py
```

如果不希望自动打开浏览器：

```bash
python main.py --no-browser
```

指定端口：

```bash
python main.py --port 5001
```

### 2.4 访问界面

打开浏览器访问：

```text
http://127.0.0.1:5000
```

### 2.5 启用 LLM 模式

可在命令行传入：

```bash
python main.py --llm-api-key "YOUR_KEY" --llm-model "gpt-4" --llm-api-base "https://api.openai.com/v1"
```

也可在前端 `LLM 配置` 面板里填写并保存。

## 3. 关键模块与原理

### 3.1 检测引擎

- `backend/crosr_engine.py`：封装 DHRNet-1D 模型加载与预测。
- `backend/weibull_openmax.py`：纯 Python 实现 Weibull 拟合与 OpenMax 开放集识别，替代原来的 `libMR`。
- `backend/detection_engine.py`：将 DHRNet 重建误差 / 特征嵌入转换为未知样本判定。

### 3.2 Orchestrator

- `backend/orchestrator.py`：协调整个检测流程。
- 处理流程：特征预测 → RAG 检索（异常时）→ XAI 解释（异常时）→ Agent 辩论（异常时）。
- 最终结果会组合检测、RAG、XAI、Agent 分析内容。

### 3.3 多智能体 Agent

- `backend/agent_layer.py`：包含规则型 Agent 和 LLM Agent。
- 规则型 Agent 使用内置规则对检测结果、RAG 匹配、XAI 归因做推理。
- LLM Agent 通过 OpenAI 兼容的 `chat/completions` 接口进行自然语言分析。

### 3.4 RAG 知识增强

- `backend/rag_engine.py`：加载 `knowledge/attack_patterns.json`、`knowledge/mitre_attack.json`。
- 根据检测结果生成的特征向量进行向量检索，返回最相似的攻击模式与攻击技术描述。

### 3.5 XAI 解释

- `backend/xai_engine.py`：对主模型输入进行特征重要性评估，输出关键特征和解释信息。
- 该模块帮助将模型预测结果转换为可解释的特征驱动说明。

### 3.6 前端

- `web-frontend/src/App.tsx`：主 UI 控制逻辑。
- 前端提供以下功能：
  - 实时检测与事件流展示
  - LLM 配置面板
  - 捕获/注入演示
  - Agent / RAG / XAI 分析展示

## 4. 当前有效目录与文件

保留的核心文件和目录如下：

- `main.py`：系统入口。
- `backend/`：后端服务与业务逻辑。
- `web-frontend/`：React 前端应用。
- `models/weibull_om/`：运行时模型权重与 OpenMax 检测器。
- `knowledge/`：RAG 攻击模式与 MITRE 知识库。
- `processed_cicids/`：当前数据集处理结果，用于现有模型评估与运行。
- `requirements.txt`：Python 依赖。
- `PROJECT_INTRO.md`、`PROJECT_USAGE.md`：项目介绍与使用说明。

### 仍然保留的训练 / 数据准备脚本

这些文件属于当前项目可继续使用的模型训练与数据准备内容：

- `DHR_Net.py`, `DHR_Net_1D.py`
- `train_net.py`, `train_net_1d.py`
- `prepare_cicids_1d.py`, `prepare_cicids2018_1d.py`, `prepare_nslkdd_1d.py`, `prepare_unsw_nb15_1d.py`
- `get_model_features.py`, `get_model_features_1d.py`
- `eval_weibull.py`, `eval_dhrnet.py`, `eval_openset.py`

## 5. 已清理的多余文件

为了保持项目可用且聚焦运行部分，已删除以下类别文件：

- `libMR/` 及与 Python 2 `libMR` 相关的旧代码。
- 历史文档：`README_CICIDS.md`、`PIPELINE.html`、`PIPELINE.md`、`project_context.md`、`TRAINING_GUIDE.md`、`timepass.md`。
- 旧 OpenMax 训练/评估流水线脚本：`compute_openmax.py`、`compute_distances.py`、`openmax_utils.py`、`MAV_Compute.py`、`run_openmax_pipeline.sh`、`run_postprocess.sh`、`scan_openmax_params.py`、`extract_best_auroc.py`、`fast_scan.py`、`check_openset.py`、`check_val_data.py`、`prepare_cifar10.py`。
- 旧 Python 环境配置文件：`environment-openmax-py2-exact.yml`、`environment-py27.yml`、`environment-py39.yml`。

## 6. 交接建议

### 6.1 先验证当前运行

- `conda activate openmax`
- `cd /home/mihu/CROSR-main`
- `python main.py --no-browser`
- 访问 `http://127.0.0.1:5000`

### 6.2 确认模型加载

- 观察后端启动日志是否显示 `CROSR Detection Engine loaded`
- 访问前端后检查 `LLM 配置`、`捕获/注入`、`Agent` 面板是否正常加载

### 6.3 后续工作重点

- 继续完成 `web-frontend` LLM 版 Agent 的稳定性与错误展现
- 在 Windows 上进行 PyInstaller 打包测试，生成 `spider-sense.exe`
- 如果需要扩展数据集，可继续用 `prepare_*` 和 `train_net_1d.py` 训练新的模型
- 若要开启真实抓包，需要补充网络抓包代码并将前端捕获按钮接入真实数据源

## 7. 安装与依赖

推荐方法（复制粘贴执行）：

使用 Conda 环境（推荐）：

```bash
conda create -n openmax python=3.10 -y
conda activate openmax
pip install -r requirements.txt
```

直接使用 pip（虚拟环境）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

可选：若使用 GPU 请安装适配的 `torch` 版本（见 https://pytorch.org/get-started/locally/）。

文件与大数据注意事项：

- 项目中 `models/`、`processed_*`、`saved_features/` 可能包含大文件，建议不要直接推送到代码仓库，而是通过 release 或外部存储管理。
- 若需要把部分模型包含在仓库，请先确认文件大小与团队策略。

## 8. Git 上线准备（建议）

我已准备了一个基础的 `.gitignore`，建议在新仓库初始化时使用。`.gitignore` 忽略常见构建产物、虚拟环境和大数据目录。

如果你希望我基于更严格的策略（例如排除或包含 `models/`），告诉我我会调整。
