# Spider-Sense v2 — 多智能体协同开放集网络入侵检测系统

## 目标

将 CROSR (CVPR 2019) 重构为：**单个 .exe 可执行文件**，内置 Web UI，集成多智能体辩论研判 + RAG知识增强 + XAI可解释性。
对标竞赛获奖方向（Agent/RAG/多模态/XAI），最大化比赛获奖概率。

---

## 交付形态

```
spider-sense.exe (双击运行)
  ├── 内置 Flask 后端 (localhost:5000)
  ├── 内置 React 前端 (webview / 自动打开浏览器)
  ├── 内置 SQLite + 向量库 (轻量 RAG)
  ├── 内置检测模型 (Transformer + 传统方法)
  └── 内置攻击特征知识库 (MITRE ATT&CK 子集 + 自有数据)
```

**打包方式**: PyInstaller (--onefile) + 启动时自动打开默认浏览器
**备用方案**: Eel (Python + HTML/JS 原生窗口，无需浏览器)

---

## 架构设计

### 四层架构

```
┌─────────────────────────────────────────────────────┐
│  展示层  │ React Web UI (内嵌)                        │
│          │ 仪表盘 │ 实时监控 │ 智能研判 │ 报告         │
├─────────────────────────────────────────────────────┤
│  编排层  │ Flask API + Orchestrator                  │
│          │ 流量调度 → 触发检测 → 分发Agent → 汇总    │
├──────────┬──────────┬──────────┬────────────────────┤
│  能力层  │ 检测引擎  │ 多Agent  │ RAG知识库 │ XAI    │
│          │ Transformer│ 3Agent  │ SQLite+向量│ SHAP  │
│          │ +集成方法  │ 辩论投票 │ ATT&CK映射│ 归因   │
├──────────┴──────────┴──────────┴────────────────────┤
│  数据层  │ SQLite │ 模型文件 │ 攻击特征向量库 │ 配置  │
└─────────────────────────────────────────────────────┘
```

### 核心创新标签

| 概念 | 对标获奖项目 | 本项目实现 |
|------|-------------|-----------|
| **多智能体辩论** | 法码判官 | 3 Agent (Detector/Analyst/Arbiter) 并行研判 + 投票 |
| **RAG知识增强** | 天盾 | 攻击模式向量库 + MITRE ATT&CK 片段检索 |
| **XAI可解释** | XAI-Loop | SHAP/LIME 特征归因 + 自然语言解释 |
| **开放集识别** | 本项目核心 | Transformer特征空间 + 自适应阈值检测未知攻击 |

---

## 流程设计

### 流程一：实时检测 (核心环路)
```
抓包(scapy) → 流特征提取 → Transformer检测器打分 → 已知/未知判定
  ├── 已知攻击 → 分类 + 置信度
  └── 未知攻击 → 标记 UNKNOWN + 触发智能研判
```

### 流程二：智能研判 (多Agent辩论)
```
检测到异常 → 并行启动3个Agent:
  ├── Detector Agent: "这是什么类型的攻击？证据是什么？"
  ├── Analyst Agent: "是否为误报？有哪些反证？"
  └── Arbiter Agent: "综合双方意见，最终判定 + 建议处置措施"
→ 投票结论 + 生成自然语言研判报告
```

### 流程三：RAG知识检索
```
异常特征向量 → 检索相似攻击模式库 (cosine similarity)
  → 返回: Top-3 相似历史案例 + MITRE ATT&CK 技术映射
  → 增强Agent研判的上下文
```

### 流程四：文件批量检测
```
上传 PCAP/CSV → 解析流特征 → 批量检测 → 生成综合报告
  → 包含: 统计概览 + 异常详情 + 每条的Agent分析摘要
```

---

## 技术选型

### 检测引擎
- **主模型**: 1D Transformer Encoder (替换原CNN DHRNet)
  - 输入: 网络流特征向量 (兼容现有4个数据集的预处理)
  - 输出: 分类logits + 特征嵌入 (用于开放集判定)
- **辅助检测器**: Isolation Forest (统计异常，与Transformer互补)
- **开放集判定**: 自适应阈值 (基于特征空间距离的改进版，替代libMR/Weibull)
- **优势**: 不再需要Python 2.7，不再需要libMR编译

### 多智能体
- **本地模式**: 内置规则引擎 + 轻量分类器 (无需外部API)
- **在线增强模式**: 可选接入 LLM API (GPT-4 / Qwen) 进行自然语言研判
- **Agent框架**: 简单状态机，3个Agent并行运行

### RAG知识库
- **向量库**: SQLite + numpy (轻量，无需安装ChromaDB/FAISS)
- **嵌入**: 简单的特征向量直接作为嵌入 (无需额外模型)
- **内容**: 
  - 攻击模式向量库 (从训练数据提取)
  - MITRE ATT&CK 技术摘要 (预处理的JSON)
  - 历史检测案例记录

### XAI可解释性
- **特征归因**: SHAP TreeExplainer (Isolation Forest) + 梯度归因 (Transformer)
- **输出**: 特征贡献度排序 + 自然语言模板生成解释

### 前端
- **框架**: 保留现有 React + TypeScript (复用70%代码)
- **新增页面**: Agent研判面板、RAG检索面板、XAI解释面板
- **打包**: Vite构建 → 静态文件 → Flask serve

---

## 可执行文件打包

### 方案: PyInstaller + Flask serve 静态文件
```
PyInstaller --onefile --add-data "web-frontend/build:web-frontend/build"
             --add-data "models:models" --add-data "data:data"
             main.py
```
- 启动时: main.py 启动 Flask → 自动打开浏览器 → localhost:5000
- 最终产出: 单个 spider-sense.exe (~400-600MB 含模型)

### 目录结构
```
spider-sense-v2/
├── main.py                    # 入口，启动Flask + 打开浏览器
├── backend/
│   ├── app.py                 # Flask API
│   ├── detection_engine.py    # Transformer + IsolationForest 检测
│   ├── agent_layer.py         # 多Agent辩论引擎
│   ├── rag_engine.py          # RAG知识检索
│   ├── xai_engine.py          # XAI解释引擎
│   ├── orchestrator.py        # 编排器
│   └── feature_mapper.py      # 特征提取 (复用改进)
├── models/
│   ├── transformer_detector/  # 训练好的Transformer模型
│   └── scaler.pkl             # 特征标准化器
├── knowledge/
│   ├── attack_patterns.npz    # 攻击特征向量库
│   ├── mitre_attack.json      # MITRE ATT&CK 知识
│   └── detection_history.db   # SQLite历史记录
├── web-frontend/              # React前端 (复用+扩展)
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx       # 总览仪表盘
│   │   │   ├── LiveCapture.tsx     # 实时检测 (复用)
│   │   │   ├── AgentPanel.tsx      # 🆕 Agent研判面板
│   │   │   ├── RAGPanel.tsx        # 🆕 RAG检索面板
│   │   │   ├── XAIPanel.tsx        # 🆕 XAI解释面板
│   │   │   ├── FileUpload.tsx      # 🆕 文件批量检测
│   │   │   └── ReportExport.tsx    # 🆕 报告导出
│   │   └── App.tsx
│   └── build/                 # Vite构建产物
└── build_scripts/
    ├── build_exe.sh           # PyInstaller打包脚本
    └── spider-sense.spec      # PyInstaller配置
```

---

## 关键问题及解决方案

| 问题 | 解决方案 |
|------|---------|
| Python 2.7依赖 (libMR) | 用自适应阈值替代Weibull，彻底移除libMR |
| 模型需要GPU | Transformer轻量化 (<100MB)，CPU推理 |
| LLM API依赖 | 默认离线模式(规则引擎)，API可选增强 |
| 打包体积大 | 模型量化(INT8)，精简依赖，onefile压缩 |
| 实时抓包权限 | Windows用npcap，Linux用libpcap，自动检测 |

---

## 4天开发计划 (Vibecoding)

| 天 | 任务 | 产出 |
|----|------|------|
| **Day 1** | 检测引擎 + API核心 | Transformer模型替换 + Flask API + 自适应阈值 |
| **Day 2** | Agent层 + RAG + XAI | 3Agent辩论引擎 + 向量检索 + SHAP解释 |
| **Day 3** | 前端全面升级 | 5个新面板 + 仪表盘重设计 + API对接 |
| **Day 4** | 打包 + 测试 + 优化 | PyInstaller打包 → .exe + 端到端测试 + 性能调优 |

---

## UI概念

### 主仪表盘 (4卡片概览)
```
┌──────────┬──────────┬──────────┬──────────┐
│ 检测总数  │ 已知攻击  │ 未知威胁  │ 系统状态  │
│  12,847  │   1,203  │     47   │  🟢 正常  │
└──────────┴──────────┴──────────┴──────────┘
```

### Tab页
```
[📊 仪表盘] [📡 实时检测] [🧠 Agent研判] [🔍 RAG检索] [📁 文件检测] [📋 报告]
```

### Agent研判面板 (核心特色)
```
┌─────────────────────────────────────────────────────┐
│ 🧠 多智能体辩论研判                                   │
│                                                      │
│ 告警: 192.168.1.105 → 10.0.0.3 | PortScan 可疑       │
│                                                      │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐                 │
│ │🕵️检测员│ │🔬分析师 │ │⚖️裁决官│                 │
│ │置信度88%│ │疑误报30%│ │确认为   │                 │
│ │端口扫描  │ │可能是   │ │恶意扫描 │                 │
│ │特征匹配  │ │正常服务 │ │建议封禁 │                 │
│ │Score:92  │ │发现     │ │IP+阻断  │                 │
│ └─────────┘ └─────────┘ └─────────┘                 │
│                                                      │
│ 📋 最终判定: 恶意端口扫描 (2/3投票通过)               │
│ 🔗 MITRE ATT&CK: T1046 - Network Service Scanning    │
│ 💡 建议: 立即阻断源IP，检查目标端口开放情况           │
└─────────────────────────────────────────────────────┘
```

---

## 自审清单

- [x] 无TBD/TODO占位符
- [x] 架构与组件描述一致 (4层架构 × 4流程)
- [x] 范围可控 (4天, 单exe交付, 非多子系统分解)
- [x] 无歧义 (技术选型、流程、打包方式均已明确)
- [x] 创新标签对标清晰 (Agent/RAG/XAI/开放集)
