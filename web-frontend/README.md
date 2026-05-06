# CROSR 网络入侵检测系统 - Web 前端

基于 CROSR (Classification-Reconstruction Learning for Open-Set Recognition) 算法的网络入侵检测系统 Web 界面。

## 功能特性

### ? 数据集测试结果可视化
- 支持 CICIDS 2017、CICIDS 2018、NSL-KDD、UNSW-NB15 四个公开数据集
- 展示 AUROC、Precision、Recall、F1-Score 等指标
- 各类别检测统计图表
- 数据集卡片式选择界面

### ? 实时抓包与模型检测
- 模拟实时网络流量捕获
- 实时显示数据包检测结果（正常/异常/未知）
- 支持按 IP/协议过滤
- 检测统计实时更新

### ?? 训练流程可视化
- 7 步训练流程进度展示
- 实时日志输出
- 支持选择不同数据集
- 训练状态实时反馈

## 技术栈

- **前端**: React 18 + TypeScript
- **图表**: Recharts
- **后端**: Flask (Python)
- **样式**: 自定义 CSS (深色主题)

## 项目结构

```
web-frontend/
├── public/
│   └── index.html          # HTML 入口
├── src/
│   ├── components/
│   │   ├── DatasetPanel.tsx     # 数据集面板
│   │   ├── LiveCapturePanel.tsx # 实时抓包面板
│   │   └── TrainingPanel.tsx    # 训练流程面板
│   ├── App.tsx              # 主应用组件
│   ├── App.css              # 样式文件
│   └── index.tsx           # React 入口
├── backend/
│   └── app.py               # Flask 后端 API
├── package.json
├── tsconfig.json
└── README.md
```

## 快速开始

### 前置要求

- Node.js 18+
- Python 3.9+
- conda (推荐)

### 安装依赖

```bash
# 1. 安装前端依赖
cd web-frontend
npm install

# 2. 创建 Python 虚拟环境 (可选)
conda create -n crosr-web python=3.9
conda activate crosr-web
pip install flask flask-cors
```

### 启动开发服务器

```bash
# 方式1: 仅启动前端 (数据模拟)
cd web-frontend
npm start
```

```bash
# 方式2: 启动完整前后端

# 终端1: 启动后端
cd web-frontend/backend
python app.py

# 终端2: 启动前端
cd web-frontend
npm start
```

访问 http://localhost:3000

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/datasets` | 获取所有数据集 |
| GET | `/api/datasets/<id>` | 获取数据集结果 |
| GET | `/api/datasets/<id>/class-results` | 获取类别检测结果 |
| POST | `/api/capture/start` | 开始抓包 |
| POST | `/api/capture/stop` | 停止抓包 |
| POST | `/api/capture/predict` | 预测数据包 |
| POST | `/api/training/start` | 开始训练 |
| GET | `/api/training/status` | 获取训练状态 |

## 数据集结果

# OpenMax 评估结果（最佳参数下）

| Dataset     | AUROC  | Precision | Recall | F1     | Bal.Acc | FPR@95TPR | Best Params             |
| ----------- | ------ | --------- | ------ | ------ | ------- | --------- | ----------------------- |
| CICIDS-2017 | 0.9651 | 0.8937    | 0.9357 | 0.9143 | 0.9210  | 0.1243    | tail=30, α=3, euclidean |
| UNSW-NB15   | 0.8949 | 0.5843    | 0.8718 | 0.6997 | 0.8396  | 0.3627    | tail=50, α=6, cosine    |
| NSL-KDD     | 0.7587 | 0.2823    | 0.7835 | 0.4148 | 0.7417  | 0.6873    | tail=50, α=1, euclidean |
| CICIDS-2018 | 0.4613 | 0.4665    | 0.9957 | 0.6344 | 0.6318  | 0.7100    | tail=5, α=1, euclidean  |

## 扩展开发

### 添加新数据集

1. 在 `DatasetPanel.tsx` 的 `DATASET_RESULTS` 中添加数据集信息
2. 在 `CLASS_DETECTION_DATA` 中添加类别检测数据
3. 在后端 `app.py` 的 `DATASET_RESULTS` 中添加对应结果

### 集成真实模型

修改 `LiveCapturePanel.tsx` 中的预测逻辑，调用后端 API:

```typescript
const response = await axios.post('/api/capture/predict', {
  features: packetFeatures
});
```

### 添加新的训练步骤

修改 `TrainingPanel.tsx` 中的 `TRAINING_STEPS` 数组。

## 许可证

MIT License - 见父项目 CROSR-main