# CROSR IDS 系统启动指南

## 环境要求

| 组件 | 版本要求 |
|------|----------|
| Python | 3.9+（推荐使用 `openmax` conda 环境） |
| Node.js | 18+ |
| npm | 9+ |

所有 Python 依赖已包含在 `openmax` conda 环境中：`flask`, `flask-cors`, `torch`, `numpy`, `scapy`

---

## 启动步骤

### 第一步：启动后端（Flask）

打开终端，激活 conda 环境后启动：

```bash
conda activate openmax
cd ~/CROSR-main/web-frontend/backend

# 抓包需要 root 权限，必须用 conda 环境的完整 Python 路径
sudo /home/mihu/miniconda3/envs/openmax/bin/python3 app.py
```

后端成功启动后输出类似：

```
[IDS] Model loaded: 6 classes, 78 features
 * Running on http://0.0.0.0:5000
```

> **注意**：不加 `sudo` 也能启动，但实时抓包功能会报权限错误。若只使用数据集展示和训练流水线页面，可以不加 `sudo`。

---

### 第二步：启动前端（React）

另开一个终端：

```bash
cd ~/CROSR-main/web-frontend
npm start
```

启动后浏览器自动打开 **http://localhost:3000**

---

## 常见问题

### `ModuleNotFoundError: No module named 'flask'`

**原因**：`sudo` 默认使用系统 Python，而不是 conda 环境里的 Python。

**解决**：用 conda 环境的完整路径，或加 `-E` 参数保留环境变量：

```bash
# 方法一（推荐）：指定完整 Python 路径
sudo /home/mihu/miniconda3/envs/openmax/bin/python3 app.py

# 方法二：保留当前环境变量
sudo -E python3 app.py
```

---

### 抓包提示 `需要 root 权限`

Scapy 抓包需要原始套接字权限，必须以 `sudo` 启动后端。见上方第一步。

---

### 前端提示 `无法获取抓包状态，请检查后端服务是否运行`

后端未启动或启动失败。确认终端一中后端已正常运行，访问以下地址验证：

```
http://localhost:5000/api/health
```

返回 `{"status": "healthy", ...}` 表示正常。

---

### `[WARNING] Failed to load model`

模型文件不存在或路径错误。确认以下文件存在：

```
~/CROSR-main/save_models/cicids_1d/latest.pth
~/CROSR-main/processed_cicids/scaler.npz
```

---

## 功能说明

| 页面 | 功能 |
|------|------|
| **Dataset Results** | 查看 CICIDS 2017/2018、NSL-KDD、UNSW-NB15 四个数据集的检测指标和分类统计图表 |
| **Live Capture** | 实时抓包并用 DHRNet 模型对每个数据包进行分类（需要 sudo） |
| **Training Pipeline** | 可视化模型训练的 7 个步骤流水线，含实时日志输出 |
