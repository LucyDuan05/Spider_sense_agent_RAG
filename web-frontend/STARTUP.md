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
| **Live Capture** | 实时抓包并用 DHRNet 模型对每个数据包进行分类，含 **Attack Simulator** 攻击注入演示（需要 sudo） |
| **Training Pipeline** | 可视化模型训练的 7 个步骤流水线，含实时日志输出 |

---

## Attack Simulator 使用方法

`Live Capture` 页面下方提供 **Attack Simulator** 模块，用于演示开放集识别能力。

### 使用步骤

1. 在 `Live Capture` 页面任选一个网络接口（推荐 `eth0`）
2. 点击 **Start Capture** 启动抓包
3. 在 Attack Simulator 区域点击对应的攻击按钮：
   - **💥 已知攻击**：DDoS、PortScan、FTP-Patator、SSH-Patator、DoS Hulk
     → 模型应正确分类为对应攻击类型
   - **❓ Unknown Attack**：从 `open_set.npz` 随机抽取模型从未见过的攻击（Bot / Heartbleed / DoS GoldenEye / DoS slowloris / Infiltration / Web Attack 等9类）
     → 模型应输出 **UNKNOWN**，体现开放集识别的拒判能力
4. 可勾选 **"仅显示注入的包（源端口 55555）"** 过滤掉系统正常流量，专注查看注入结果
5. 表格中注入的包以**黄色高亮**显示，并标注源端口 `:55555`

### 演示要点

| 操作 | 期望结果 | 说明 |
|------|----------|------|
| 点击 `DDoS` | 表格出现 `DDoS` 预测，置信度 > 90% | 已知类别正确分类 |
| 点击 `❓ Unknown Attack` | 表格出现 `UNKNOWN` 预测，置信度 < 65% | 开放集拒判生效 |
| 不注入仅抓 eth0 | 大部分为 `BENIGN` | 正常流量识别 |
