import React, { useState, useCallback, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import DetailDrawer from './components/DetailDrawer';

interface EventItem {
  id: number;
  timestamp: string;
  prediction: string;
  confidence: number;
  isUnknown: boolean;
  unknownScore: number;
  srcIp: string;
  dstIp: string;
  protocol: string;
  features: number[];
  rawResult: any;
  sourceType: 'simulated' | 'injected' | 'captured';  // 数据来源
}

const PROTOCOLS = ['TCP', 'UDP', 'HTTP', 'DNS'];
const IP_INT = ['192.168.1.10', '192.168.1.25', '192.168.1.50'];
const IP_EXT = ['10.0.0.5', '172.16.0.30', '203.0.113.99', '198.51.100.20', '185.220.101.34'];

function genFeatures(type: string): number[] {
  const base = Array.from({ length: 78 }, () => Math.random() * 0.5);
  if (type === 'ddos') { for (let i = 0; i < 12; i++) base[i] = 0.6 + Math.random() * 0.4; }
  else if (type === 'scan') { for (let i = 20; i < 35; i++) base[i] = 0.5 + Math.random() * 0.5; }
  else if (type === 'brute') { base[10] = 0.7 + Math.random() * 0.3; base[14] = Math.random() * 0.2; }
  return base;
}

// BENIGN → NORMAL display mapping (不影响后端)
function displayLabel(label: string): string {
  return label === 'BENIGN' ? 'NORMAL' : label;
}

function App() {
  const [events, setEvents] = useState<EventItem[]>([]);
  const [selected, setSelected] = useState<EventItem | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [source, setSource] = useState<'live' | 'scan' | 'brute' | 'mixed'>('live');
  const [running, setRunning] = useState(false);
  const [captureRunning, setCaptureRunning] = useState(false);
  const [attackType, setAttackType] = useState('DDoS');
  const [attackCount, setAttackCount] = useState(8);
  const [captureMessage, setCaptureMessage] = useState('');
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [captureStats, setCaptureStats] = useState<any>(null);
  const [availableInterfaces, setAvailableInterfaces] = useState<{name: string; ip: string; description: string}[]>([]);
  const [selectedInterface, setSelectedInterface] = useState('');

  const attackTypes = ['DDoS', 'DoS Hulk', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'Unknown Attack'];

  // ── 抓包轮询 ──
  const captureTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const nextId = useRef(1);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [drawerData, setDrawerData] = useState<any>(null);
  const [drawerLoading, setDrawerLoading] = useState(false);
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [llmModel, setLlmModel] = useState('gpt-4');
  const [llmApiBase, setLlmApiBase] = useState('https://api.openai.com/v1');
  const [llmApiKey, setLlmApiKey] = useState('');
  const [llmMessage, setLlmMessage] = useState('');
  const [llmError, setLlmError] = useState<string | null>(null);

  useEffect(() => {
    if (captureRunning) {
      // 每 2 秒轮询一次抓包结果
      captureTimerRef.current = setInterval(async () => {
        try {
          const [packetResp, statusResp] = await Promise.all([
            axios.get('/api/capture/packets'),
            axios.get('/api/capture/status'),
          ]);
          if (statusResp.data?.success) {
            setCaptureStats(statusResp.data.data.capture_engine);
          }
          if (packetResp.data?.success && packetResp.data.data.length > 0) {
            setEvents(prev => [
              ...packetResp.data.data.map((item: any) => ({
                id: nextId.current++,
                timestamp: item.timestamp || new Date().toISOString(),
                prediction: item.prediction,
                confidence: item.confidence,
                isUnknown: item.isUnknown,
                unknownScore: item.unknownScore,
                srcIp: item.srcIp,
                dstIp: item.dstIp,
                protocol: item.protocol,
                features: item.features || [],
                rawResult: item.rawResult || {},
                sourceType: 'captured' as const,
              })),
              ...prev,
            ].slice(0, 500));
          }
        } catch {}
      }, 2000);
    }
    return () => {
      if (captureTimerRef.current) {
        clearInterval(captureTimerRef.current);
        captureTimerRef.current = null;
      }
    };
  }, [captureRunning]);

  const runOne = useCallback(async () => {
    let features: number[];
    let proto: string;
    let src: string;
    let dst: string;
    if (source === 'scan') features = genFeatures('scan');
    else if (source === 'brute') features = genFeatures('brute');
    else if (source === 'mixed') features = genFeatures(['normal','ddos','scan','brute'][Math.floor(Math.random()*4)]);
    else { const r = Math.random(); features = r < 0.4 ? genFeatures('normal') : r < 0.7 ? genFeatures('ddos') : genFeatures('scan'); }
    src = IP_EXT[Math.floor(Math.random() * IP_EXT.length)];
    dst = IP_INT[Math.floor(Math.random() * IP_INT.length)];
    proto = PROTOCOLS[Math.floor(Math.random() * PROTOCOLS.length)];
    try {
      const resp = await axios.post('/api/detect', { features, flow_info: { src_ip: src, dst_ip: dst, protocol: proto, packet_length: Math.floor(Math.random() * 1400) + 60 } });
      if (resp.data?.success) {
        const det = resp.data.data.detection;
        setEvents(prev => [{
          id: nextId.current++,
          timestamp: new Date().toISOString(),
          prediction: det.prediction,
          confidence: det.class_confidence,
          isUnknown: det.is_unknown,
          unknownScore: det.unknown_prob || 0,
          srcIp: src, dstIp: dst, protocol: proto, features, rawResult: resp.data.data,
          sourceType: 'simulated' as const,
        }, ...prev].slice(0, 500));
      }
    } catch {}
  }, [source]);

  useEffect(() => {
    if (running) { runOne(); timerRef.current = setInterval(runOne, 2000); }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running, runOne]);

  const startCapture = async () => {
    try {
      setCaptureError(null);
      // 不传 interface，由后端 auto-detect（Windows 没有 lo 接口）
      const resp = await axios.post('/api/capture/start', {
        interface: selectedInterface || undefined,
      });
      if (resp.data?.success) {
        setCaptureRunning(true);
        const iface = resp.data.interface || 'auto';
        setCaptureMessage(`捕获已启动 (${iface})`);
      } else {
        setCaptureError(resp.data?.error || '启动捕获失败');
      }
    } catch (err: any) {
      setCaptureError(err?.response?.data?.error || '启动捕获失败，请检查 Npcap 是否已安装');
    }
  };

  const stopCapture = async () => {
    try {
      setCaptureError(null);
      const resp = await axios.post('/api/capture/stop');
      if (resp.data?.success) {
        setCaptureRunning(false);
        setCaptureMessage(resp.data.message || '捕获已停止');
      }
    } catch (err: any) {
      setCaptureError(err?.response?.data?.error || '停止捕获失败');
    }
  };

  const injectAttack = async () => {
    try {
      setCaptureError(null);
      setCaptureMessage('注入中...');
      const resp = await axios.post('/api/capture/inject', {
        attack_type: attackType,
        count: attackCount,
      });
      if (resp.data?.success) {
        const injected = resp.data.data || [];
        setEvents(prev => [
          ...injected.map((item: any, idx: number) => ({
            id: nextId.current++,
            timestamp: item.timestamp || new Date().toISOString(),
            prediction: item.prediction,
            confidence: item.confidence,
            isUnknown: item.isUnknown,
            unknownScore: item.unknownScore,
            srcIp: item.srcIp,
            dstIp: item.dstIp,
            protocol: item.protocol,
            features: item.features || [],
            rawResult: item.rawResult || {},
            sourceType: 'injected' as const,
          })),
          ...prev,
        ].slice(0, 500));
        setCaptureMessage(resp.data.message || '注入完成');
      }
    } catch (err: any) {
      setCaptureError(err?.response?.data?.error || '注入失败');
      setCaptureMessage('注入失败');
    }
  };

  const loadLlmConfig = async () => {
    try {
      const resp = await axios.get('/api/config/llm');
      if (resp.data?.success) {
        const data = resp.data.data;
        setLlmEnabled(data.enabled);
        setLlmModel(data.model || 'gpt-4');
        setLlmApiBase(data.api_base || 'https://api.openai.com/v1');
      }
    } catch (err) {
      console.warn('Failed to load LLM config', err);
    }
  };

    // ── 加载可用网卡列表 ──
  const loadInterfaces = async () => {
    try {
      const resp = await axios.get('/api/capture/status');
      const ifaces = resp.data?.data?.capture_engine?.available_interfaces;
      if (ifaces && ifaces.length > 0) {
        setAvailableInterfaces(ifaces);
        if (!selectedInterface) setSelectedInterface(ifaces[0].name);
      }
    } catch {}
  };

  const saveLlmConfig = async () => {
    try {
      setLlmError(null);
      setLlmMessage('正在保存 LLM 配置...');
      const resp = await axios.post('/api/config/llm', {
        api_key: llmEnabled ? llmApiKey : '',
        model: llmModel,
        api_base: llmApiBase,
      });
      if (resp.data?.success) {
        setLlmMessage('LLM 配置已保存。当前模式: ' + (resp.data.data.enabled ? 'LLM' : 'Rule-Based'));
      }
    } catch (err: any) {
      setLlmError(err?.response?.data?.error || '保存失败');
      setLlmMessage('保存失败');
    }
  };

  useEffect(() => {
    loadLlmConfig();
    loadInterfaces();
  }, []);

  const handleSelect = async (ev: EventItem) => {
    setSelected(ev); setDrawerOpen(true); setDrawerLoading(true); setDrawerData(null);
    try {
      const [ragResp, xaiResp, debateResp] = await Promise.all([
        axios.post('/api/rag/search', { features: ev.features, top_k: 3 }),
        axios.post('/api/xai/explain', { features: ev.features }),
        axios.post('/api/debate', { detection: ev.rawResult?.detection || { prediction: ev.prediction, class_confidence: ev.confidence, is_unknown: ev.isUnknown, unknown_prob: ev.unknownScore }, flow_info: { src_ip: ev.srcIp, dst_ip: ev.dstIp, protocol: ev.protocol }, features: ev.features }),
      ]);
      setDrawerData({ rag: ragResp.data?.data || null, xai: xaiResp.data?.data || null, debate: debateResp.data?.data?.debate || null });
    } catch {} finally { setDrawerLoading(false); }
  };

  const stats = {
    total: events.length,
    critical: events.filter(e => e.unknownScore > 0.5 || e.isUnknown).length,
    attack: events.filter(e => e.prediction !== 'BENIGN' && e.prediction !== 'UNKNOWN').length,
    benign: events.filter(e => e.prediction === 'BENIGN').length,
  };
  const topAttacks = (() => {
    const counts: Record<string, number> = {};
    events.forEach(e => { if (e.prediction !== 'BENIGN') counts[e.prediction] = (counts[e.prediction] || 0) + 1; });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 3);
  })();
  const getScoreColor = (score: number) => score > 0.5 ? 'critical' : score > 0.3 ? 'high' : 'low';

  return (
    <div className="app">
      {/* Top Bar */}
      <div className="topbar">
        <div>
          <div className="topbar-title">Spider-Sense v2</div>
          <div className="topbar-subtitle">实时网络攻击检测 · 多智能体协作 · RAG知识增强 · XAI可解释</div>
        </div>
        <div className="topbar-stats">
          <div className="topbar-stat"><div className="val">{stats.total}</div><div className="lbl">检测事件</div></div>
          <div className="topbar-stat"><div className="val" style={{ color: stats.critical > 0 ? 'var(--red)' : 'var(--green)' }}>{stats.critical}</div><div className="lbl">未知威胁</div></div>
          <div className="topbar-stat"><div className="val" style={{ color: stats.attack > 0 ? 'var(--yellow)' : 'var(--text)' }}>{stats.attack}</div><div className="lbl">已知攻击</div></div>
          <div className="topbar-stat"><div className="val" style={{ color: stats.benign > 0 ? 'var(--green)' : 'var(--text)' }}>{stats.benign}</div><div className="lbl">正常流量</div></div>
          {llmEnabled && (
            <div className="topbar-stat">
              <div className="val" style={{ fontSize: 13, color: '#00d4ff' }}>🤖 LLM</div>
              <div className="lbl">{llmModel}</div>
            </div>
          )}
        </div>
      </div>

      <div className="main-layout">
        {/* Sidebar */}
        <div className="input-bar">
          <div className="section-title">数据源</div>
          <button className={source === 'live' ? 'active' : ''} onClick={() => setSource('live')}>混合流量模拟</button>
          <button className={source === 'scan' ? 'active' : ''} onClick={() => setSource('scan')}>端口扫描模式</button>
          <button className={source === 'brute' ? 'active' : ''} onClick={() => setSource('brute')}>暴力破解模式</button>
          <button className={source === 'mixed' ? 'active' : ''} onClick={() => setSource('mixed')}>全部攻击模式</button>

          <div style={{ marginTop: 12 }}>
            <button className={running ? '' : 'active'} onClick={() => setRunning(!running)} style={{ textAlign: 'center' }}>
              {running ? '⏹ 停止模拟' : '▶ 开始模拟数据'}
            </button>
          </div>
          {running && <div style={{ fontSize: 10, color: 'var(--green)', marginTop: 4 }}>● 检测中</div>}

          <div className="section-title" style={{ marginTop: 20 }}>真实注入</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {/* 网卡选择 */}
            {availableInterfaces.length > 0 && !captureRunning && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <div style={{ fontSize: 9, color: 'var(--muted)' }}>选择网卡</div>
                <select
                  value={selectedInterface}
                  onChange={e => setSelectedInterface(e.target.value)}
                  style={{ width: '100%', padding: 6, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)', fontSize: 10 }}
                >
                  {availableInterfaces.map(iface => (
                    <option key={iface.name} value={iface.name}>
                      {iface.name} ({iface.ip})
                    </option>
                  ))}
                </select>
              </div>
            )}
            <button
              className={captureRunning ? '' : 'active'}
              onClick={captureRunning ? stopCapture : startCapture}
              style={{ textAlign: 'center' }}
            >
              {captureRunning ? '⏹ 停止捕获' : '▶ 启动捕获'}
            </button>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <select value={attackType} onChange={e => setAttackType(e.target.value)} style={{ width: '100%', padding: 8, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)' }}>
                {attackTypes.map(type => <option key={type} value={type}>{type}</option>)}
              </select>
              <input
                type="number"
                min={1}
                max={50}
                value={attackCount}
                onChange={e => setAttackCount(Number(e.target.value))}
                style={{ width: '100%', padding: 8, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)' }}
              />
            </div>
            <button className="active" onClick={injectAttack} style={{ textAlign: 'center' }}>
              注入选中攻击样本
            </button>
            {captureMessage && <div style={{ fontSize: 10, color: 'var(--text)', marginTop: 4 }}>{captureMessage}</div>}
            {captureError && <div style={{ fontSize: 10, color: 'var(--red)', marginTop: 4 }}>{captureError}</div>}
            {captureRunning && captureStats && (
              <div style={{ fontSize: 9, color: 'var(--muted)', lineHeight: 1.6, marginTop: 2 }}>
                接口: {captureStats.interface || 'N/A'} · 包: {captureStats.captured_count || 0}
                {captureStats.error && <div style={{ color: 'var(--red)' }}>错误: {captureStats.error}</div>}
              </div>
            )}
          </div>

          <div className="section-title" style={{ marginTop: 20 }}>LLM 配置</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <label style={{ fontSize: 11, color: 'var(--muted)' }}>
              <input type="checkbox" checked={llmEnabled} onChange={e => setLlmEnabled(e.target.checked)} style={{ marginRight: 6 }} />
              启用 LLM 模式
            </label>
            <input
              type="password"
              placeholder="LLM API Key"
              value={llmApiKey}
              onChange={e => setLlmApiKey(e.target.value)}
              style={{ width: '100%', padding: 8, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)' }}
            />
            <input
              type="text"
              placeholder="LLM Model (e.g. gpt-4 or ds)"
              value={llmModel}
              onChange={e => setLlmModel(e.target.value)}
              style={{ width: '100%', padding: 8, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)' }}
            />
            <input
              type="text"
              placeholder="LLM API Base (optional)"
              value={llmApiBase}
              onChange={e => setLlmApiBase(e.target.value)}
              style={{ width: '100%', padding: 8, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface2)', color: 'var(--text)' }}
            />
            <button className="active" onClick={saveLlmConfig} style={{ textAlign: 'center' }}>
              保存 LLM 配置
            </button>
            {llmMessage && <div style={{ fontSize: 10, color: 'var(--text)', marginTop: 4 }}>{llmMessage}</div>}
            {llmError && <div style={{ fontSize: 10, color: 'var(--red)', marginTop: 4 }}>{llmError}</div>}
            {/* 安全提示 */}
            <div style={{ fontSize: 9, color: 'var(--muted)', lineHeight: 1.6, padding: '4px 0', borderTop: '1px solid var(--border)', marginTop: 4 }}>
              🔒 API Key 仅在当前会话内存中保存，<b>重启后自动清除</b>，不会写入磁盘。
            </div>
          </div>

          {/* Tech highlights */}
          <div className="section-title" style={{ marginTop: 20 }}>核心技术</div>
          <div style={{ fontSize: 10, color: 'var(--muted)', lineHeight: 1.8 }}>
            <div>🧠 多智能体协同研判</div>
            <div>🔍 RAG 知识增强检索</div>
            <div>📊 XAI 可解释性归因</div>
            <div>🎯 开放集未知攻击识别</div>
            <div>⚡ 纯 Python — 无 Python2 依赖</div>
          </div>

          {/* Dataset results */}
          <div className="section-title" style={{ marginTop: 16 }}>性能指标 (CICIDS-2017)</div>
          <div style={{ fontSize: 11, color: 'var(--text)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 0' }}><span>AUROC</span><span style={{ color: 'var(--green)' }}>0.915</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 0' }}><span>原版 (libMR)</span><span style={{ color: 'var(--muted)' }}>0.965</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 0' }}><span>识别类别</span><span>6类</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 0' }}><span>检测模型</span><span>DHRNet-1D</span></div>
          </div>

          {/* Top attacks */}
          {topAttacks.length > 0 && (
            <>
              <div className="section-title" style={{ marginTop: 16 }}>攻击类型分布</div>
              {topAttacks.map(([name, count]) => (
                <div key={name} style={{ fontSize: 11, color: 'var(--muted)', display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                  <span>{name}</span><span>{count}</span>
                </div>
              ))}
            </>
          )}
        </div>

        {/* Event Stream */}
        <div className="event-stream">
          {events.length === 0 ? (
            <div className="empty-state">
              <div className="icon" style={{ fontSize: 40 }}>🕸️</div>
              <div className="msg" style={{ fontSize: 15, color: 'var(--text)', marginBottom: 8 }}>Spider-Sense v2 — 蜘蛛感应</div>
              <div className="msg" style={{ marginBottom: 16 }}>多智能体协同 · 开放集识别 · RAG增强 · XAI可解释</div>
              <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.8, maxWidth: 480, margin: '0 auto', textAlign: 'left' }}>
                <p>本系统将 CVPR 2019 CROSR 论文方案重构为完整的网络入侵检测控制台。核心改进：</p>
                <p style={{ marginTop: 8 }}>🔹 <b>纯 Python OpenMax</b> — 用 NumPy/SciPy 重写 Weibull 拟合，彻底移除 libMR/Python 2.7 依赖，AUROC 0.915 接近原版 0.965</p>
                <p>🔹 <b>多智能体辩论</b> — 检测员/分析师/裁决官三个 Agent 独立研判、投票裁决每条告警</p>
                <p>🔹 <b>RAG 知识增强</b> — 68 条攻击模式向量 + 10 项 MITRE ATT&CK 技术，检索相似案例辅助判断</p>
                <p>🔹 <b>XAI 可解释</b> — 特征扰动归因分析，揭示哪些流量特征驱动了检测决策</p>
                <p style={{ marginTop: 8 }}>点击左侧「开始模拟数据」或选择数据源，即可体验检测流程。回放和抓包数据会标注来源。</p>
              </div>
            </div>
          ) : (
            events.map(ev => (
              <div
                key={ev.id}
                className={`event-card ${selected?.id === ev.id ? 'selected' : ''}`}
                onClick={() => handleSelect(ev)}
              >
                <div className={`score ${getScoreColor(ev.unknownScore)}`}>
                  {Math.round(ev.unknownScore * 100)}
                </div>
                <div className="info">
                  <div className="label">
                    {displayLabel(ev.prediction)}
                    {ev.isUnknown && <span style={{ color: 'var(--yellow)', fontSize: 10, marginLeft: 6 }}>未知攻击</span>}
                  </div>
                  <div className="flow">{ev.srcIp} → {ev.dstIp} &middot; {ev.protocol}</div>
                </div>
                <div className="meta">
                  <div className="time">{ev.timestamp.split('T')[1]?.split('.')[0] || ev.timestamp}</div>
                  <span className={`tag ${ev.prediction === 'BENIGN' ? 'tag-green' : ev.isUnknown ? 'tag-yellow' : 'tag-red'}`}>
                    {(ev.confidence * 100).toFixed(0)}%
                  </span>
                  {/* 数据来源标签 */}
                  <span className={`source-tag source-${ev.sourceType}`}>
                    {ev.sourceType === 'simulated' ? '模拟' : ev.sourceType === 'injected' ? '回放' : '捕获'}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Drawer */}
      <div className={`drawer-overlay ${drawerOpen ? 'open' : ''}`} onClick={() => { setDrawerOpen(false); setSelected(null); }} />
      <div className={`drawer ${drawerOpen ? 'open' : ''}`}>
        {selected && (
          <>
            <div className="drawer-header">
              <div>
                <h2>{displayLabel(selected.prediction)}{selected.isUnknown ? ' (未知攻击)' : ''}</h2>
                <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>
                  {selected.srcIp} → {selected.dstIp} &middot; {selected.protocol} &middot; {selected.timestamp.split('T')[1]?.split('.')[0]}
                </div>
              </div>
              <button className="drawer-close" onClick={() => { setDrawerOpen(false); setSelected(null); }}>✕</button>
            </div>
            <div className="drawer-body">
              <DetailDrawer event={selected} data={drawerData} loading={drawerLoading} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
