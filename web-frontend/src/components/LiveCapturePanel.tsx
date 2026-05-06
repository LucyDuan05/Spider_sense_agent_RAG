import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'

interface PacketItem {
  id: number
  timestamp: string
  srcIp: string
  dstIp: string
  protocol: string
  length: number
  prediction: string
  confidence: number
  srcPort?: number  // Added to identify injected packets (port 55555)
}

const INTERFACES = ['eth0', 'wlan0', 'any', 'en0', 'lo']

const ATTACK_TYPES = ['DDoS', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'DoS Hulk', 'Unknown Attack'] as const
type AttackType = typeof ATTACK_TYPES[number]

function predictionClass(prediction: string): string {
  if (prediction === 'BENIGN') return 'normal'
  if (prediction === 'UNKNOWN') return 'unknown'
  return 'anomaly'
}

const LiveCapturePanel: React.FC = () => {
  const [isCapturing, setIsCapturing] = useState(false)
  const [isStarting, setIsStarting] = useState(false)
  const [selectedInterface, setSelectedInterface] = useState('eth0')
  const [packets, setPackets] = useState<PacketItem[]>([])
  const [message, setMessage] = useState('等待开始...')
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState({ total: 0, normal: 0, anomaly: 0, unknown: 0 })
  const [injecting, setInjecting] = useState<AttackType | null>(null)
  const [injectMessage, setInjectMessage] = useState<string>('')
  const [showOnlyInjected, setShowOnlyInjected] = useState(false)

  const nextIdRef = useRef(1)
  // Tracks total packet count seen from backend to detect new packets
  const lastTotalRef = useRef(0)

  useEffect(() => {
    if (!isCapturing) return

    const interval = window.setInterval(async () => {
      try {
        const resp = await axios.get('/api/capture/status')
        const data = resp.data?.data
        if (!data) return

        // If backend stopped unexpectedly (permission error, etc.), sync state
        if (!data.isRunning) {
          setIsCapturing(false)
          if (data.flowDetail?.error) {
            setError(data.flowDetail.error)
          }
        }

        setMessage(`最新检测: ${data.prediction || 'N/A'}`)

        // Key fix: use totalCount as cursor to process ALL new packets
        const newTotal: number = data.totalCount ?? 0
        if (newTotal > lastTotalRef.current) {
          const diff = newTotal - lastTotalRef.current
          const recent: any[] = data.recent ?? []
          // recent is newest-first; take up to `diff` new ones
          const newFlows = recent.slice(0, Math.min(diff, recent.length))
          lastTotalRef.current = newTotal

          if (newFlows.length > 0) {
            const newPackets: PacketItem[] = newFlows.map((flow: any) => ({
              id: nextIdRef.current++,
              timestamp: flow.timestamp ?? new Date().toISOString(),
              srcIp: flow.src_ip ?? 'N/A',
              dstIp: flow.dst_ip ?? 'N/A',
              protocol: flow.protocol ?? 'OTHER',
              length: flow.packet_length ?? 0,
              prediction: flow.prediction ?? 'UNKNOWN',
              confidence: flow.confidence ?? 0,
              srcPort: flow.src_port ?? 0,
            }))

            setPackets(prev => [...newPackets, ...prev].slice(0, 100))
            setStats(prev => {
              let { total, normal, anomaly, unknown } = prev
              newPackets.forEach(p => {
                total++
                if (p.prediction === 'BENIGN') normal++
                else if (p.prediction === 'UNKNOWN') unknown++
                else anomaly++
              })
              return { total, normal, anomaly, unknown }
            })
          }
        }
      } catch {
        setError('无法获取抓包状态，请检查后端服务是否运行。')
        setIsCapturing(false)
      }
    }, 1000)

    return () => window.clearInterval(interval)
  }, [isCapturing])

  const handleStart = async () => {
    setError(null)
    setIsStarting(true)
    try {
      const resp = await axios.post('/api/capture/start', { interface: selectedInterface })
      if (resp.data?.success) {
        setIsCapturing(true)
        setPackets([])
        setStats({ total: 0, normal: 0, anomaly: 0, unknown: 0 })
        lastTotalRef.current = 0
        setMessage(resp.data.message ?? '开始抓包')
      } else {
        setError(resp.data?.error ?? '启动抓包失败')
      }
    } catch {
      setError('启动抓包请求失败，请检查后端服务器。')
    } finally {
      setIsStarting(false)
    }
  }

  const handleStop = async () => {
    try {
      await axios.post('/api/capture/stop')
    } catch {
      // ignore
    } finally {
      setIsCapturing(false)
      setMessage('抓包已停止')
    }
  }

  const handleInject = async (attackType: AttackType) => {
    setInjectMessage('')
    setInjecting(attackType)
    try {
      const resp = await axios.post('/api/capture/inject', {
        attack_type: attackType,
        count: 20,
      })
      if (resp.data?.success) {
        setInjectMessage(resp.data.message ?? `已开始注入 ${attackType}`)
      } else {
        setInjectMessage(`注入失败: ${resp.data?.error ?? 'unknown'}`)
      }
    } catch (err: any) {
      setInjectMessage(`注入请求失败: ${err?.response?.data?.error ?? err?.message ?? 'unknown'}`)
    } finally {
      // Re-enable button after 3s (100 packets @ 50/sec = 2s)
      setTimeout(() => setInjecting(null), 3000)
    }
  }

  return (
    <div>
      <h2 className="panel-title">
        <span>📡</span> Real-time Packet Capture &amp; Detection
      </h2>

      {/* Controls */}
      <div className="capture-controls">
        <select
          className="btn btn-secondary"
          value={selectedInterface}
          onChange={e => setSelectedInterface(e.target.value)}
          disabled={isCapturing}
          style={{ minWidth: 120 }}
        >
          {INTERFACES.map(iface => (
            <option key={iface} value={iface}>{iface}</option>
          ))}
        </select>

        {!isCapturing ? (
          <button className="btn btn-primary" onClick={handleStart} disabled={isStarting}>
            {isStarting ? '启动中...' : '▶ Start Capture'}
          </button>
        ) : (
          <button className="btn btn-danger" onClick={handleStop}>
            ⏹ Stop Capture
          </button>
        )}
      </div>

      {/* Status bar */}
      <div className="capture-status" style={{ marginBottom: 20 }}>
        <div className={`status-dot ${isCapturing ? 'active' : 'inactive'}`} />
        <span style={{ color: isCapturing ? '#00ff88' : '#ff4757', fontWeight: 'bold' }}>
          {isCapturing ? '抓包中' : '已停止'}
        </span>
        <span style={{ color: '#888', marginLeft: 12 }}>{message}</span>
      </div>

      {error && (
        <div style={{ color: '#ff4757', marginBottom: 16, padding: '10px 14px', background: 'rgba(255,71,87,0.1)', borderRadius: 8 }}>
          ⚠ {error}
        </div>
      )}

      {/* Attack Simulator */}
      <div className="attack-simulator" style={{
        marginBottom: 20,
        padding: '16px 18px',
        background: 'rgba(255,193,7,0.05)',
        border: '1px solid rgba(255,193,7,0.2)',
        borderRadius: 8,
      }}>
        <h3 style={{ color: '#ffc107', margin: '0 0 12px 0', fontSize: 15 }}>
          🎯 Attack Simulator
        </h3>
        <div style={{
          color: '#ffc107',
          fontSize: 12,
          marginBottom: 12,
          padding: '8px 10px',
          background: 'rgba(255,193,7,0.08)',
          borderRadius: 4,
        }}>
          💡 Tips: 点击 Start Capture 后可注入攻击样本。已知攻击（DDoS等）→ 模型正确分类；Unknown Attack 注入模型从未见过的攻击类型（Bot/Heartbleed等）→ 模型输出 UNKNOWN，体现开放集识别能力。
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
          {ATTACK_TYPES.map(attack => {
            const enabled = isCapturing && injecting === null
            const isUnknown = attack === 'Unknown Attack'
            return (
              <button
                key={attack}
                className="btn btn-secondary"
                disabled={!enabled}
                onClick={() => handleInject(attack)}
                style={{
                  opacity: enabled ? 1 : 0.4,
                  cursor: enabled ? 'pointer' : 'not-allowed',
                  fontSize: 13,
                  padding: '6px 14px',
                  background: isUnknown ? 'rgba(138,43,226,0.2)' : undefined,
                  border: isUnknown ? '1px solid #8a2be2' : undefined,
                  color: isUnknown ? '#c38aff' : undefined,
                }}
              >
                {injecting === attack ? `注入中... ${attack}` : (isUnknown ? `❓ ${attack}` : `💥 ${attack}`)}
              </button>
            )
          })}
        </div>
        {injectMessage && (
          <div style={{ color: '#00d4ff', fontSize: 12, marginTop: 8 }}>
            {injectMessage}
          </div>
        )}
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid rgba(255,193,7,0.2)' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 13, color: '#ffc107' }}>
            <input
              type="checkbox"
              checked={showOnlyInjected}
              onChange={(e) => setShowOnlyInjected(e.target.checked)}
              style={{ cursor: 'pointer' }}
            />
            <span>仅显示注入的包（源端口 55555）</span>
          </label>
        </div>
      </div>

      {/* Stats */}
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4,1fr)', marginBottom: 20 }}>
        <div className="metric-card">
          <div className="metric-value">{stats.total}</div>
          <div className="metric-label">Total Packets</div>
        </div>
        <div className="metric-card" style={{ borderTop: '2px solid #00ff88' }}>
          <div className="metric-value" style={{ color: '#00ff88' }}>{stats.normal}</div>
          <div className="metric-label">Normal (BENIGN)</div>
        </div>
        <div className="metric-card" style={{ borderTop: '2px solid #ff4757' }}>
          <div className="metric-value" style={{ color: '#ff4757' }}>{stats.anomaly}</div>
          <div className="metric-label">Anomaly</div>
        </div>
        <div className="metric-card" style={{ borderTop: '2px solid #ffc107' }}>
          <div className="metric-value" style={{ color: '#ffc107' }}>{stats.unknown}</div>
          <div className="metric-label">Unknown</div>
        </div>
      </div>

      {/* Packet Table */}
      <h3 style={{ color: '#fff', marginBottom: 12 }}>
        Recent Detection Results
        {showOnlyInjected && <span style={{ color: '#ffc107', fontSize: 13, marginLeft: 10 }}>(仅显示注入的包)</span>}
      </h3>
      <div className="packet-list">
        {packets.length === 0 ? (
          <div style={{ color: '#666', textAlign: 'center', padding: 40 }}>
            {isCapturing ? '等待数据包...' : '点击 "Start Capture" 开始抓包'}
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                {['Time', 'Src IP', 'Dst IP', 'Protocol', 'Prediction', 'Confidence'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '10px 12px', color: '#888', fontWeight: 600 }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {packets
                .filter(pkt => !showOnlyInjected || pkt.srcPort === 55555)
                .map(pkt => {
                  const isInjected = pkt.srcPort === 55555
                  return (
                    <tr
                      key={pkt.id}
                      style={{
                        borderBottom: '1px solid rgba(255,255,255,0.04)',
                        background: isInjected ? 'rgba(255,193,7,0.08)' : 'transparent',
                        borderLeft: isInjected ? '3px solid #ffc107' : 'none',
                      }}
                      className="packet-item-row"
                    >
                      <td style={{ padding: '8px 12px', color: '#888', whiteSpace: 'nowrap' }}>
                        {pkt.timestamp.split('T')[1]?.split('.')[0] ?? pkt.timestamp}
                      </td>
                      <td style={{ padding: '8px 12px', color: '#00d4ff' }}>
                        {pkt.srcIp}
                        {isInjected && <span style={{ color: '#ffc107', fontSize: 11, marginLeft: 4 }}>:55555</span>}
                      </td>
                      <td style={{ padding: '8px 12px', color: '#00d4ff' }}>{pkt.dstIp}</td>
                      <td style={{ padding: '8px 12px', color: '#e0e0e0' }}>{pkt.protocol}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span className={`packet-type ${predictionClass(pkt.prediction)}`}>
                          {pkt.prediction}
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px', color: '#888' }}>
                        {(pkt.confidence * 100).toFixed(1)}%
                      </td>
                    </tr>
                  )
                })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

export default LiveCapturePanel
