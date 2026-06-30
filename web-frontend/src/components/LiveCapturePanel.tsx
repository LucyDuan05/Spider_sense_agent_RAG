import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';

interface DetectionItem {
  id: number;
  timestamp: string;
  prediction: string;
  confidence: number;
  isUnknown: boolean;
  unknownScore: number;
  srcIp: string;
  dstIp: string;
  protocol: string;
}

const IP_POOL = {
  internal: ['192.168.1.10', '192.168.1.25', '192.168.1.50', '192.168.1.100'],
  external: ['10.0.0.5', '172.16.0.30', '203.0.113.99', '198.51.100.20'],
};
const PROTOCOLS = ['TCP', 'UDP', 'HTTP', 'DNS', 'ICMP'];

const LiveCapturePanel: React.FC = () => {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<DetectionItem[]>([]);
  const [stats, setStats] = useState({ total: 0, benign: 0, anomaly: 0, unknown: 0 });
  const [message, setMessage] = useState('Ready');
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState(2);
  const nextId = useRef(1);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load real feature samples from backend
  const [featurePool, setFeaturePool] = useState<number[][]>([]);

  useEffect(() => {
    // Generate synthetic feature vectors (78-dim, matching CICIDS input)
    const generateFeatures = (type: 'normal' | 'ddos' | 'portscan' | 'brute') => {
      const base = Array.from({ length: 78 }, () => Math.random() * 0.5);
      if (type === 'ddos') {
        // High traffic volume patterns
        for (let i = 0; i < 10; i++) base[i] = 0.6 + Math.random() * 0.4;
        base[4] = 0.8 + Math.random() * 0.2; // src_bytes high
        base[5] = 0.8 + Math.random() * 0.2; // dst_bytes high
      } else if (type === 'portscan') {
        // Many connections, small packets
        for (let i = 20; i < 35; i++) base[i] = 0.5 + Math.random() * 0.5;
        base[4] = Math.random() * 0.3; // small packets
        base[5] = Math.random() * 0.3;
      } else if (type === 'brute') {
        // Repeated auth patterns
        base[10] = 0.7 + Math.random() * 0.3; // num_failed_logins
        base[11] = 0.1; // logged_in = false
        base[14] = Math.random() * 0.2;
      }
      return base;
    };

    const pool: number[][] = [];
    // Mix of different traffic types
    for (let i = 0; i < 30; i++) {
      const r = Math.random();
      if (r < 0.3) pool.push(generateFeatures('normal'));
      else if (r < 0.5) pool.push(generateFeatures('ddos'));
      else if (r < 0.7) pool.push(generateFeatures('portscan'));
      else pool.push(generateFeatures('brute'));
    }
    setFeaturePool(pool);
  }, []);

  const pickRandom = () => {
    const feat = featurePool[Math.floor(Math.random() * featurePool.length)];
    const src = IP_POOL.external[Math.floor(Math.random() * IP_POOL.external.length)];
    const dst = IP_POOL.internal[Math.floor(Math.random() * IP_POOL.internal.length)];
    const proto = PROTOCOLS[Math.floor(Math.random() * PROTOCOLS.length)];
    return { features: feat, srcIp: src, dstIp: dst, protocol: proto };
  };

  const runDetection = async () => {
    if (featurePool.length === 0) return;

    try {
      const { features, srcIp, dstIp, protocol } = pickRandom();

      const resp = await axios.post('/api/detect', {
        features,
        flow_info: { src_ip: srcIp, dst_ip: dstIp, protocol, packet_length: Math.floor(Math.random() * 1500) + 60 },
      });

      if (resp.data?.success) {
        const det = resp.data.data.detection;
        const item: DetectionItem = {
          id: nextId.current++,
          timestamp: new Date().toISOString(),
          prediction: det.prediction,
          confidence: det.class_confidence,
          isUnknown: det.is_unknown,
          unknownScore: det.unknown_prob || 0,
          srcIp,
          dstIp,
          protocol,
        };

        setResults(prev => [item, ...prev].slice(0, 200));
        setStats(prev => {
          const s = { ...prev, total: prev.total + 1 };
          if (det.is_unknown) s.unknown++;
          else if (det.prediction === 'BENIGN') s.benign++;
          else s.anomaly++;
          return s;
        });
        setMessage(`Last: ${det.prediction} (${(det.class_confidence * 100).toFixed(0)}%)`);
        setError(null);
      }
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Detection failed');
    }
  };

  const toggleRunning = () => {
    if (running) {
      if (timerRef.current) clearInterval(timerRef.current);
      setRunning(false);
      setMessage('Stopped');
    } else {
      setRunning(true);
      setMessage('Running...');
      // Run immediately, then at interval
      runDetection();
      timerRef.current = setInterval(runDetection, 3000 / speed);
    }
  };

  useEffect(() => {
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  return (
    <div>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">Live Detection</div>
        <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 16 }}>
          Simulated live traffic detection. Samples are generated from realistic attack patterns and processed through the full pipeline.
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
          <button className={`btn ${running ? 'btn-danger' : 'btn-primary'}`} onClick={toggleRunning}>
            {running ? 'Stop' : 'Start Detection'}
          </button>
          <select value={speed} onChange={e => setSpeed(Number(e.target.value))} style={{ width: 'auto', fontSize: 12 }}>
            <option value={1}>Slow (3s)</option>
            <option value={2}>Normal (1.5s)</option>
            <option value={5}>Fast (0.6s)</option>
          </select>
          <span style={{ fontSize: 12, color: '#6b7280' }}>
            Status: <span style={{ color: running ? '#4ade80' : '#6b7280' }}>{message}</span>
          </span>
        </div>

        {error && <div style={{ color: '#f87171', fontSize: 12, marginBottom: 12 }}>{error}</div>}

        <div className="stats-row" style={{ marginBottom: 0 }}>
          <div className="stat-card">
            <div className="stat-value">{stats.total}</div>
            <div className="stat-label">Total</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ color: '#4ade80' }}>{stats.benign}</div>
            <div className="stat-label">Benign</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ color: '#f87171' }}>{stats.anomaly}</div>
            <div className="stat-label">Known Attack</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ color: '#facc15' }}>{stats.unknown}</div>
            <div className="stat-label">Unknown</div>
          </div>
        </div>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Source</th>
              <th>Dest</th>
              <th>Proto</th>
              <th>Prediction</th>
              <th>Confidence</th>
              <th>Unknown Score</th>
            </tr>
          </thead>
          <tbody>
            {results.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', color: '#4b5563', padding: 32 }}>
                  {running ? 'Waiting for detections...' : 'Click Start Detection to begin'}
                </td>
              </tr>
            ) : (
              results.slice(0, 50).map(r => (
                <tr key={r.id}>
                  <td style={{ color: '#6b7280', fontSize: 12 }}>{r.timestamp.split('T')[1]?.split('.')[0] || r.timestamp}</td>
                  <td style={{ color: '#60a5fa', fontFamily: 'monospace', fontSize: 12 }}>{r.srcIp}</td>
                  <td style={{ color: '#60a5fa', fontFamily: 'monospace', fontSize: 12 }}>{r.dstIp}</td>
                  <td style={{ fontSize: 12 }}>{r.protocol}</td>
                  <td>
                    <span className={`tag ${r.isUnknown ? 'tag-yellow' : r.prediction === 'BENIGN' ? 'tag-green' : 'tag-red'}`}>
                      {r.prediction}
                    </span>
                  </td>
                  <td style={{ fontSize: 12 }}>{(r.confidence * 100).toFixed(0)}%</td>
                  <td style={{ fontSize: 12, color: r.unknownScore > 0.5 ? '#f87171' : '#9ca3af' }}>
                    {(r.unknownScore * 100).toFixed(1)}%
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default LiveCapturePanel;
