import React, { useState, useCallback } from 'react';
import axios from 'axios';

interface BatchResult {
  timestamp: string;
  flow_info: any;
  detection: {
    prediction: string;
    class_confidence: number;
    is_unknown: boolean;
    is_anomaly: boolean;
  };
  pipeline_time_ms: number;
}

const FileUpload: React.FC = () => {
  const [csvText, setCsvText] = useState('');
  const [results, setResults] = useState<BatchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      setCsvText(ev.target?.result as string || '');
    };
    reader.readAsText(file);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      setCsvText(ev.target?.result as string || '');
    };
    reader.readAsText(file);
  }, []);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setProgress(0);
    try {
      // Parse CSV (comma-separated values per line)
      const lines = csvText.trim().split('\n');
      const samples: any[] = [];

      for (const line of lines) {
        const values = line.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
        if (values.length >= 10) {
          samples.push({ features: values, flow_info: {} });
        }
      }

      if (samples.length === 0) {
        setError('未找到有效数据行。请确保每行至少包含10个逗号分隔的数值。');
        setLoading(false);
        return;
      }

      // Process in batches of 10
      const batchSize = 10;
      const allResults: BatchResult[] = [];

      for (let i = 0; i < samples.length; i += batchSize) {
        const batch = samples.slice(i, i + batchSize);
        const resp = await axios.post('/api/detect/batch', { samples: batch });
        if (resp.data?.success) {
          allResults.push(...resp.data.data);
        }
        setProgress(Math.round(((i + batch.length) / samples.length) * 100));
      }

      setResults(allResults);
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || '批量分析失败');
    } finally {
      setLoading(false);
    }
  };

  const stats = {
    total: results.length,
    benign: results.filter(r => r.detection?.prediction === 'BENIGN').length,
    anomaly: results.filter(r => r.detection?.is_anomaly).length,
    unknown: results.filter(r => r.detection?.is_unknown).length,
  };

  return (
    <div>
      <h2 className="panel-title">
        <span>📁</span> File Batch Detection
      </h2>
      <p style={{ color: '#888', marginBottom: 20 }}>
        上传CSV/PCAP特征文件进行批量检测分析
      </p>

      {/* Upload area */}
      <div
        onDragOver={e => e.preventDefault()}
        onDrop={handleFileDrop}
        style={{
          padding: '40px 20px',
          border: '2px dashed rgba(255,255,255,0.2)',
          borderRadius: 12,
          textAlign: 'center',
          marginBottom: 20,
          background: 'rgba(255,255,255,0.02)',
          cursor: 'pointer',
          transition: 'border-color 0.3s',
        }}
      >
        <div style={{ fontSize: 36, marginBottom: 12 }}>📂</div>
        <div style={{ color: '#aaa', marginBottom: 8 }}>
          拖放CSV文件到此处，或点击选择文件
        </div>
        <input
          type="file"
          accept=".csv,.txt"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          id="file-upload-input"
        />
        <label htmlFor="file-upload-input" className="btn btn-secondary" style={{ cursor: 'pointer' }}>
          选择文件
        </label>
      </div>

      {/* Text input fallback */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ color: '#aaa', fontSize: 12, marginBottom: 6 }}>
          或直接粘贴CSV数据 (每行一个样本，逗号分隔特征值)
        </div>
        <textarea
          value={csvText}
          onChange={e => setCsvText(e.target.value)}
          placeholder={`1.2,3.4,0.5,2.1,1.0,0.3,0.8,0.2,0.1,0.6,0.4,0.7,0.9,0.3,0.5,0.8,...\n0.3,1.2,0.8,0.1,0.5,0.2,0.7,0.4,0.6,0.3,0.9,0.1,0.4,0.2,0.8,0.5,...`}
          rows={8}
          style={{
            width: '100%',
            background: 'rgba(0,0,0,0.3)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 6,
            padding: '10px 14px',
            color: '#e0e0e0',
            fontFamily: 'monospace',
            fontSize: 11,
            resize: 'vertical',
          }}
        />
      </div>

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 20 }}>
        <button
          className="btn btn-primary"
          onClick={handleAnalyze}
          disabled={loading || !csvText.trim()}
        >
          {loading ? `🔄 分析中... ${progress}%` : '🔬 开始批量分析'}
        </button>
        {csvText.trim() && (
          <span style={{ color: '#666', fontSize: 12 }}>
            预估样本数: {csvText.trim().split('\n').filter(l => l.includes(',')).length}
          </span>
        )}
      </div>

      {error && (
        <div style={{ color: '#ff4757', marginBottom: 16, padding: '10px 14px', background: 'rgba(255,71,87,0.1)', borderRadius: 8 }}>
          ⚠ {error}
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div>
          {/* Summary */}
          <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4,1fr)', marginBottom: 20 }}>
            <div className="metric-card">
              <div className="metric-value">{stats.total}</div>
              <div className="metric-label">总样本</div>
            </div>
            <div className="metric-card" style={{ borderTop: '2px solid #00ff88' }}>
              <div className="metric-value" style={{ color: '#00ff88' }}>{stats.benign}</div>
              <div className="metric-label">正常</div>
            </div>
            <div className="metric-card" style={{ borderTop: '2px solid #ff4757' }}>
              <div className="metric-value" style={{ color: '#ff4757' }}>{stats.anomaly}</div>
              <div className="metric-label">异常</div>
            </div>
            <div className="metric-card" style={{ borderTop: '2px solid #ffc107' }}>
              <div className="metric-value" style={{ color: '#ffc107' }}>{stats.unknown}</div>
              <div className="metric-label">未知攻击</div>
            </div>
          </div>

          {/* Table */}
          <div className="packet-list">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  {['#', 'Prediction', 'Confidence', 'Is Unknown', 'Is Anomaly', 'Time (ms)'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#888', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.slice(0, 100).map((r, i) => (
                  <tr key={i} style={{
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                    background: r.detection?.is_anomaly ? 'rgba(255,71,87,0.05)' : 'transparent',
                  }}>
                    <td style={{ padding: '6px 10px', color: '#666' }}>{i + 1}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: 11,
                        background: r.detection?.is_unknown ? 'rgba(255,193,7,0.2)' :
                                     r.detection?.is_anomaly ? 'rgba(255,71,87,0.2)' :
                                     'rgba(46,213,115,0.2)',
                        color: r.detection?.is_unknown ? '#ffc107' :
                                r.detection?.is_anomaly ? '#ff4757' : '#2ed573',
                      }}>
                        {r.detection?.prediction || 'N/A'}
                      </span>
                    </td>
                    <td style={{ padding: '6px 10px', color: '#aaa' }}>
                      {r.detection?.class_confidence ? (r.detection.class_confidence * 100).toFixed(1) + '%' : 'N/A'}
                    </td>
                    <td style={{ padding: '6px 10px', color: r.detection?.is_unknown ? '#ff4757' : '#2ed573' }}>
                      {r.detection?.is_unknown ? '⚠ Yes' : 'No'}
                    </td>
                    <td style={{ padding: '6px 10px', color: r.detection?.is_anomaly ? '#ff4757' : '#2ed573' }}>
                      {r.detection?.is_anomaly ? '⚠ Yes' : 'No'}
                    </td>
                    <td style={{ padding: '6px 10px', color: '#666' }}>
                      {r.pipeline_time_ms?.toFixed(1)}ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {results.length > 100 && (
              <div style={{ padding: '12px', textAlign: 'center', color: '#666', fontSize: 12 }}>
                显示前100条，共 {results.length} 条结果
              </div>
            )}
          </div>
        </div>
      )}

      {results.length === 0 && !loading && (
        <div style={{ textAlign: 'center', padding: 40, color: '#666' }}>
          <div style={{ fontSize: 36, marginBottom: 12 }}>📁</div>
          <div style={{ fontSize: 14 }}>上传文件或粘贴CSV数据开始批量分析</div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
