import React, { useState } from 'react';
import axios from 'axios';

interface AgentOpinion {
  agent: string;
  role: string;
  verdict: string;
  confidence: number;
  evidence: string[];
  counter_evidence: string[];
  analysis: string;
  llm_model?: string;
}

interface DebateResult {
  verdict: string;
  vote_count: string;
  action: string;
  recommendation: string;
  detector: AgentOpinion;
  analyst: AgentOpinion;
  arbiter: AgentOpinion;
  debate_mode: string;
  timestamp: string;
}

const VERDICT_STYLE: Record<string, { label: string; cls: string }> = {
  CONFIRMED_THREAT: { label: 'Confirmed Threat', cls: 'tag-red' },
  LIKELY_THREAT: { label: 'Likely Threat', cls: 'tag-yellow' },
  UNCERTAIN: { label: 'Uncertain', cls: 'tag-yellow' },
  LIKELY_BENIGN: { label: 'Likely Benign', cls: 'tag-green' },
};

const SAMPLES: Record<string, { features: string; info: string }> = {
  ddos: {
    features: Array.from({ length: 78 }, () => (Math.random() * 0.4 + 0.3).toFixed(3)).join(','),
    info: '{"src_ip":"10.0.0.99","dst_ip":"192.168.1.1","protocol":"UDP","packet_length":1500}',
  },
  portscan: {
    features: Array.from({ length: 78 }, () => (Math.random() * 0.3 + 0.1).toFixed(3)).join(','),
    info: '{"src_ip":"172.16.0.50","dst_ip":"192.168.1.1","protocol":"TCP","packet_length":60}',
  },
};

const AgentPanel: React.FC = () => {
  const [features, setFeatures] = useState('');
  const [flowInfo, setFlowInfo] = useState('{}');
  const [debateResult, setDebateResult] = useState<DebateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeSample, setActiveSample] = useState<string | null>(null);

  const handleSample = (type: string) => {
    const s = SAMPLES[type];
    if (s) {
      setFeatures(s.features);
      setFlowInfo(s.info);
      setActiveSample(type);
    }
  };

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const featArray = features.split(',').map(f => parseFloat(f.trim())).filter(f => !isNaN(f));
      const flowObj = JSON.parse(flowInfo);

      // Detect
      const detResp = await axios.post('/api/detect', { features: featArray, flow_info: flowObj });
      if (!detResp.data?.success) { setError(detResp.data?.error || 'Detection failed'); setLoading(false); return; }
      const detection = detResp.data.data.detection;

      // Debate
      const debResp = await axios.post('/api/debate', { detection, flow_info: flowObj, features: featArray });
      if (debResp.data?.success) {
        setDebateResult(debResp.data.data.debate);
      } else {
        setError(debResp.data?.error || 'Debate failed');
      }
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  const renderAgent = (opinion: AgentOpinion) => (
    <div className="debate-card">
      <div className="role">{opinion.role}</div>
      <div style={{ marginBottom: 10 }}>
        <span className={`tag ${VERDICT_STYLE[opinion.verdict]?.cls || 'tag-blue'}`}>
          {VERDICT_STYLE[opinion.verdict]?.label || opinion.verdict}
        </span>
        <span style={{ fontSize: 12, color: '#6b7280', marginLeft: 8 }}>
          {(opinion.confidence * 100).toFixed(0)}% confidence
        </span>
      </div>
      {opinion.evidence.length > 0 && (
        <div className="evidence">
          <div style={{ color: '#4ade80', fontSize: 11, fontWeight: 600, marginBottom: 4 }}>Evidence</div>
          <ul style={{ paddingLeft: 16 }}>
            {opinion.evidence.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </div>
      )}
      {opinion.counter_evidence?.length > 0 && (
        <div className="evidence">
          <div style={{ color: '#f87171', fontSize: 11, fontWeight: 600, marginBottom: 4 }}>Counter</div>
          <ul style={{ paddingLeft: 16 }}>
            {opinion.counter_evidence.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </div>
      )}
      <div style={{ fontSize: 12, color: '#9ca3af', lineHeight: 1.5, marginTop: 10 }}>
        {opinion.analysis}
      </div>
    </div>
  );

  return (
    <div>
      {/* Input */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">Sample Selection</div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
          <button className={`btn ${activeSample === 'ddos' ? 'btn-primary' : ''}`} onClick={() => handleSample('ddos')}>DDoS Attack</button>
          <button className={`btn ${activeSample === 'portscan' ? 'btn-primary' : ''}`} onClick={() => handleSample('portscan')}>Port Scan</button>
        </div>
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>Feature Vector</div>
          <textarea value={features} onChange={e => { setFeatures(e.target.value); setActiveSample(null); }} rows={2} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <button className="btn btn-primary" onClick={handleRun} disabled={loading || !features}>
            {loading ? 'Running...' : 'Detect & Debate'}
          </button>
          {error && <span style={{ color: '#f87171', fontSize: 12 }}>{error}</span>}
        </div>
      </div>

      {/* Results */}
      {debateResult && (
        <div>
          <div className="card" style={{ marginBottom: 16, borderLeft: `3px solid ${debateResult.verdict === 'MALICIOUS' ? '#ef4444' : debateResult.verdict === 'SUSPICIOUS' ? '#eab308' : '#22c55e'}` }}>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#e5e7eb', marginBottom: 8 }}>
              {debateResult.verdict === 'MALICIOUS' ? 'Malicious' : debateResult.verdict === 'SUSPICIOUS' ? 'Suspicious' : 'Benign'}
              <span style={{ fontSize: 12, color: '#6b7280', fontWeight: 400, marginLeft: 10 }}>{debateResult.vote_count}</span>
            </div>
            <div style={{ fontSize: 13, color: '#9ca3af', marginBottom: 8 }}>{debateResult.action}</div>
            <div style={{ fontSize: 13, color: '#d1d5db', lineHeight: 1.6 }}>{debateResult.recommendation}</div>
            <div style={{ fontSize: 11, color: '#4b5563', marginTop: 8 }}>
              Mode: {debateResult.debate_mode}
              {debateResult?.detector?.llm_model && (
                <span style={{ marginLeft: 8, color: '#00d4ff' }}>
                  · LLM: {debateResult.detector.llm_model}
                </span>
              )}
            </div>
          </div>
          <div className="debate-grid">
            {renderAgent(debateResult.detector)}
            {renderAgent(debateResult.analyst)}
            {renderAgent(debateResult.arbiter)}
          </div>
        </div>
      )}

      {!debateResult && !loading && (
        <div style={{ textAlign: 'center', padding: 48, color: '#4b5563', fontSize: 13 }}>
          Select a sample and run detection to see multi-agent debate analysis
        </div>
      )}
    </div>
  );
};

export default AgentPanel;
