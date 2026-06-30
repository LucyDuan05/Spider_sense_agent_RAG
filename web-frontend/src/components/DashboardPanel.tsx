import React, { useEffect, useState } from 'react';
import axios from 'axios';

interface Props {
  engineStats: any;
}

const DashboardPanel: React.FC<Props> = ({ engineStats }) => {
  const [health, setHealth] = useState<any>(null);
  const [datasetResults, setDatasetResults] = useState<any>(null);

  useEffect(() => {
    axios.get('/api/health').then(r => setHealth(r.data)).catch(() => {});
    axios.get('/api/datasets').then(r => setDatasetResults(r.data.data)).catch(() => {});
  }, []);

  return (
    <div>
      {/* Status bar */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-value" style={{ color: health?.status === 'healthy' ? '#4ade80' : '#f87171' }}>
            {health?.status === 'healthy' ? 'Online' : 'Offline'}
          </div>
          <div className="stat-label">System Status</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            {health?.components?.engine?.num_classes || '-'}
          </div>
          <div className="stat-label">Known Classes</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            {health?.components?.agent?.mode === 'llm' ? 'LLM' : 'Rule'}
          </div>
          <div className="stat-label">Agent Mode</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            {health?.components?.rag?.num_patterns || '-'}
          </div>
          <div className="stat-label">Knowledge Patterns</div>
        </div>
      </div>

      {/* Core capabilities */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">Detection Pipeline</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr auto 1fr auto 1fr', gap: 8, alignItems: 'center', fontSize: 12 }}>
          <div style={{ textAlign: 'center', padding: '12px 8px', background: '#0f1117', borderRadius: 6 }}>
            <div style={{ fontWeight: 600, color: '#e5e7eb' }}>DHRNet</div>
            <div style={{ color: '#6b7280', marginTop: 4 }}>Feature Extraction</div>
          </div>
          <div style={{ color: '#374151' }}>→</div>
          <div style={{ textAlign: 'center', padding: '12px 8px', background: '#0f1117', borderRadius: 6 }}>
            <div style={{ fontWeight: 600, color: '#e5e7eb' }}>Weibull OpenMax</div>
            <div style={{ color: '#6b7280', marginTop: 4 }}>Open-Set Scoring</div>
          </div>
          <div style={{ color: '#374151' }}>→</div>
          <div style={{ textAlign: 'center', padding: '12px 8px', background: '#0f1117', borderRadius: 6 }}>
            <div style={{ fontWeight: 600, color: '#e5e7eb' }}>3-Agent Debate</div>
            <div style={{ color: '#6b7280', marginTop: 4 }}>Verification</div>
          </div>
          <div style={{ color: '#374151' }}>→</div>
          <div style={{ textAlign: 'center', padding: '12px 8px', background: '#0f1117', borderRadius: 6 }}>
            <div style={{ fontWeight: 600, color: '#e5e7eb' }}>Verdict</div>
            <div style={{ color: '#6b7280', marginTop: 4 }}>Actionable Output</div>
          </div>
        </div>
      </div>

      {/* Dataset results */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">Dataset Performance (AUROC)</div>
        {datasetResults ? (
          <table>
            <thead>
              <tr>
                <th>Dataset</th>
                <th>AUROC</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Classes</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(datasetResults).map(([key, d]: [string, any]) => (
                <tr key={key}>
                  <td style={{ fontWeight: 500 }}>{d.name}</td>
                  <td>
                    <span className={`tag ${d.auroc >= 0.85 ? 'tag-green' : d.auroc >= 0.7 ? 'tag-yellow' : 'tag-red'}`}>
                      {(d.auroc * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td>{(d.precision * 100).toFixed(1)}%</td>
                  <td>{(d.recall * 100).toFixed(1)}%</td>
                  <td>{(d.f1 * 100).toFixed(1)}%</td>
                  <td style={{ color: '#6b7280' }}>{d.knownClasses}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ color: '#6b7280', fontSize: 13 }}>Loading...</div>
        )}
      </div>

      {/* Innovation points */}
      <div className="card">
        <div className="card-header">Technical Approach</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {[
            { title: 'Open-Set Recognition', desc: 'DHRNet reconstruction features + Weibull OpenMax for unknown attack detection without libMR/Python 2.7 dependency.' },
            { title: 'Multi-Agent Verification', desc: 'Three independent agents (Detector, Analyst, Arbiter) debate each alert and vote for final verdict.' },
            { title: 'RAG Knowledge Base', desc: 'Vector similarity search over attack pattern library + MITRE ATT&CK technique mapping for context enrichment.' },
            { title: 'XAI Attribution', desc: 'Feature-level perturbation analysis reveals which traffic characteristics drive each detection decision.' },
          ].map(item => (
            <div key={item.title} style={{ padding: 12, background: '#0f1117', borderRadius: 6 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#e5e7eb', marginBottom: 4 }}>{item.title}</div>
              <div style={{ fontSize: 12, color: '#6b7280', lineHeight: 1.5 }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DashboardPanel;
