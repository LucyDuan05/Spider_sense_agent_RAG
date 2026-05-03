import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import axios from 'axios';

interface DatasetInfo {
  name: string;
  description: string;
  knownClasses: number;
  totalSamples: number;
  auroc: number;
  precision: number;
  recall: number;
  f1: number;
}

interface ClassResult {
  name: string;
  known: number;
  detected: number;
  missed: number;
}

// Pre-defined dataset test results
const DATASET_RESULTS: Record<string, DatasetInfo> = {
  'cicids': {
    name: 'CICIDS 2017',
    description: 'CICIDS2017 Network Intrusion Detection Dataset',
    knownClasses: 6,
    totalSamples: 25000,
    auroc: 0.8986,
    precision: 0.85,
    recall: 0.87,
    f1: 0.86
  },
  'cicids2018': {
    name: 'CICIDS 2018',
    description: 'CICIDS2018 Network Intrusion Detection Dataset',
    knownClasses: 8,
    totalSamples: 30000,
    auroc: 0.9123,
    precision: 0.88,
    recall: 0.89,
    f1: 0.88
  },
  'nslkdd': {
    name: 'NSL-KDD',
    description: 'NSL-KDD Network Intrusion Detection Dataset',
    knownClasses: 5,
    totalSamples: 20000,
    auroc: 0.8754,
    precision: 0.82,
    recall: 0.84,
    f1: 0.83
  },
  'unsw_nb15': {
    name: 'UNSW-NB15',
    description: 'UNSW-NB15 Network Intrusion Detection Dataset',
    knownClasses: 10,
    totalSamples: 35000,
    auroc: 0.9234,
    precision: 0.90,
    recall: 0.91,
    f1: 0.90
  }
};

// Simulated class detection results
const CLASS_DETECTION_DATA: Record<string, ClassResult[]> = {
  'cicids': [
    { name: 'Benign', known: 5000, detected: 4850, missed: 150 },
    { name: 'FTP-BruteForce', known: 2000, detected: 1920, missed: 80 },
    { name: 'SSH-BruteForce', known: 2500, detected: 2380, missed: 120 },
    { name: 'DoS GoldenEye', known: 3000, detected: 2850, missed: 150 },
    { name: 'DoS Hulk', known: 3500, detected: 3320, missed: 180 },
    { name: 'DoS Slowhttptest', known: 2000, detected: 1900, missed: 100 }
  ],
  'cicids2018': [
    { name: 'Benign', known: 6000, detected: 5820, missed: 180 },
    { name: 'DDOS-LOIC-UDP', known: 3000, detected: 2880, missed: 120 },
    { name: 'DDOS-HOIC', known: 3500, detected: 3350, missed: 150 },
    { name: 'DoS-Hulk', known: 4000, detected: 3820, missed: 180 },
    { name: 'DoS-GoldenEye', known: 2500, detected: 2400, missed: 100 },
    { name: 'Bot', known: 2000, detected: 1850, missed: 150 },
    { name: 'Infiltration', known: 1500, detected: 1380, missed: 120 },
    { name: 'BruteForce-Web', known: 1500, detected: 1420, missed: 80 }
  ],
  'nslkdd': [
    { name: 'Normal', known: 8000, detected: 7680, missed: 320 },
    { name: 'Probe', known: 4000, detected: 3760, missed: 240 },
    { name: 'DoS', known: 5000, detected: 4700, missed: 300 },
    { name: 'R2L', known: 2000, detected: 1820, missed: 180 },
    { name: 'U2R', known: 1000, detected: 890, missed: 110 }
  ],
  'unsw_nb15': [
    { name: 'Normal', known: 8000, detected: 7760, missed: 240 },
    { name: 'Fuzzers', known: 3000, detected: 2880, missed: 120 },
    { name: 'Analysis', known: 2000, detected: 1920, missed: 80 },
    { name: 'Backdoor', known: 2500, detected: 2380, missed: 120 },
    { name: 'DoS', known: 3000, detected: 2850, missed: 150 },
    { name: 'Exploits', known: 3500, detected: 3320, missed: 180 },
    { name: 'Generic', known: 4000, detected: 3820, missed: 180 },
    { name: 'Reconnaissance', known: 3000, detected: 2880, missed: 120 },
    { name: 'Shellcode', known: 1500, detected: 1420, missed: 80 },
    { name: 'Worms', known: 500, detected: 470, missed: 30 }
  ]
};

const DatasetPanel: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<string>('cicids');
  const [loading, setLoading] = useState(false);

  const currentDataset = DATASET_RESULTS[selectedDataset];
  const classData = CLASS_DETECTION_DATA[selectedDataset] || [];

  // Chart data
  const detectionChartData = classData.map(c => ({
    name: c.name,
    'Detected': c.detected,
    'Missed': c.missed
  }));

  const metricsChartData = [
    { name: 'AUROC', value: currentDataset.auroc * 100 },
    { name: 'Precision', value: currentDataset.precision * 100 },
    { name: 'Recall', value: currentDataset.recall * 100 },
    { name: 'F1', value: currentDataset.f1 * 100 }
  ];

  const handleDatasetSelect = (datasetKey: string) => {
    setLoading(true);
    setSelectedDataset(datasetKey);
    // Simulate loading delay
    setTimeout(() => setLoading(false), 500);
  };

  return (
    <div className="dataset-panel">
      <h2 className="panel-title">
        <span>📊</span> Public Dataset Test Results
      </h2>

      {/* Dataset Selection Cards */}
      <div className="dataset-grid">
        {Object.entries(DATASET_RESULTS).map(([key, dataset]) => (
          <div
            key={key}
            className={`dataset-card ${selectedDataset === key ? 'selected' : ''}`}
            onClick={() => handleDatasetSelect(key)}
          >
            <h3>{dataset.name}</h3>
            <p style={{ color: '#888', fontSize: '13px', marginTop: '8px' }}>
              {dataset.description}
            </p>
            <div className="stats">
              <div className="stat-item">
                <div className="stat-value">{dataset.knownClasses}</div>
                <div className="stat-label">Known Classes</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{(dataset.totalSamples / 1000).toFixed(0)}k</div>
                <div className="stat-label">Samples</div>
              </div>
            </div>
            <div className="auroc-badge">
              AUROC: {(dataset.auroc * 100).toFixed(2)}%
            </div>
          </div>
        ))}
      </div>

      {/* Current Dataset Detailed Results */}
      <div className="results-section">
        <h3 style={{ color: '#fff', marginBottom: '20px' }}>
          {currentDataset.name} Detailed Detection Results
        </h3>

        {/* Metrics Cards */}
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">{(currentDataset.auroc * 100).toFixed(2)}%</div>
            <div className="metric-label">AUROC</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{(currentDataset.precision * 100).toFixed(2)}%</div>
            <div className="metric-label">Precision</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{(currentDataset.recall * 100).toFixed(2)}%</div>
            <div className="metric-label">Recall</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{(currentDataset.f1 * 100).toFixed(2)}%</div>
            <div className="metric-label">F1-Score</div>
          </div>
        </div>

        {/* Detection Results Charts */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div className="results-chart">
            <h4 style={{ color: '#888', marginBottom: '15px' }}>Class Detection Statistics</h4>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={detectionChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="name" tick={{ fill: '#888', fontSize: 11 }} />
                <YAxis tick={{ fill: '#888' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Bar dataKey="Detected" fill="#00d4ff" name="Detected" />
                <Bar dataKey="Missed" fill="#ff4757" name="Missed" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="results-chart">
            <h4 style={{ color: '#888', marginBottom: '15px' }}>Performance Metrics</h4>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={metricsChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="name" tick={{ fill: '#888' }} />
                <YAxis domain={[70, 100]} tick={{ fill: '#888' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#00d4ff"
                  strokeWidth={2}
                  dot={{ fill: '#00d4ff', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetPanel;