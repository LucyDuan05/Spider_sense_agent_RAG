import React, { useState } from 'react';
import './App.css';
import DatasetPanel from './components/DatasetPanel';
import LiveCapturePanel from './components/LiveCapturePanel';
import TrainingPanel from './components/TrainingPanel';

type TabType = 'dataset' | 'live' | 'training';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dataset');

  return (
    <div className="app">
      <header className="app-header">
        <h1>🕸️ 蜘蛛感应 <span style={{ fontSize: '18px', color: '#88ddff', fontWeight: 400 }}>Spider-Sense IDS</span></h1>
        <p className="subtitle">开放集识别网络入侵检测系统 · Open-Set Recognition based Network Intrusion Detection</p>
      </header>

      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'dataset' ? 'active' : ''}`}
          onClick={() => setActiveTab('dataset')}
        >
          Dataset Results
        </button>
        <button
          className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          Live Capture
        </button>
        <button
          className={`tab-btn ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          Training Pipeline
        </button>
      </nav>

      <main className="app-content">
        {activeTab === 'dataset' && <DatasetPanel />}
        {activeTab === 'live' && <LiveCapturePanel />}
        {activeTab === 'training' && <TrainingPanel />}
      </main>

      <footer className="app-footer">
        <p>CROSR Open-Set Recognition | Network Intrusion Detection System v1.0</p>
      </footer>
    </div>
  );
}

export default App;


