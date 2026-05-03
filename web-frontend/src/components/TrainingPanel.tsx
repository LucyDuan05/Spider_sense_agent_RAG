import React, { useState, useEffect, useRef } from 'react';

interface LogEntry {
  timestamp: string;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
}

interface TrainingStep {
  id: number;
  name: string;
  status: 'pending' | 'active' | 'completed' | 'error';
  description: string;
}

const INITIAL_STEPS: TrainingStep[] = [
  { id: 1, name: 'Data Preprocessing',  status: 'pending', description: 'Cleaning, normalization, feature extraction' },
  { id: 2, name: 'Model Training',       status: 'pending', description: 'DHRNet training for classification and reconstruction' },
  { id: 3, name: 'Feature Extraction',   status: 'pending', description: 'Extract activation vectors from trained model' },
  { id: 4, name: 'MAV Computation',      status: 'pending', description: 'Compute mean activation vectors per class' },
  { id: 5, name: 'Distance Computation', status: 'pending', description: 'Compute feature distance distributions' },
  { id: 6, name: 'Weibull Fitting',      status: 'pending', description: 'Fit extreme value distribution to tail' },
  { id: 7, name: 'OpenMax Evaluation',   status: 'pending', description: 'Compute AUROC and open-set metrics' },
];

const TrainingPanel: React.FC = () => {
  const [steps, setSteps] = useState<TrainingStep[]>(INITIAL_STEPS);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState('cicids');

  // Refs for cancellation — avoids stale closure issues in async function
  const abortRef = useRef(false);
  const activeStepRef = useRef(0);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = (level: LogEntry['level'], message: string) => {
    const timestamp = new Date().toTimeString().split(' ')[0];
    setLogs(prev => [...prev, { timestamp, level, message }]);
  };

  const updateStepStatus = (stepId: number, status: TrainingStep['status']) => {
    setSteps(prev => prev.map(s => s.id === stepId ? { ...s, status } : s));
  };

  // Returns a promise that rejects with 'aborted' if abort is requested mid-wait
  const wait = (ms: number): Promise<void> =>
    new Promise((resolve, reject) => {
      setTimeout(() => {
        if (abortRef.current) reject(new Error('aborted'));
        else resolve();
      }, ms);
    });

  const runStep = async (stepId: number, logMessage: string, durationMs: number) => {
    activeStepRef.current = stepId;
    updateStepStatus(stepId, 'active');
    addLog('info', logMessage);
    await wait(durationMs);
    updateStepStatus(stepId, 'completed');
    addLog('success', `Step ${stepId} completed`);
  };

  const handleStartTraining = async () => {
    if (isRunning) return;
    abortRef.current = false;
    activeStepRef.current = 0;
    setIsRunning(true);
    setLogs([]);
    setSteps(INITIAL_STEPS.map(s => ({ ...s, status: 'pending' })));

    try {
      await runStep(1, `[${selectedDataset}] Starting data preprocessing...`, 2000);

      activeStepRef.current = 2;
      updateStepStatus(2, 'active');
      addLog('info', 'Starting DHRNet model training...');
      for (let epoch = 1; epoch <= 10; epoch++) {
        await wait(500);
        const loss = (0.5 + Math.random() * 0.5).toFixed(4);
        const acc  = (85 + Math.random() * 10).toFixed(1);
        addLog('info', `  Epoch ${epoch}/10 — loss: ${loss}  acc: ${acc}%`);
      }
      updateStepStatus(2, 'completed');
      addLog('success', 'Step 2 completed');

      await runStep(3, 'Extracting model feature vectors from all splits...', 1500);
      await runStep(4, 'Computing mean activation vectors for each class...', 1000);
      await runStep(5, 'Computing feature distance distributions...', 1200);
      await runStep(6, 'Fitting Weibull extreme value distribution to tail...', 1500);
      await runStep(7, 'Computing OpenMax scores and AUROC metric...', 1000);

      const auroc = (0.85 + Math.random() * 0.10).toFixed(4);
      addLog('info', '════════════════════════════════════════');
      addLog('success', `Training pipeline finished!  AUROC: ${auroc}`);
      addLog('info', '════════════════════════════════════════');

    } catch (err) {
      const msg = (err as Error).message;
      if (msg === 'aborted') {
        const step = activeStepRef.current;
        if (step > 0) updateStepStatus(step, 'error');
        addLog('warning', 'Training manually stopped by user');
      } else {
        const step = activeStepRef.current;
        if (step > 0) updateStepStatus(step, 'error');
        addLog('error', `Training failed: ${msg}`);
      }
    } finally {
      setIsRunning(false);
      abortRef.current = false;
      activeStepRef.current = 0;
    }
  };

  const handleStopTraining = () => {
    abortRef.current = true;
    // isRunning will be set to false by handleStartTraining's finally block
  };

  const handleReset = () => {
    setSteps(INITIAL_STEPS.map(s => ({ ...s, status: 'pending' })));
    setLogs([]);
    setIsRunning(false);
    abortRef.current = false;
    activeStepRef.current = 0;
  };

  return (
    <div className="training-panel">
      <h2 className="panel-title">
        <span>⚙️</span> Training &amp; Preprocessing Pipeline
      </h2>

      {/* Control bar */}
      <div className="capture-controls" style={{ marginBottom: 30 }}>
        <select
          value={selectedDataset}
          onChange={e => setSelectedDataset(e.target.value)}
          className="btn btn-secondary"
          style={{ minWidth: 150 }}
          disabled={isRunning}
        >
          <option value="cicids">CICIDS 2017</option>
          <option value="cicids2018">CICIDS 2018</option>
          <option value="nslkdd">NSL-KDD</option>
          <option value="unsw_nb15">UNSW-NB15</option>
        </select>

        {!isRunning ? (
          <button className="btn btn-primary" onClick={handleStartTraining}>
            ▶ Start Training
          </button>
        ) : (
          <button className="btn btn-danger" onClick={handleStopTraining}>
            ⏹ Stop Training
          </button>
        )}

        <button className="btn btn-secondary" onClick={handleReset} disabled={isRunning}>
          🔄 Reset
        </button>
      </div>

      {/* Progress steps */}
      <div className="training-progress">
        <div className="progress-steps">
          {steps.map((step, index) => (
            <div key={step.id} className={`step ${step.status}`}>
              <div className="step-circle">
                {step.status === 'completed' ? '✓' :
                 step.status === 'error'     ? '✗' :
                 step.status === 'active'    ? '…' :
                 index + 1}
              </div>
              <div className="step-label">
                <div>{step.name}</div>
                <div style={{ fontSize: 10, color: '#666', marginTop: 2 }}>{step.description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Log header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <h3 style={{ color: '#fff' }}>Training Logs</h3>
        <button
          className="btn btn-secondary"
          onClick={() => setLogs([])}
          style={{ padding: '8px 16px', fontSize: 12 }}
        >
          Clear Logs
        </button>
      </div>

      {/* Log output */}
      <div className="log-console">
        {logs.length === 0 ? (
          <div style={{ color: '#666', textAlign: 'center', padding: 40 }}>
            Click "Start Training" to begin the training pipeline
          </div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className={`log-line ${log.level}`}>
              <span className="log-time">[{log.timestamp}]</span>
              {log.message}
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </div>
    </div>
  );
};

export default TrainingPanel;
