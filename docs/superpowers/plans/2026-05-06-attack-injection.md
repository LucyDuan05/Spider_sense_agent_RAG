# Attack Injection for Live Capture Demo - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "Attack Simulator" to LiveCapturePanel that injects crafted attack packets to loopback via scapy, enabling meaningful demo of attack detection in course/graduation project scenarios.

**Architecture:** Frontend button → POST `/api/capture/inject` → backend uses scapy to send crafted packets to 127.0.0.1 → existing sniff on `lo` captures them → model classifies → results appear in existing packet table.

**Tech Stack:** React/TypeScript (frontend), Flask/Python (backend), scapy (packet crafting).

---

## File Structure

**Files to modify:**
- `web-frontend/backend/inference_engine.py` — add `inject_attack_packets()` method to `IDSInferenceEngine` class
- `web-frontend/backend/app.py` — add `/api/capture/inject` endpoint
- `web-frontend/src/components/LiveCapturePanel.tsx` — add Attack Simulator UI section
- `web-frontend/src/App.css` — add styles for attack simulator (if needed)

**Files to create:** None. Plan reuses existing files to match codebase patterns.

---

## Task 1: Add `inject_attack_packets` method to inference engine

**Files:**
- Modify: `web-frontend/backend/inference_engine.py` (end of `IDSInferenceEngine` class, before `run_detection`)

- [ ] **Step 1: Add the method**

Open `web-frontend/backend/inference_engine.py`. Find the `run_detection` method (around line 273). Insert the following method immediately **before** `def run_detection`:

```python
    def inject_attack_packets(self, attack_type, count=20):
        """Inject crafted attack packets to loopback for demo purposes.

        Packets are sent to 127.0.0.1 with attack-characteristic ports/flags/sizes
        so the existing sniff loop on `lo` captures and classifies them.
        """
        from scapy.all import IP, TCP, send
        import time

        templates = {
            'DDoS':        {'dport': 80,   'flags': 'S',  'size': 60},
            'PortScan':    {'dport': None, 'flags': 'S',  'size': 40},
            'FTP-Patator': {'dport': 21,   'flags': 'PA', 'size': 100},
            'SSH-Patator': {'dport': 22,   'flags': 'PA', 'size': 100},
            'DoS Hulk':    {'dport': 80,   'flags': 'PA', 'size': 1400},
        }

        tpl = templates.get(attack_type, templates['DDoS'])
        print(f"[Inject] Starting injection of {count} '{attack_type}' packets to 127.0.0.1")

        try:
            for i in range(count):
                dport = (i % 1024 + 1) if tpl['dport'] is None else tpl['dport']
                pkt = IP(src='127.0.0.1', dst='127.0.0.1') / TCP(
                    dport=dport, sport=40000 + (i % 10000), flags=tpl['flags']
                ) / (b'X' * tpl['size'])
                send(pkt, verbose=False)
                time.sleep(0.05)
            print(f"[Inject] Completed injection of {count} '{attack_type}' packets")
        except PermissionError:
            print(f"[Inject] ERROR: Permission denied. Run backend with sudo.")
        except Exception as exc:
            print(f"[Inject] ERROR: {exc}")
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('web-frontend/backend/inference_engine.py').read())"`
Expected: no output (syntax valid)

- [ ] **Step 3: Manually verify the method attaches to the class**

Run: `python -c "import sys; sys.path.insert(0, 'web-frontend/backend'); from inference_engine import IDSInferenceEngine; print(hasattr(IDSInferenceEngine, 'inject_attack_packets'))"`
Expected: `True`

---

## Task 2: Add `/api/capture/inject` endpoint

**Files:**
- Modify: `web-frontend/backend/app.py` (add new route after `/api/capture/stop`)

- [ ] **Step 1: Add the endpoint**

Open `web-frontend/backend/app.py`. Find the `stop_capture` function (around line 194-201). Insert the following endpoint **immediately after** the `stop_capture` function and **before** `@app.route('/api/capture/predict', ...)`:

```python
SUPPORTED_ATTACKS = {'DDoS', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'DoS Hulk'}


@app.route('/api/capture/inject', methods=['POST'])
def inject_attack():
    """Inject crafted attack packets to loopback for demo purposes."""
    err = _engine_guard()
    if err:
        return err

    data = request.json or {}
    attack_type = data.get('attack_type', 'DDoS')
    count = int(data.get('count', 20))

    if attack_type not in SUPPORTED_ATTACKS:
        return jsonify({
            'success': False,
            'error': f'不支持的攻击类型: {attack_type}. 可选: {sorted(SUPPORTED_ATTACKS)}'
        }), 400

    if not engine.is_running:
        return jsonify({
            'success': False,
            'error': '抓包未运行，请先在 lo 接口上启动抓包'
        }), 400

    if count < 1 or count > 200:
        return jsonify({'success': False, 'error': 'count 必须在 1-200 之间'}), 400

    thread = threading.Thread(target=engine.inject_attack_packets, args=(attack_type, count))
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'message': f'开始注入 {attack_type} ({count}包)',
        'attack_type': attack_type,
        'count': count,
        'timestamp': datetime.now().isoformat()
    })
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('web-frontend/backend/app.py').read())"`
Expected: no output

- [ ] **Step 3: Start backend (manual)**

User should run:
```bash
cd web-frontend/backend && sudo python app.py
```
Backend should start on port 5000 with no errors.

- [ ] **Step 4: Test endpoint returns 400 when capture not running**

Run:
```bash
curl -s -X POST http://localhost:5000/api/capture/inject \
  -H "Content-Type: application/json" \
  -d '{"attack_type":"DDoS","count":5}'
```
Expected: `{"success": false, "error": "抓包未运行..."}` with HTTP 400

- [ ] **Step 5: Test endpoint rejects unsupported attack types**

Start capture first:
```bash
curl -s -X POST http://localhost:5000/api/capture/start \
  -H "Content-Type: application/json" -d '{"interface":"lo"}'
```
Then:
```bash
curl -s -X POST http://localhost:5000/api/capture/inject \
  -H "Content-Type: application/json" \
  -d '{"attack_type":"InvalidType","count":5}'
```
Expected: `{"success": false, "error": "不支持的攻击类型..."}` with HTTP 400

- [ ] **Step 6: Test successful injection**

Capture still running on `lo`. Run:
```bash
curl -s -X POST http://localhost:5000/api/capture/inject \
  -H "Content-Type: application/json" \
  -d '{"attack_type":"DDoS","count":5}'
```
Expected: `{"success": true, "message": "开始注入 DDoS (5包)", ...}`

Then poll:
```bash
curl -s http://localhost:5000/api/capture/status | python -m json.tool
```
Expected: `totalCount` > 0, `recent` contains packets with `dst_ip: 127.0.0.1`

- [ ] **Step 7: Stop capture**

```bash
curl -s -X POST http://localhost:5000/api/capture/stop
```

---

## Task 3: Add Attack Simulator UI to LiveCapturePanel

**Files:**
- Modify: `web-frontend/src/components/LiveCapturePanel.tsx`

- [ ] **Step 1: Add attack types constant and state**

Open `web-frontend/src/components/LiveCapturePanel.tsx`. Find the `INTERFACES` constant (line 15):

```typescript
const INTERFACES = ['eth0', 'wlan0', 'any', 'en0']
```

Replace with:

```typescript
const INTERFACES = ['eth0', 'wlan0', 'any', 'en0', 'lo']

const ATTACK_TYPES = ['DDoS', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'DoS Hulk'] as const
type AttackType = typeof ATTACK_TYPES[number]
```

- [ ] **Step 2: Add injection state to component**

Find the component state declarations (around line 24-30). Add injection state right after the existing `useState` calls, before `nextIdRef`:

```typescript
  const [injecting, setInjecting] = useState<AttackType | null>(null)
  const [injectMessage, setInjectMessage] = useState<string>('')
```

- [ ] **Step 3: Add injection handler function**

Find `handleStop` function (around line 119-128). Add `handleInject` immediately after `handleStop`:

```typescript
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
      // Re-enable button after 2s (injection runs ~1s for 20 packets)
      setTimeout(() => setInjecting(null), 2000)
    }
  }
```

- [ ] **Step 4: Add Attack Simulator JSX section**

Find the "Stats" section (line 176-194, the `<div className="metrics-grid">` block). Insert the following JSX **immediately before** the stats grid:

```tsx
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
          💡 Tips: 使用攻击模拟器前，请先将接口切换为 <code style={{ background: 'rgba(0,0,0,0.3)', padding: '1px 5px', borderRadius: 3 }}>lo</code>，再点击 Start Capture。按钮在 lo+抓包运行时才可点击。
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
          {ATTACK_TYPES.map(attack => {
            const enabled = isCapturing && selectedInterface === 'lo' && injecting === null
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
                }}
              >
                {injecting === attack ? `注入中... ${attack}` : `💥 ${attack}`}
              </button>
            )
          })}
        </div>
        {injectMessage && (
          <div style={{ color: '#00d4ff', fontSize: 12, marginTop: 8 }}>
            {injectMessage}
          </div>
        )}
      </div>
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd web-frontend && npx tsc --noEmit`
Expected: no output (or only warnings unrelated to this change)

- [ ] **Step 6: Start frontend dev server (manual)**

User runs:
```bash
cd web-frontend && npm start
```
Expected: browser opens at http://localhost:3000 with no compile errors.

- [ ] **Step 7: Manual UI verification**

Navigate to Live Capture panel. Verify:
- Attack Simulator section appears above the stats grid
- Tips message is visible and mentions switching to `lo`
- 5 attack buttons are shown but disabled (greyed out) when not capturing
- Select `lo` from interface dropdown → buttons still disabled (not capturing yet)
- Click `Start Capture` → buttons become enabled
- Click `DDoS` button → button shows "注入中..." briefly, message appears, new packets flow into the table below with prediction shown
- Stop capture → buttons disable again

---

## Task 4: End-to-end demo verification

- [ ] **Step 1: Run full demo flow**

1. Backend running with `sudo python app.py` on port 5000
2. Frontend running at http://localhost:3000
3. In UI:
   - Switch interface to `lo`
   - Click `Start Capture`
   - Click each attack button in turn: DDoS, PortScan, FTP-Patator, SSH-Patator, DoS Hulk
   - Verify packets appear in table with varying `Prediction` values (not all BENIGN)
   - Verify the stats cards update (Total, Normal, Anomaly, Unknown)
4. Click `Stop Capture`
5. Verify buttons disable

- [ ] **Step 2: Confirm no regressions**

- `eth0` / `wlan0` / `any` selection still works for normal capture (no injection)
- Dataset Panel still shows correct metrics (from previous task)
- Training Panel still functional

---

## Self-Review Checklist

- [x] **Spec coverage**: All spec sections mapped to tasks (Task 1 = inject method, Task 2 = API endpoint, Task 3 = UI, Task 4 = E2E verification)
- [x] **No placeholders**: All code blocks are complete
- [x] **Type consistency**: `AttackType`, `ATTACK_TYPES`, `inject_attack_packets(attack_type, count)` signatures match across tasks
- [x] **File paths**: All paths exact and relative to project root
- [x] **Tips section**: Included per user request (Task 3 Step 4)
- [x] **Page clarity**: Attack Simulator visually distinct (yellow border/bg), clear disabled state, status feedback

## Notes for Executor

- **Root permission required**: Backend must run with `sudo` for scapy to send raw packets and sniff.
- **Model classification caveat**: Because features are extracted per-packet (not per-flow), the model may not classify each injected packet as the exact intended attack type. This is acceptable for demo — the point is showing non-BENIGN detections flowing in real-time, not 100% label accuracy. The spec acknowledges this.
- **No unit tests added**: Backend logic is simple pass-through to scapy; UI logic is standard React state. Manual verification in Tasks 2 and 4 covers correctness.
