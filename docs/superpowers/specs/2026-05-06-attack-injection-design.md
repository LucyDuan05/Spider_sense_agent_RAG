---
name: Attack Injection for Live Capture Demo
description: Add attack traffic injection capability to demonstrate IDS detection in course/graduation project demos
type: feature
date: 2026-05-06
---

# Attack Injection for Live Capture Demo

## Overview

**Problem**: Current live capture functionality can only capture real network traffic on eth0, which is mostly benign. This makes it impossible to demonstrate the system's attack detection capabilities in a demo/presentation setting.

**Solution**: Add an "Attack Simulator" to the LiveCapturePanel that uses scapy to craft and inject attack-characteristic packets to the loopback interface (lo), which are then captured and classified by the existing detection pipeline.

**Target Scenario**: Course/graduation project demonstration where the system needs to show real-time attack detection without actual malicious traffic.

## Architecture & Data Flow

```
[Frontend] User clicks attack button
    ↓ POST /api/capture/inject {attack_type, count}
[Backend] inject_attack_packets() constructs scapy packets
    ↓ send() to 127.0.0.1 (loopback)
[Existing] sniff(iface='lo') captures packets
    ↓ _packet_callback() extracts features
[Model] Inference engine classifies packet
    ↓ Updates recent_results
[Frontend] Polls /api/capture/status every 1s
    ↓ Displays new detections in packet table
```

**Key Constraint**: Capture must be running on `lo` interface for injection to work.

## Frontend Changes

### LiveCapturePanel.tsx

Add "Attack Simulator" section below existing controls:

```
─── Attack Simulator ──────────────────────────────
💡 Tips: 使用攻击模拟器前，请先将接口切换为 lo，再点击 Start Capture

[ DDoS ] [ PortScan ] [ FTP-Patator ] [ SSH-Patator ] [ DoS Hulk ]
（按钮在 lo+抓包运行时才可点击）
───────────────────────────────────────────────────
```

**UI Logic**:
- Tips section always visible
- Attack buttons enabled only when `isCapturing && selectedInterface === 'lo'`
- On click: POST to `/api/capture/inject`, disable button briefly, show injection status
- Injected packets appear in existing packet table with their classifications

## Backend Changes

### app.py

New endpoint:

```python
@app.route('/api/capture/inject', methods=['POST'])
def inject_attack():
    data = request.json or {}
    attack_type = data.get('attack_type', 'DDoS')
    count = int(data.get('count', 20))
    
    if not engine.is_running:
        return jsonify({'success': False, 'error': '抓包未运行'}), 400
    
    thread = threading.Thread(target=engine.inject_attack_packets, args=(attack_type, count))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': f'开始注入 {attack_type} ({count}包)'})
```

### inference_engine.py

New method in `IDSInferenceEngine`:

```python
def inject_attack_packets(self, attack_type, count):
    """Inject crafted attack packets to loopback for demo purposes"""
    from scapy.all import IP, TCP, send
    import time
    
    templates = {
        'DDoS':        {'dport': 80,   'flags': 'S',  'size': 60},
        'PortScan':    {'dport': None, 'flags': 'S',  'size': 40},
        'FTP-Patator': {'dport': 21,   'flags': 'PA', 'size': 100},
        'SSH-Patator': {'dport': 22,   'flags': 'PA', 'size': 100},
        'DoS Hulk':    {'dport': 80,   'flags': 'PA', 'size': 1400},
    }
    
    t = templates.get(attack_type, templates['DDoS'])
    
    for i in range(count):
        dport = (i % 1024 + 1) if t['dport'] is None else t['dport']
        pkt = IP(src='127.0.0.1', dst='127.0.0.1') / TCP(dport=dport, flags=t['flags']) / (b'X' * t['size'])
        send(pkt, verbose=False)
        time.sleep(0.05)  # 50ms间隔，避免过快
```

## Packet Construction Strategy

Each attack type has distinct characteristics:

| Attack Type | Target Port | TCP Flags | Packet Size | Characteristics |
|-------------|-------------|-----------|-------------|-----------------|
| DDoS | 80 | S (SYN) | 60B | High-frequency SYN flood |
| PortScan | 1-1024 (rotating) | S | 40B | Scans multiple ports |
| FTP-Patator | 21 | PA (PSH+ACK) | 100B | Brute-force pattern |
| SSH-Patator | 22 | PA | 100B | Brute-force pattern |
| DoS Hulk | 80 | PA | 1400B | Large packets, high frequency |

**Why**: The model was trained on flow-level features from CICFlowMeter. While single-packet feature extraction won't perfectly match flow features, crafting packets with attack-characteristic ports, flags, and sizes increases the likelihood of correct classification for demo purposes.

## Error Handling

**Frontend**:
- Check `isCapturing && selectedInterface === 'lo'` before enabling buttons
- Display error message if API call fails

**Backend**:
- `/api/capture/inject` returns 400 if capture not running
- `send()` may require root permissions; catch `PermissionError` and log
- If capture is not on `lo`, packets will be sent but not captured (user responsibility, Tips warns about this)

**Edge Cases**:
- User stops capture during injection: injection thread completes, but packets won't be displayed (acceptable)
- Rapid button clicks: multiple injection threads run concurrently (acceptable for demo)

## Implementation Notes

1. **No tcpreplay dependency**: Uses only scapy, which is already in requirements
2. **Non-blocking injection**: Runs in daemon thread to avoid blocking Flask
3. **Minimal UI changes**: Reuses existing packet display table
4. **Clear user guidance**: Tips section prevents common mistakes

## Success Criteria

- User can select `lo` interface, start capture, click attack buttons, and see classified attack packets in the table
- Different attack types show different classifications (though not guaranteed 100% accurate due to single-packet vs flow-level feature mismatch)
- System remains responsive during injection
- Clear error messages when preconditions not met
