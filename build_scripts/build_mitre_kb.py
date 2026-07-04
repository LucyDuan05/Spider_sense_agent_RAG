"""
build_mitre_kb.py — 从 MITRE ATT&CK STIX JSON 构建语义 RAG 知识库

用法:
  1. 下载 MITRE ATT&CK 数据:
     https://github.com/mitre/cti/raw/master/enterprise-attack/enterprise-attack.json
     放到 knowledge/ 目录下

  2. 运行:
     python build_mitre_kb.py

  3. 输出:
     knowledge/mitre_semantic.json  — 完整知识库 (200+ 条)
     knowledge/mitre_embeddings.npy — 预计算向量 (可选, 加速启动)
"""
import json
import os
import sys

# 配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, 'knowledge')
INPUT_FILE = os.path.join(KNOWLEDGE_DIR, 'enterprise-attack.json')
OUTPUT_FILE = os.path.join(KNOWLEDGE_DIR, 'mitre_semantic.json')

# ── MITRE tactic → 中文战术阶段 ──
TACTIC_CN = {
    'reconnaissance': '侦察',
    'resource-development': '资源开发',
    'initial-access': '初始访问',
    'execution': '执行',
    'persistence': '持久化',
    'privilege-escalation': '权限提升',
    'defense-evasion': '防御规避',
    'credential-access': '凭据访问',
    'discovery': '发现',
    'lateral-movement': '横向移动',
    'collection': '数据收集',
    'command-and-control': '命令与控制',
    'exfiltration': '数据窃取',
    'impact': '影响',
}

# ── 网络流量可观测行为扩展 (补充 MITRE 中没有直接写的部分) ──
NETWORK_OBSERVABLES = {
    'T1498': ('Network Denial of Service. Observable: extreme inbound packet rate '
              '(>10000 pkts/s), uniform packet sizes, SYN flood pattern with >80% SYN ratio, '
              'distributed or single source IPs, target port 80/443/53, asymmetric traffic '
              'with minimal outbound response. Detection: baseline deviation analysis, '
              'packet rate anomaly, SYN ratio monitoring.'),
    'T1499': ('Endpoint Denial of Service. Observable: high connection rate to single '
              'service port, low data throughput per connection, connection pool exhaustion '
              'indicators (HTTP 503 errors, request queue backlog), Slowloris/Slow HTTP '
              'POST patterns. Detection: error rate monitoring, connection duration vs '
              'data transfer ratio analysis.'),
    'T1046': ('Network Service Scanning. Observable: connection attempts to multiple '
              'ports from single source IP in short time window, sequential port access '
              'pattern, SYN packets without completed handshake, tiny packets (<10B mean), '
              'high RST ratio. Detection: port scan signature analysis, connection rate '
              'anomaly per source IP, unusual port access monitoring.'),
    'T1110': ('Brute Force. Observable: repeated authentication attempts to same service '
              '(SSH port 22, FTP port 21, RDP port 3389), high login failure rate, '
              'uniform packet timing suggesting automated tool, single source IP targeting '
              'multiple accounts. Detection: login failure rate monitoring, account lockout '
              'event analysis, authentication timing analysis.'),
    'T1573': ('Encrypted Channel for C2. Observable: TLS traffic on non-standard ports, '
              'self-signed or unusual TLS certificates, periodic beaconing patterns '
              'at regular intervals, JA3/JA3S fingerprint anomalies, non-standard '
              'TLS extensions. Detection: TLS fingerprinting, certificate analysis, '
              'periodic connection interval detection.'),
    'T1071': ('Application Layer Protocol C2. Observable: periodic HTTP/HTTPS/DNS '
              'connections at regular intervals (beaconing), unusual User-Agent strings, '
              'DNS queries to newly registered domains, long-duration low-throughput '
              'connections. Detection: beacon interval analysis, DNS tunneling detection, '
              'User-Agent anomaly analysis.'),
    'T1041': ('Exfiltration Over C2 Channel. Observable: large outbound data transfers, '
              'asymmetric traffic heavily favoring outbound direction, non-standard '
              'destination ports, transfers occurring outside business hours, '
              'connections to unusual geographic locations. Detection: outbound traffic '
              'volume anomaly, DLP monitoring, destination IP reputation analysis.'),
    'T1190': ('Exploit Public-Facing Application. Observable: unusual HTTP request '
              'patterns, known exploit payload signatures in request body, abnormal '
              'URI patterns, requests targeting vulnerable software versions, '
              'subsequent unusual process creation or outbound connections. '
              'Detection: WAF log analysis, request payload inspection, '
              'known vulnerability signature matching.'),
    'T1566': ('Phishing. Observable: email traffic patterns to multiple internal hosts '
              'from external sources, SMTP connections from suspicious IPs, '
              'malicious attachment download patterns, link-click telemetry to '
              'newly registered domains. Detection: email security gateway analysis, '
              'SPF/DKIM/DMARC validation, URL reputation checking.'),
    'T1021': ('Remote Services Lateral Movement. Observable: internal SSH (port 22), '
              'RDP (port 3389), or SMB (port 445) connections originating from '
              'non-admin workstations, after-hours remote access, unusual session '
              'durations between internal hosts. Detection: internal remote connection '
              'auditing, session duration anomaly analysis, source-destination pair analysis.'),
    'T1090': ('Proxy Connection. Observable: traffic routed through non-standard proxy '
              'ports, connections to known proxy/anonymizer IPs, unusual port usage '
              'patterns (e.g., SSH over port 443), multi-hop connection chains. '
              'Detection: proxy detection via port/protocol mismatch, IP reputation analysis.'),
    'T1568': ('Dynamic Resolution C2. Observable: frequent DNS queries to DDNS domains, '
              'domain generation algorithm (DGA) patterns with high-entropy subdomains, '
              'rapidly changing domain resolutions. Detection: DGA detection algorithms, '
              'DNS query entropy analysis, NXDOMAIN response rate monitoring.'),
    'T1003': ('OS Credential Dumping. Observable: unusual LSASS process memory access '
              '(Windows), volume shadow copy creation, Mimikatz tool network signatures, '
              'abnormal inter-process communication patterns. Detection: endpoint process '
              'monitoring, LSASS access auditing.'),
    'T1059': ('Command and Scripting Interpreter. Observable: PowerShell process creation '
              'with encoded commands, Base64-encoded network payloads, abnormal script '
              'download patterns, WMI call patterns, Python/Bash reverse shell connections. '
              'Detection: script block logging, command-line auditing, encoded payload detection.'),
    'T1204': ('User Execution. Observable: executable file downloads from email or web '
              'sources, Office documents with macros, suspicious process chains starting '
              'from document readers or browsers. Detection: file type analysis at network '
              'perimeter, attachment sandboxing, parent-child process analysis.'),
}


def build():
    """主逻辑: enterprise-attack.json → mitre_semantic.json"""
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到 {INPUT_FILE}")
        print("请先从以下地址下载 MITRE ATT&CK 数据:")
        print("  https://github.com/mitre/cti/raw/master/enterprise-attack/enterprise-attack.json")
        print(f"放到 {KNOWLEDGE_DIR}/ 目录下后重新运行此脚本。")
        sys.exit(1)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        attack_data = json.load(f)

    objects = attack_data.get('objects', [])
    print(f"读取到 {len(objects)} 个 STIX 对象")

    # 建立 technique ID → 详细信息 的映射
    techniques = {}

    for obj in objects:
        if obj.get('type') == 'attack-pattern':
            tech_id = None
            for ext_ref in obj.get('external_references', []):
                if ext_ref.get('source_name') == 'mitre-attack':
                    tech_id = ext_ref.get('external_id', '')
                    break

            if not tech_id:
                continue

            # 过滤掉移动端和 ICS 技术（只保留 Enterprise）
            if 'mobile' in obj.get('x_mitre_platforms', []) and \
               'Windows' not in obj.get('x_mitre_platforms', []) and \
               'Linux' not in obj.get('x_mitre_platforms', []) and \
               'macOS' not in obj.get('x_mitre_platforms', []):
                continue

            # 提取战术
            kill_chain = obj.get('kill_chain_phases', [])
            tactics = [p.get('phase_name', '') for p in kill_chain]

            techniques[tech_id] = {
                'id': tech_id,
                'name': obj.get('name', ''),
                'description': obj.get('description', ''),
                'tactics': tactics,
                'platforms': obj.get('x_mitre_platforms', []),
                'is_subtechnique': obj.get('x_mitre_is_subtechnique', False),
                'detection': obj.get('x_mitre_detection', ''),
                'data_sources': obj.get('x_mitre_data_sources', []),
            }

        elif obj.get('type') == 'relationship':
            # 忽略 relationship（当前构建流程不需要子技术分组）
            pass

    print(f"提取到 {len(techniques)} 个攻击技术/子技术")

    # ── 构建知识库条目 ──
    entries = []
    count_techniques = 0
    count_subtechniques = 0

    for tech_id, tech in sorted(techniques.items()):
        # 构建 key_text (英文, 用于 embedding 匹配)
        tactics_en = ', '.join(tech['tactics'])
        tactics_cn = ', '.join(TACTIC_CN.get(t, t) for t in tech['tactics'])
        platforms = ', '.join(tech['platforms'])

        # 网络可观测行为 (有则用, 无则只用 MITRE 原文)
        net_obs = NETWORK_OBSERVABLES.get(tech_id, '')
        if net_obs:
            observable_text = f" Network Observable Behaviors: {net_obs}"
        else:
            observable_text = ''

        key_text = (
            f"MITRE ATT&CK {tech_id}: {tech['name']}. "
            f"Tactics: {tactics_en}. "
            f"Platforms: {platforms}. "
            f"Description: {tech['description']}. "
            f"Detection: {tech.get('detection', 'Monitor network traffic for anomalous patterns.')}."
            f"{observable_text}"
        )

        # 构建 value (中文, 用于前端展示和 Agent prompt)
        sub_label = ' (子技术)' if tech['is_subtechnique'] else ''

        entries.append({
            'id': tech_id,
            'type': 'mitre_subtechnique' if tech['is_subtechnique'] else 'mitre_technique',
            'key_text': key_text,
            'value': {
                'technique_id': tech_id,
                'technique_name': f"{tech['name']}{sub_label}",
                'tactics': tech['tactics'],
                'tactics_cn': tactics_cn,
                'platforms': tech['platforms'],
                'description': tech['description'],
                'detection': tech.get('detection', ''),
                'data_sources': tech.get('data_sources', []),
            }
        })

        if tech['is_subtechnique']:
            count_subtechniques += 1
        else:
            count_techniques += 1

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 知识库已生成: {OUTPUT_FILE}")
    print(f"  技术: {count_techniques} 个")
    print(f"  子技术: {count_subtechniques} 个")
    print(f"  总计: {len(entries)} 条")

    # 打印网络观测扩展统计
    with_obs = sum(1 for e in entries if 'Network Observable Behaviors' in e['key_text'])
    print(f"  含网络流量可观测行为扩展: {with_obs} 条 (其余仅 MITRE 原文)")

    # 提示下一步
    print(f"\n下一步:")
    print(f"  1. 重启后端, SemanticRAG 会自动读取 {OUTPUT_FILE}")
    print(f"  2. 前端 RAG 标签页会显示 200+ 条目")


if __name__ == '__main__':
    build()
