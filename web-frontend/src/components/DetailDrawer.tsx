import React, { useState } from 'react';

interface Props { event: any; data: any; loading: boolean; onAnalyze?: () => void; }

const VERDICT_CN: Record<string, string> = {
  CONFIRMED_THREAT: '确认威胁', LIKELY_THREAT: '疑似威胁',
  UNCERTAIN: '不确定', LIKELY_BENIGN: '疑似误报',
};

const DetailDrawer: React.FC<Props> = ({ event, data, loading, onAnalyze }) => {
  const [tab, setTab] = useState<'agent' | 'rag' | 'xai'>('agent');

  if (loading) return <div style={{ color: 'var(--muted)', padding: 24, textAlign: 'center', fontSize: 13 }}>正在加载分析数据...</div>;
  if (!data) return <div style={{ color: 'var(--muted)', padding: 24, textAlign: 'center', fontSize: 13 }}>暂无分析数据</div>;

  const { debate, rag, xai } = data;

  return (
    <div>
      <div className="drawer-tabs">
        <button className={`drawer-tab ${tab === 'agent' ? 'active' : ''}`} onClick={() => setTab('agent')}>🧠 智能体辩论</button>
        <button className={`drawer-tab ${tab === 'rag' ? 'active' : ''}`} onClick={() => setTab('rag')}>🔍 知识库证据</button>
        <button className={`drawer-tab ${tab === 'xai' ? 'active' : ''}`} onClick={() => setTab('xai')}>📊 决策归因</button>
      </div>

      {/* Agent Debate — 按需触发 */}
      {tab === 'agent' && (
        <div>
          {!debate ? (
            <div style={{ textAlign: 'center', padding: '40px 20px' }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>🤖</div>
              <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 16 }}>
                Agent 辩论需要调用 LLM API，消耗 Token。
                <br />点击下方按钮按需分析此流量。
              </div>
              <button
                onClick={onAnalyze}
                disabled={loading}
                style={{
                  padding: '10px 24px',
                  borderRadius: 8,
                  border: '1px solid var(--border)',
                  background: 'var(--surface2)',
                  color: 'var(--text)',
                  cursor: 'pointer',
                  fontSize: 13,
                }}
              >
                {loading ? '⏳ 分析中...' : '🤖 运行 Agent 深度分析'}
              </button>
            </div>
          ) : (
            <>
              <div className="agent-grid">
                {[
                  { key: 'detector', name: '检测员 Agent', icon: '🕵️', desc: '从攻击特征和证据角度分析', data: debate.detector },
                  { key: 'analyst', name: '分析师 Agent', icon: '🔬', desc: '从误报角度审视，寻找良性解释', data: debate.analyst },
                  { key: 'arbiter', name: '裁决官 Agent', icon: '⚖️', desc: '综合双方意见，做出最终判定', data: debate.arbiter },
                ].map(a => (
                  <div key={a.key} className="agent-card">
                    <div className="agent-header">
                      <div>
                        <div className="agent-name">{a.icon} {a.name}</div>
                        <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 1 }}>{a.desc}</div>
                      </div>
                      <span className={`tag ${a.data?.verdict?.includes('THREAT') ? 'tag-red' : a.data?.verdict?.includes('BENIGN') ? 'tag-green' : 'tag-yellow'}`}>
                        {VERDICT_CN[a.data?.verdict] || a.data?.verdict || 'N/A'}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.6 }}>
                      置信度: {(a.data?.confidence * 100).toFixed(0)}% &middot; {a.data?.analysis || ''}
                    </div>
                    <div style={{ marginTop: 8, fontSize: 11 }}>
                      {(a.data?.evidence || []).map((e: string, i: number) => (
                        <div key={i} style={{ color: 'var(--green)', marginBottom: 2 }}>✓ {e}</div>
                      ))}
                      {(a.data?.counter_evidence || []).map((e: string, i: number) => (
                        <div key={i} style={{ color: 'var(--red)', marginBottom: 2 }}>✗ {e}</div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <div className="final-verdict">
                <h4>
                  {debate.verdict === 'MALICIOUS' ? '🚨 恶意流量' : debate.verdict === 'SUSPICIOUS' ? '⚠️ 可疑流量' : '✅ 良性 / 误报'}
                  <span style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 400, marginLeft: 8 }}>
                    {debate.vote_count}（三智能体投票结果）
                  </span>
                </h4>
                <p>{debate.recommendation || debate.action || '无处置建议'}</p>
                <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 6 }}>
                  研判模式: {debate.debate_mode === 'llm' ? 'LLM 大模型' : '规则引擎（离线）'}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* RAG Evidence */}
      {tab === 'rag' && rag && (
        <div>
          <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 12 }}>
            基于当前流量特征向量，在攻击模式知识库中检索最相似的已知攻击模式：
          </div>
          <div className="rag-list">
            {(rag.matches || []).length > 0 ? rag.matches.map((m: any, i: number) => (
              <div key={i} className="rag-item">
                <div className="rag-header">
                  <div className="rag-name">{m.name}</div>
                  <div className="rag-sim">相似度 {(m.similarity * 100).toFixed(0)}%</div>
                </div>
                <div className="rag-desc">{m.description}</div>
                {m.mitre && (
                  <div className="rag-mitre">
                    <span className="tag tag-purple">{m.mitre.id}</span>
                    <span style={{ marginLeft: 6, color: 'var(--muted)' }}>{m.mitre.name}</span>
                    {m.mitre.tactics?.length > 0 && (
                      <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--muted)' }}>
                        战术阶段: {m.mitre.tactics.join(', ')}
                      </span>
                    )}
                  </div>
                )}
                <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
                  严重程度: <span style={{ color: m.severity === 'critical' ? 'var(--red)' : m.severity === 'high' ? 'var(--yellow)' : 'var(--green)' }}>
                    {m.severity === 'critical' ? '严重' : m.severity === 'high' ? '高' : m.severity === 'medium' ? '中' : '低'}
                  </span>
                </div>
              </div>
            )) : <div style={{ color: 'var(--muted)', fontSize: 12 }}>知识库中未找到相似攻击模式</div>}
          </div>
          {rag.stats && (
            <div style={{ marginTop: 16, fontSize: 10, color: 'var(--muted)' }}>
              知识库规模: {rag.stats.num_patterns} 条攻击模式 &middot; {rag.stats.num_mitre_techniques} 项 MITRE ATT&CK 技术
            </div>
          )}
        </div>
      )}

      {/* XAI */}
      {tab === 'xai' && xai && (
        <div>
          <div style={{ fontSize: 13, color: 'var(--text)', marginBottom: 16, lineHeight: 1.6, padding: '10px 12px', background: 'var(--bg)', borderRadius: 6, border: '1px solid var(--border)' }}>
            {xai.explanation || '无法生成解释'}
          </div>

          <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 10, fontWeight: 600 }}>特征贡献排序（扰动归因分析）</div>
          {(xai.top_features || []).length > 0 ? (
            <div className="xai-features">
              {xai.top_features.slice(0, 10).map((f: any, i: number) => (
                <div key={i} className="xai-feature">
                  <div className="f-name" title={f.name}>{f.name}</div>
                  <div className="f-bar">
                    <div className="f-fill" style={{
                      width: `${Math.min(100, f.importance * 200)}%`,
                      background: f.importance > 0.005 ? 'var(--red)' : 'var(--blue)',
                    }} />
                  </div>
                  <div className="f-val">{(f.importance * 100).toFixed(2)}%</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: 'var(--muted)', fontSize: 12 }}>该事件的模型置信度过高，特征扰动未产生显著影响，无法提取有效的特征归因。</div>
          )}

          <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 16 }}>
            分析方法: {xai.method || '特征扰动'} &middot; 判定结果: {xai.prediction} &middot; 未知攻击: {xai.is_unknown ? '是' : '否'}
          </div>
        </div>
      )}
    </div>
  );
};

export default DetailDrawer;
