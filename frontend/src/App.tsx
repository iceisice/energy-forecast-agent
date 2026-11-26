import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Activity, Thermometer, Clock, Calendar, AlertTriangle, CheckCircle2, Search, Droplets, HelpCircle, Info, Send, Bot, ShieldAlert, FileText } from 'lucide-react';
import './App.css';

interface HistoryData {
  timestamp: string;
  usage: number;
  temperature: number;
  humidity: number;
  hour: number;
  month: number;
}

interface AgentAnalysisResult {
  prediction: number;
  is_ood: boolean;
  ood_reason: string;
  retrieved_docs: string[];
  report: string;
}

function App() {
  const [history, setHistory] = useState<HistoryData[]>([]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  const [targetDate, setTargetDate] = useState('2025-08-05');
  const [targetHour, setTargetHour] = useState('14');

  const [inputTemp, setInputTemp] = useState<string>('25');
  const [inputHumidity, setInputHumidity] = useState<string>('60');
  const [inputMonth, setInputMonth] = useState<string>('8');
  const [inputHour, setInputHour] = useState<string>('14');
  const [isWeekend, setIsWeekend] = useState<number>(0);

  // Agent State
  const [agentQuery, setAgentQuery] = useState('');
  const [agentResult, setAgentResult] = useState<AgentAnalysisResult | null>(null);
  const [agentError, setAgentError] = useState<string | null>(null);
  const [agentLoading, setAgentLoading] = useState(false);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/history?target_date=${targetDate}`);
      const data = await response.json();

      if (Array.isArray(data) && data.length > 0) {
        setHistory(data);
        const selected = data.find((d: any) => d.hour === Number(targetHour)) || data[0];
        setInputTemp(String(selected.temperature.toFixed(1)));
        setInputHumidity(String(selected.humidity.toFixed(0)));
        setInputMonth(String(selected.month));
        setInputHour(String(selected.hour));
      }
    } catch (error) {
      console.error('Error fetching history:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature: parseFloat(inputTemp),
          humidity: parseFloat(inputHumidity),
          hour: parseInt(inputHour),
          month: parseInt(inputMonth),
          is_weekend: isWeekend,
          is_holiday: 0
        }),
      });
      const data = await response.json();
      setPrediction(data.prediction);
    } catch (error) {
      console.error('Error predicting:', error);
    }
  };

  const handleAgentAnalyze = async () => {
    if (!agentQuery.trim()) return;
    setAgentLoading(true);
    setAgentResult(null);
    setAgentError(null);
    try {
      const response = await fetch('http://localhost:8000/api/agent/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: agentQuery }),
      });
      const data = await response.json();
      if (!response.ok || data.detail || data.error) {
        setAgentError(data.detail || data.error || '분석 중 오류가 발생했습니다.');
      } else {
        setAgentResult(data);
      }
    } catch (error) {
      setAgentError('백엔드 서버(localhost:8000) 연결에 실패했습니다.');
    } finally {
      setAgentLoading(false);
    }
  };

  const handleNumberChange = (setter: React.Dispatch<React.SetStateAction<string>>, value: string) => {
    if (value === '' || /^-?\d*\.?\d*$/.test(value)) {
      if (value.length > 1 && value.startsWith('0') && value[1] !== '.') {
        setter(value.substring(1));
      } else {
        setter(value);
      }
    }
  };

  const openRationale = () => {
    window.open('/rationale.html', '_blank', 'width=1000,height=800');
  };

  return (
    <div className="container">
      <header className="header main-header">
        <div className="header-content">
          <div className="title-area">
            <h1><Activity size={32} /> 에너지 수요 예측 시스템</h1>
            <button className="btn-help-circle" onClick={openRationale} title="가중치 산정 근거 보기">
              <HelpCircle size={20} />
            </button>
          </div>
          <p>AI-Powered Energy Forecasting (Reflecting KEPCO Regional Weights)</p>
        </div>
      </header>

      <main>
        <div className="card filter-card">
          <div className="filter-title">
            <Search size={20} color="#1e3a8a" />
            <h2>데이터 조회 설정</h2>
          </div>
          <div className="filter-inputs">
            <div className="input-field">
              <label>조회 일자</label>
              <input type="date" min="2025-01-01" max="2025-12-31" value={targetDate} onChange={(e) => setTargetDate(e.target.value)} />
            </div>
            <div className="input-field">
              <label>조회 시간</label>
              <select value={targetHour} onChange={(e) => setTargetHour(e.target.value)}>
                {Array.from({length: 24}).map((_, i) => (
                  <option key={i} value={i}>{i}시</option>
                ))}
              </select>
            </div>
            <button className="btn-search" onClick={fetchHistory} disabled={loading}>
              {loading ? '로딩 중...' : '조회 실행'}
            </button>
          </div>
        </div>

        <div className="stats-grid">
          <div className="card stat-card">
            <div className="stat-header">
              <div className="stat-icon bg-blue"><Thermometer size={18} /></div>
              <h3>가중 평균 기온 <Info size={14} title="지역별 전력 소비량 가중치 적용" /></h3>
            </div>
            <p className="stat-value tabular">
              {history.length > 0 ? (history.find(d => d.hour === Number(targetHour))?.temperature.toFixed(1) || history[0].temperature.toFixed(1)) : '--'}
              <span className="unit">°C</span>
            </p>
          </div>

          <div className="card stat-card">
            <div className="stat-header">
              <div className="stat-icon bg-cyan"><Droplets size={18} /></div>
              <h3>가중 평균 습도</h3>
            </div>
            <p className="stat-value tabular">
              {history.length > 0 ? (history.find(d => d.hour === Number(targetHour))?.humidity.toFixed(0) || history[0].humidity.toFixed(0)) : '--'}
              <span className="unit">%</span>
            </p>
          </div>

          <div className="card stat-card">
            <div className="stat-header">
              <div className="stat-icon bg-green"><Activity size={18} /></div>
              <h3>국가 전력 사용량</h3>
            </div>
            <p className="stat-value tabular">
              {history.length > 0 ? (history.find(d => d.hour === Number(targetHour))?.usage.toLocaleString() || history[0].usage.toLocaleString()) : '--'}
              <span className="unit">MW</span>
            </p>
          </div>
        </div>

        <div className="card chart-container">
          <h2>{targetDate} 전력 수요 변동 추이 (가중 기상 데이터 적용)</h2>
          {history.length > 0 ? (
            <div style={{ width: '100%', height: 350 }}>
              <ResponsiveContainer>
                <LineChart data={history} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis
                    dataKey="hour"
                    interval={0}
                    tick={{ fontSize: 11, fontWeight: 600, fill: '#64748b' }}
                    tickFormatter={(val) => `${val}h`}
                    axisLine={{ stroke: '#cbd5e1' }}
                    tickLine={false}
                  />
                  <YAxis yAxisId="left" tick={{fontSize: 12}} />
                  <YAxis yAxisId="right" orientation="right" tick={{fontSize: 12}} />
                  <Tooltip formatter={(value) => typeof value === 'number' ? value.toLocaleString() : value} />
                  <Legend verticalAlign="top" height={36} />
                  <Line yAxisId="left" type="monotone" dataKey="usage" stroke="#3b82f6" strokeWidth={3} name="사용량(MW)" dot={{ r: 4 }} activeDot={{ r: 6 }} />
                  <Line yAxisId="right" type="monotone" dataKey="temperature" stroke="#ef4444" name="가중기온(°C)" dot={false} strokeDasharray="5 5" />
                  <Line yAxisId="right" type="monotone" dataKey="humidity" stroke="#06b6d4" name="가중습도(%)" dot={false} strokeDasharray="3 3" opacity={0.6} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="result-placeholder">조회된 데이터가 없습니다.</div>
          )}
        </div>

        <div className="predict-section">
          <div className="card">
            <h2><Clock size={20} color="#1e3a8a" /> 고정밀 수요 예측 시뮬레이터</h2>
            <div className="form-grid">
              <div className="form-group">
                <label>예상 월 (Month)</label>
                <select value={inputMonth} onChange={(e) => setInputMonth(e.target.value)}>
                  {Array.from({length: 12}).map((_, i) => (
                    <option key={i+1} value={i+1}>{i+1}월</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>예상 가중기온 (°C)</label>
                <input type="text" value={inputTemp} onChange={(e) => handleNumberChange(setInputTemp, e.target.value)} />
              </div>
              <div className="form-group">
                <label>예상 가중습도 (%)</label>
                <input type="text" value={inputHumidity} onChange={(e) => handleNumberChange(setInputHumidity, e.target.value)} />
              </div>
              <div className="form-group">
                <label>시간대 (0-23)</label>
                <input type="text" value={inputHour} onChange={(e) => handleNumberChange(setInputHour, e.target.value)} />
              </div>
              <div className="form-group" style={{ gridColumn: 'span 2' }}>
                <label>주말 여부</label>
                <select value={isWeekend} onChange={(e) => setIsWeekend(Number(e.target.value))}>
                  <option value={0}>평일 (Working Day)</option>
                  <option value={1}>주말/공휴일 (Weekend)</option>
                </select>
              </div>
            </div>
            <button className="btn-predict" onClick={handlePredict} disabled={!inputTemp || !inputHumidity || !inputHour}>
              고정밀 AI 분석 실행
            </button>
          </div>

          <div className="card result-card">
            <h2><Calendar size={20} color="#1e3a8a" /> 분석 결과 보고</h2>
            {prediction !== null ? (
              <div className="result-display">
                <p className="result-label">지역별 비중 반영 예상 수요</p>
                <p className="result-value tabular">{prediction.toLocaleString()}<span className="unit">MW</span></p>
                <div className="result-info">
                  <p>한전 전력 판매량 가중치가 적용된 <strong>{inputMonth}월</strong>의 기상 조건을 분석했습니다.</p>
                </div>
              </div>
            ) : (
              <div className="result-display" style={{ color: '#94a3b8', paddingTop: '60px' }}>
                <Activity size={48} style={{ opacity: 0.2, marginBottom: '16px' }} />
                <p>시뮬레이션 조건을 입력하고 분석을 실행하십시오.</p>
              </div>
            )}
          </div>
        </div>

        {/* AI Agent Section */}
        <div className="card agent-card" style={{ marginTop: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Bot size={28} color="#1e3a8a" />
              <h2 style={{ margin: 0 }}>AI 수요 리스크 분석 에이전트 (OOD + RAG)</h2>
            </div>
            {agentResult && (
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '6px 14px',
                  borderRadius: '9999px',
                  fontSize: '0.85rem',
                  fontWeight: 700,
                  backgroundColor: agentResult.is_ood ? '#fef2f2' : '#ecfdf5',
                  color: agentResult.is_ood ? '#dc2626' : '#059669',
                  border: `1px solid ${agentResult.is_ood ? '#fecaca' : '#a7f3d0'}`,
                }}
              >
                {agentResult.is_ood ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
                {agentResult.is_ood ? 'OOD 리스크 감지 (분포 이탈)' : '정상 범위 (In-Distribution)'}
              </span>
            )}
          </div>

          <div className="agent-container" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div className="agent-input-group" style={{ display: 'flex', gap: '8px' }}>
              <input
                type="text"
                placeholder="예: 내일 기온 38도, 습도 80%일 때 전력 수요 예측하고 리스크 분석해 줘"
                value={agentQuery}
                onChange={(e) => setAgentQuery(e.target.value)}
                style={{ flex: 1, padding: '12px 14px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.95rem' }}
                onKeyDown={(e) => e.key === 'Enter' && handleAgentAnalyze()}
              />
              <button
                className="btn-predict"
                onClick={handleAgentAnalyze}
                disabled={agentLoading || !agentQuery.trim()}
                style={{ width: 'auto', padding: '0 22px', margin: 0, display: 'inline-flex', alignItems: 'center', gap: '6px' }}
              >
                {agentLoading ? '분석 중...' : <><Send size={18} /> 분석 실행</>}
              </button>
            </div>

            {agentError && (
              <div style={{ background: '#fef2f2', color: '#b91c1c', padding: '14px 16px', borderRadius: '8px', border: '1px solid #fecaca', fontSize: '0.9rem' }}>
                <strong>오류:</strong> {agentError}
              </div>
            )}

            {agentResult && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
                  <div style={{ background: '#f8fafc', padding: '14px 16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
                    <span style={{ fontSize: '0.8rem', color: '#64748b', display: 'block', marginBottom: '4px' }}>ML 모델 예측 수요</span>
                    <strong style={{ fontSize: '1.35rem', color: '#0f172a' }}>{agentResult.prediction.toLocaleString()} MW</strong>
                  </div>
                  <div style={{ background: agentResult.is_ood ? '#fff1f2' : '#f8fafc', padding: '14px 16px', borderRadius: '10px', border: `1px solid ${agentResult.is_ood ? '#fecdd3' : '#e2e8f0'}` }}>
                    <span style={{ fontSize: '0.8rem', color: '#64748b', display: 'block', marginBottom: '4px' }}>OOD 판별 사유</span>
                    <strong style={{ fontSize: '0.92rem', color: agentResult.is_ood ? '#be123c' : '#334155' }}>{agentResult.ood_reason}</strong>
                  </div>
                </div>

                {agentResult.retrieved_docs && agentResult.retrieved_docs.length > 0 && (
                  <div style={{ background: '#eff6ff', padding: '14px 16px', borderRadius: '10px', border: '1px solid #bfdbfe' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#1e40af', fontWeight: 700, fontSize: '0.9rem', marginBottom: '8px' }}>
                      <FileText size={16} /> RAG 참조 매뉴얼 근거 ({agentResult.retrieved_docs.length}건)
                    </div>
                    <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '0.85rem', color: '#1e3a8a', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      {agentResult.retrieved_docs.map((doc, idx) => (
                        <li key={idx} style={{ lineHeight: '1.45' }}>{doc}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="agent-result" style={{ background: '#f8fafc', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: '#1e3a8a' }}>
                    <ShieldAlert size={20} />
                    <strong style={{ fontSize: '1.05rem' }}>AI 에이전트 종합 의사결정 리포트</strong>
                  </div>
                  <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.65', fontSize: '0.95rem', color: '#334155' }}>
                    {agentResult.report}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>&copy; 2026 Public Energy Data Intelligence Lab.</p>
      </footer>
    </div>
  );
}

export default App;
