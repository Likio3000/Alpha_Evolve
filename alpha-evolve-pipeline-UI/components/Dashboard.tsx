import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type RunPayload = Record<string, string | number | boolean>;

type EventMsg =
  | { type: 'status'; msg: string; code?: number; args?: string[] }
  | { type: 'candidate'; idx: number; total: number; params: string; raw: string }
  | { type: 'score'; sharpe_best: number; raw: string }
  | { type: 'final'; run_dir: string; sharpe_best: number | null }
  | { type: 'log'; raw: string };

const API_BASE = (typeof window !== 'undefined' ? (window.location.origin.replace(/:\\d+$/, ':8000')) : 'http://127.0.0.1:8000');

const defaultPayload: RunPayload = {
  iters: 2,
  gens: 10,
  data_dir: '',
  base_config: '',
  bt_top: 10,
  no_clean: false,
  dry_run: false,
  // Passthrough evolution knobs (optional)
  selection_metric: 'auto',
  ramp_fraction: 0.33,
  ramp_min_gens: 5,
  novelty_boost_w: 0.02,
  novelty_struct_w: 0.0,
  hof_corr_mode: 'flat',
  ic_tstat_w: 0.0,
  temporal_decay_half_life: 0,
  rank_softmax_beta_floor: 0.0,
  rank_softmax_beta_target: 2.0,
  corr_penalty_w: 0.35,
};

const Dashboard: React.FC = () => {
  const [form, setForm] = useState<RunPayload>(defaultPayload);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('idle');
  const [candidate, setCandidate] = useState<{ idx: number; total: number; params: string } | null>(null);
  const [scores, setScores] = useState<number[]>([]);
  const [best, setBest] = useState<number | null>(null);
  const [finalInfo, setFinalInfo] = useState<{ run_dir: string | null; sharpe_best: number | null }>({ run_dir: null, sharpe_best: null });
  const [logs, setLogs] = useState<string[]>([]);
  const esRef = useRef<EventSource | null>(null);

  const onChange = useCallback((id: string, val: string | number | boolean) => {
    setForm(prev => ({ ...prev, [id]: val }));
  }, []);

  const startRun = useCallback(async () => {
    setStatus('starting');
    setCandidate(null);
    setScores([]);
    setBest(null);
    setFinalInfo({ run_dir: null, sharpe_best: null });
    setLogs([]);
    try {
      const resp = await fetch(`${API_BASE}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await resp.json();
      const jid = data.job_id as string;
      setJobId(jid);
      setStatus('running');
      const es = new EventSource(`${API_BASE}/api/events/${jid}`);
      esRef.current = es;
      es.onmessage = (ev: MessageEvent) => {
        try {
          const msg = JSON.parse(ev.data) as EventMsg;
          if (msg.type === 'status') {
            setStatus(`${msg.msg}${msg.code !== undefined ? ` (${msg.code})` : ''}`);
          } else if (msg.type === 'candidate') {
            setCandidate({ idx: msg.idx, total: msg.total, params: msg.params });
          } else if (msg.type === 'score') {
            setScores(prev => {
              const next = [...prev, msg.sharpe_best];
              setBest(prevBest => (prevBest === null ? msg.sharpe_best : Math.max(prevBest, msg.sharpe_best)));
              return next;
            });
          } else if (msg.type === 'final') {
            setFinalInfo({ run_dir: msg.run_dir, sharpe_best: msg.sharpe_best });
          } else if (msg.type === 'log') {
            setLogs(prev => (prev.length > 1000 ? prev.slice(-1000) : prev).concat(msg.raw));
          }
        } catch (_) {
          // ignore
        }
      };
      es.onerror = () => {
        setStatus('disconnected');
        es.close();
      };
    } catch (e) {
      setStatus('error');
    }
  }, [form]);

  const stopRun = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
      setStatus('stopped');
    }
  }, []);

  const fetchLastRun = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/last-run`);
      const data = await resp.json();
      setFinalInfo({ run_dir: data.run_dir, sharpe_best: data.sharpe_best });
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchLastRun();
  }, [fetchLastRun]);

  const sparkline = useMemo(() => {
    if (scores.length === 0) return null;
    const W = 260, H = 60, PAD = 4;
    const min = Math.min(...scores), max = Math.max(...scores);
    const span = max - min || 1;
    const step = (W - PAD * 2) / Math.max(1, scores.length - 1);
    const points = scores.map((s, i) => {
      const x = PAD + i * step;
      const y = H - PAD - ((s - min) / span) * (H - PAD * 2);
      return `${x},${y}`;
    }).join(' ');
    return (
      <svg width={W} height={H} className="bg-slate-900 rounded">
        <polyline fill="none" stroke="#22c55e" strokeWidth={2} points={points} />
      </svg>
    );
  }, [scores]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-800 p-4 rounded-lg">
          <h3 className="text-slate-200 font-semibold mb-2">Run Controls</h3>
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <label className="text-slate-300">Iterations</label>
              <input className="bg-slate-700 px-2 py-1 rounded" type="number" value={form.iters as number} onChange={e=>onChange('iters', Number(e.target.value))} />
              <label className="text-slate-300">Generations</label>
              <input className="bg-slate-700 px-2 py-1 rounded" type="number" value={form.gens as number} onChange={e=>onChange('gens', Number(e.target.value))} />
              <label className="text-slate-300">Data Dir</label>
              <input className="bg-slate-700 px-2 py-1 rounded" value={form.data_dir as string} onChange={e=>onChange('data_dir', e.target.value)} placeholder="./data" />
              <label className="text-slate-300">Base Config</label>
              <input className="bg-slate-700 px-2 py-1 rounded" value={form.base_config as string} onChange={e=>onChange('base_config', e.target.value)} placeholder="configs/crypto.toml" />
              <label className="text-slate-300">BT Top</label>
              <input className="bg-slate-700 px-2 py-1 rounded" type="number" value={form.bt_top as number} onChange={e=>onChange('bt_top', Number(e.target.value))} />
            </div>
            <div className="flex items-center space-x-4 mt-2">
              <button onClick={startRun} className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-white">Start Run</button>
              <button onClick={stopRun} className="px-4 py-2 bg-slate-600 hover:bg-slate-500 rounded text-white">Stop</button>
              <span className="text-slate-400">Status: {status}</span>
            </div>
          </div>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <h3 className="text-slate-200 font-semibold mb-2">Live Sharpe(best)</h3>
          <div className="flex items-center space-x-4">
            {sparkline}
            <div className="text-slate-300">
              <div>Points: {scores.length}</div>
              <div>Best: {best === null ? '-' : best.toFixed(3)}</div>
            </div>
          </div>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <h3 className="text-slate-200 font-semibold mb-2">Current Candidate</h3>
          <div className="text-slate-300 text-sm">
            <div>Idx: {candidate ? `${candidate.idx}/${candidate.total}` : '-'}</div>
            <div className="truncate" title={candidate?.params || ''}>Params: {candidate?.params || '-'}</div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800 p-4 rounded-lg">
        <h3 className="text-slate-200 font-semibold mb-2">Last Run</h3>
        <div className="flex items-center justify-between text-sm text-slate-300">
          <div>Run dir: {finalInfo.run_dir || '-'}</div>
          <div>Best Sharpe: {finalInfo.sharpe_best === null ? '-' : finalInfo.sharpe_best.toFixed(3)}</div>
          <button onClick={fetchLastRun} className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded">Refresh</button>
        </div>
      </div>

      <div className="bg-slate-800 p-4 rounded-lg">
        <h3 className="text-slate-200 font-semibold mb-2">Live Log</h3>
        <div className="h-64 overflow-auto bg-slate-900 rounded p-2 text-green-300 text-xs font-mono whitespace-pre-wrap">
          {logs.slice(-500).map((l, i) => (<div key={i}>{l}</div>))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

