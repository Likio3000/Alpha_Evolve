import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import QuantileRibbon from './components/QuantileRibbon'
import RampWeights from './components/RampWeights'
import NoveltyTrend from './components/NoveltyTrend'
import HofTable from './components/HofTable'
import JaccardHeatmap from './components/JaccardHeatmap'
import Inspector from './components/Inspector'
import FitnessHistogram from './components/FitnessHistogram'
import TopKScatter from './components/TopKScatter'

type EventMsg =
  | { type: 'status'; msg: string; code?: number; args?: string[] }
  | { type: 'candidate'; idx: number; total: number; params: string; raw: string }
  | { type: 'score'; sharpe_best: number; raw: string }
  | { type: 'final'; run_dir: string; sharpe_best: number | null }
  | { type: 'diag'; data: any }
  | { type: 'log'; raw: string }

const API = (typeof window !== 'undefined' ? window.location.origin.replace(/:\d+$/, ':8000') : 'http://127.0.0.1:8000')

const App: React.FC = () => {
  const [status, setStatus] = useState('idle')
  const [jobId, setJobId] = useState<string | null>(null)
  const [scores, setScores] = useState<number[]>([])
  const [best, setBest] = useState<number | null>(null)
  const [candidate, setCandidate] = useState<{ idx: number; total: number; params: string } | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [lastRun, setLastRun] = useState<{ run_dir: string | null; sharpe_best: number | null }>({ run_dir: null, sharpe_best: null })
  const [gens, setGens] = useState<any[]>([])
  const [selectedProg, setSelectedProg] = useState<any | null>(null)
  const esRef = useRef<EventSource | null>(null)

  const start = useCallback(async () => {
    setStatus('starting')
    setScores([])
    setBest(null)
    setCandidate(null)
    setLogs([])
    setGens([])
    const payload = {
      iters: 1,
      gens: 25,
      bt_top: 5,
      no_clean: false,
      dry_run: false,
      selection_metric: 'auto',
      ramp_fraction: 0.33,
      ramp_min_gens: 5,
      novelty_boost_w: 0.02,
    }
    const resp = await fetch(`${API}/api/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
    const data = await resp.json()
    setJobId(data.job_id)
    setStatus('running')
    const es = new EventSource(`${API}/api/events/${data.job_id}`)
    esRef.current = es
    es.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as EventMsg
        if (msg.type === 'status') setStatus(msg.msg)
        else if (msg.type === 'candidate') setCandidate({ idx: msg.idx, total: msg.total, params: msg.params })
        else if (msg.type === 'score') {
          setScores(prev => { const next = prev.concat(msg.sharpe_best); setBest(b => (b==null? msg.sharpe_best : Math.max(b, msg.sharpe_best))); return next })
        }
        else if (msg.type === 'final') setLastRun({ run_dir: msg.run_dir, sharpe_best: msg.sharpe_best })
        else if (msg.type === 'diag') setGens(prev => prev.concat(msg.data))
        else if (msg.type === 'log') setLogs(prev => (prev.length > 1000 ? prev.slice(-1000) : prev).concat(msg.raw))
      } catch {}
    }
    es.onerror = () => { setStatus('disconnected'); es.close(); }
  }, [])

  const stop = useCallback(async () => {
    if (!jobId) return
    try { await fetch(`${API}/api/stop/${jobId}`, { method: 'POST' }) } catch {}
    if (esRef.current) { esRef.current.close(); esRef.current = null }
    setStatus('stopped')
  }, [jobId])

  const sparkline = useMemo(() => {
    if (scores.length === 0) return null
    const W = 320, H = 80, PAD = 6
    const min = Math.min(...scores), max = Math.max(...scores)
    const span = max - min || 1
    const step = (W - PAD*2) / Math.max(1, scores.length - 1)
    const pts = scores.map((s, i) => `${PAD + i*step},${H - PAD - ((s-min)/span)*(H-PAD*2)}`).join(' ')
    return <svg width={W} height={H} style={{background:'#0f172a', borderRadius:6}}><polyline fill="none" stroke="#22c55e" strokeWidth={2} points={pts}/></svg>
  }, [scores])

  const genCount = gens.length
  const lastBest = gens[genCount-1]?.best?.fitness
  const lastMedian = gens[genCount-1]?.pop_quantiles?.median
  const lastHof = gens[genCount-1]?.hof || []
  const lastHist = gens[genCount-1]?.pop_hist
  const lastTopK = gens[genCount-1]?.topK || []

  return (
    <div style={{fontFamily:'system-ui, sans-serif', color:'#e2e8f0', background:'#020617', minHeight:'100vh', padding:'16px'}}>
      <h1 style={{fontSize:24, fontWeight:700, marginBottom:8}}>Alpha Evolve — Live Dashboard</h1>
      <div style={{marginBottom:16, color:'#94a3b8'}}>Monitor evolution in real time. Config stays in TOML; this is an insight companion.</div>

      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12}}>
        <div style={{background:'#0b1220', padding:12, borderRadius:8}}>
          <div style={{fontWeight:600, marginBottom:8}}>Run Controls</div>
          <div style={{display:'flex', gap:8}}>
            <button onClick={start} style={{background:'#7c3aed', border:'none', color:'#fff', padding:'8px 12px', borderRadius:6}}>Start</button>
            <button onClick={stop} style={{background:'#475569', border:'none', color:'#fff', padding:'8px 12px', borderRadius:6}}>Stop</button>
            <div style={{alignSelf:'center', color:'#94a3b8'}}>Status: {status}</div>
          </div>
        </div>
        <div style={{background:'#0b1220', padding:12, borderRadius:8}}>
          <div style={{fontWeight:600, marginBottom:8}}>Sharpe(best) Sparkline</div>
          <div style={{display:'flex', gap:12, alignItems:'center'}}>
            {sparkline}
            <div style={{color:'#94a3b8'}}>
              <div>Points: {scores.length}</div>
              <div>Best: {best==null? '-' : best.toFixed(3)}</div>
            </div>
          </div>
        </div>
        <div style={{background:'#0b1220', padding:12, borderRadius:8}}>
          <div style={{fontWeight:600, marginBottom:8}}>Current Candidate</div>
          <div style={{color:'#94a3b8'}}>
            <div>Idx: {candidate? `${candidate.idx}/${candidate.total}`: '-'}</div>
            <div title={candidate?.params || ''} style={{whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis'}}>Params: {candidate?.params || '-'}</div>
          </div>
        </div>
      </div>

      <div style={{display:'grid', gridTemplateColumns:'2fr 1fr', gap:12, marginTop:12}}>
        <div style={{background:'#0b1220', padding:12, borderRadius:8}}>
          <div style={{fontWeight:600, marginBottom:8}}>Generation Snapshot</div>
          <div style={{color:'#94a3b8'}}>Gens: {genCount} · Last best={lastBest ?? '-'} · Last median={lastMedian ?? '-'}</div>
          <div style={{marginTop:12}}>
            <QuantileRibbon gens={gens} width={640} height={220} />
          </div>
          <div style={{marginTop:8, maxHeight:220, overflow:'auto', fontSize:12}}>
            <table style={{width:'100%', borderCollapse:'collapse'}}>
              <thead><tr style={{color:'#94a3b8'}}><th style={{textAlign:'left'}}>Gen</th><th>BestFit</th><th>Median</th><th>P95</th><th>P25</th><th>Eval(s)</th></tr></thead>
              <tbody>
                {gens.slice(-50).map((g:any) => (
                  <tr key={g.generation}>
                    <td>{g.generation}</td>
                    <td>{g.best?.fitness?.toFixed?.(3) ?? '-'}</td>
                    <td>{g.pop_quantiles?.median?.toFixed?.(3) ?? '-'}</td>
                    <td>{g.pop_quantiles?.p95?.toFixed?.(3) ?? '-'}</td>
                    <td>{g.pop_quantiles?.p25?.toFixed?.(3) ?? '-'}</td>
                    <td>{g.gen_eval_seconds?.toFixed?.(2) ?? '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{marginTop:12}}>
            <RampWeights gens={gens} width={640} height={160} />
          </div>
          <div style={{marginTop:12}}>
            <NoveltyTrend gens={gens} width={640} height={140} />
          </div>
        </div>
        <div style={{background:'#0b1220', padding:12, borderRadius:8}}>
          <div style={{fontWeight:600, marginBottom:8}}>Last Run</div>
          <div style={{color:'#94a3b8', fontSize:14}}>
            <div>Dir: {lastRun.run_dir || '-'}</div>
            <div>Best Sharpe: {lastRun.sharpe_best==null? '-' : lastRun.sharpe_best.toFixed(3)}</div>
          </div>
        </div>
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1.4fr 1fr', gap:12, marginTop:12}}>
        <div style={{background:'#0b1220', padding:12, borderRadius:8}}>
          <div style={{fontWeight:600, marginBottom:8}}>HOF Snapshot (last gen)</div>
          <HofTable items={lastHof} onSelect={setSelectedProg} />
          <div style={{marginTop:12}}>
            <JaccardHeatmap hofOpcodes={gens[genCount-1]?.hof_opcodes} programs={lastHof.map((h:any)=> h.program)} width={480} height={320} />
          </div>
        </div>
        <div style={{background:'#0b1220', padding:12, borderRadius:8}}>
          <div style={{fontWeight:600, marginBottom:8}}>Inspector</div>
          <Inspector item={selectedProg} />
        </div>
      </div>

      <div style={{background:'#0b1220', padding:12, borderRadius:8, marginTop:12}}>
        <div style={{fontWeight:600, marginBottom:8}}>Live Log</div>
        <div style={{maxHeight:260, overflow:'auto', background:'#020617', padding:8, borderRadius:6, fontFamily:'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize:12, color:'#22c55e'}}>
          {logs.slice(-400).map((l, i) => <div key={i}>{l}</div>)}
        </div>
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginTop:12}}>
        <div style={{background:'#0b1220', padding:12, borderRadius:8}}>
          <FitnessHistogram hist={lastHist} width={480} height={200} />
        </div>
        <div style={{background:'#0b1220', padding:12, borderRadius:8}}>
          <TopKScatter topK={lastTopK} width={480} height={220} />
        </div>
      </div>
    </div>
  )
}

export default App
