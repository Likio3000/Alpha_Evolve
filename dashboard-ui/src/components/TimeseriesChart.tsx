import React, { useMemo } from 'react'

interface Props {
  series: { date: string[]; equity?: number[]; drawdown?: number[]; ret_net?: number[] }
  width?: number
  height?: number
}

const TimeseriesChart: React.FC<Props> = ({ series, width=520, height=280 }) => {
  const PAD = 28
  const n = series?.date?.length || 0
  if (!n) return <div style={{color:'#94a3b8', fontSize:12}}>No timeseries</div>
  const xs = Array.from({length:n}, (_,i)=>i)
  const eq = series.equity || []
  const dd = series.drawdown || []
  const eqMin = Math.min(...eq.filter(Number.isFinite), 0)
  const eqMax = Math.max(...eq.filter(Number.isFinite), 1)
  const eqSpan = (eqMax - eqMin) || 1
  const ddMin = Math.min(...dd.filter(Number.isFinite), 0)
  const ddMax = Math.max(...dd.filter(Number.isFinite), 0)
  const xStep = (width - 2*PAD) / Math.max(1, n-1)
  const x = (i:number)=> PAD + i * xStep
  const yEq = (v:number)=> (height - PAD) - ((v - eqMin)/eqSpan) * (height - 2*PAD)
  const yDd = (v:number)=> (height - PAD) - ((v - ddMin)/(ddMax - ddMin || 1)) * (height - 2*PAD)

  const eqPts = useMemo(()=> eq.map((v,i)=> Number.isFinite(v)? `${x(i)},${yEq(v)}` : null).filter(Boolean).join(' '), [eq, n])
  const ddPts = useMemo(()=> dd.map((v,i)=> Number.isFinite(v)? `${x(i)},${yDd(v)}` : null).filter(Boolean).join(' '), [dd, n])

  return (
    <svg width={width} height={height} style={{background:'#0f172a', borderRadius:8}}>
      <g>
        {/* axes */}
        <line x1={PAD} y1={height-PAD} x2={width-PAD} y2={height-PAD} stroke="#334155"/>
        <line x1={PAD} y1={PAD} x2={PAD} y2={height-PAD} stroke="#334155"/>
        {/* equity */}
        {eqPts && <polyline points={eqPts} fill="none" stroke="#22c55e" strokeWidth={2} />}
        {/* drawdown */}
        {ddPts && <polyline points={ddPts} fill="none" stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 3" />}
        {/* legend */}
        <rect x={PAD} y={8} width={10} height={10} fill="#22c55e"/>
        <text x={PAD+16} y={17} fill="#94a3b8" fontSize={12}>Equity</text>
        <rect x={PAD+80} y={8} width={10} height={10} fill="#ef4444"/>
        <text x={PAD+96} y={17} fill="#94a3b8" fontSize={12}>Drawdown</text>
      </g>
    </svg>
  )
}

export default TimeseriesChart

