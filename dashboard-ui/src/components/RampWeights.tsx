import React, { useMemo } from 'react'

interface Props {
  gens: any[]
  width?: number
  height?: number
}

const RampWeights: React.FC<Props> = ({ gens, width=640, height=160 }) => {
  const PAD = 24
  const series = useMemo(() => {
    const xs = gens.map((g:any, i:number) => i)
    const keys = ['corr_w','ic_std_w','turnover_w','sharpe_w'] as const
    const vals = keys.map(k => gens.map((g:any)=> g?.ramp?.[k] ?? null))
    return { xs, keys, vals }
  }, [gens])
  const n = series.xs.length
  if (n === 0) return <div style={{color:'#94a3b8', fontSize:12}}>No ramp data</div>
  const xStep = (width - 2*PAD) / Math.max(1, n-1)
  const all = series.vals.flat().filter(v => typeof v === 'number') as number[]
  const yMin = Math.min(0, ...all), yMax = Math.max(1, ...all)
  const ySpan = yMax - yMin || 1
  const yScale = (v:number|null) => v==null? null : (height - PAD - ((v - yMin)/ySpan) * (height - 2*PAD))
  const xScale = (i:number) => PAD + i * xStep
  const colors = ['#f59e0b','#ef4444','#10b981','#60a5fa']
  const labels = ['corr_w','ic_std_w','turnover_w','sharpe_w']

  return (
    <svg width={width} height={height} style={{background:'#0f172a', borderRadius:8}}>
      <g>
        <line x1={PAD} y1={height-PAD} x2={width-PAD} y2={height-PAD} stroke="#334155"/>
        <line x1={PAD} y1={PAD} x2={PAD} y2={height-PAD} stroke="#334155"/>
        {series.vals.map((arr, idx) => {
          const pts = arr.map((v,i)=>{
            const y = yScale(v as number|null)
            return y==null? null : `${xScale(i)},${y}`
          }).filter(Boolean).join(' ')
          return <polyline key={idx} points={pts} fill="none" stroke={colors[idx]} strokeWidth={2} />
        })}
        {/* legend */}
        {labels.map((lab, i)=> (
          <g key={lab}>
            <rect x={PAD + i*120} y={8} width={10} height={10} fill={colors[i]} />
            <text x={PAD + i*120 + 16} y={17} fill="#94a3b8" fontSize={12}>{lab}</text>
          </g>
        ))}
      </g>
    </svg>
  )
}

export default RampWeights

