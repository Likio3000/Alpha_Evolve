import React, { useMemo } from 'react'

interface Props {
  gens: any[]
  width?: number
  height?: number
}

const NoveltyTrend: React.FC<Props> = ({ gens, width=640, height=140 }) => {
  const PAD = 24
  const data = useMemo(() => gens.map((g:any, i:number)=> ({
    x: i,
    y: g?.novelty?.hof_mean_corr_best ?? null,
    gen: g?.generation
  })), [gens])
  const n = data.length
  if (n === 0) return <div style={{color:'#94a3b8', fontSize:12}}>No novelty data</div>
  const vals = data.map(d=>d.y).filter((v:any)=> typeof v === 'number') as number[]
  if (vals.length === 0) return <div style={{color:'#94a3b8', fontSize:12}}>No novelty data</div>
  const xStep = (width - 2*PAD) / Math.max(1, n-1)
  const yMin = 0, yMax = 1 // novelty as mean abs corr component in [0,1]
  const yScale = (v:number|null)=> v==null? null : (height - PAD - (v - yMin)/(yMax-yMin) * (height - 2*PAD))
  const xScale = (i:number)=> PAD + i*xStep
  const pts = data.map((d,i)=>{
    const y = yScale(d.y)
    return y==null? null : `${xScale(i)},${y}`
  }).filter(Boolean).join(' ')

  return (
    <svg width={width} height={height} style={{background:'#0f172a', borderRadius:8}}>
      <g>
        <line x1={PAD} y1={height-PAD} x2={width-PAD} y2={height-PAD} stroke="#334155"/>
        <line x1={PAD} y1={PAD} x2={PAD} y2={height-PAD} stroke="#334155"/>
        {/* target bands for novelty (lower corr => higher novelty) */}
        <rect x={PAD} y={yScale(0.2) as number} width={width-2*PAD} height={(yScale(0.4) as number)-(yScale(0.2) as number)} fill="#16a34a22"/>
        <rect x={PAD} y={yScale(0.4) as number} width={width-2*PAD} height={(yScale(0.6) as number)-(yScale(0.4) as number)} fill="#f59e0b22"/>
        <rect x={PAD} y={yScale(0.6) as number} width={width-2*PAD} height={(yScale(0.8) as number)-(yScale(0.6) as number)} fill="#ef444422"/>
        {pts && <polyline points={pts} fill="none" stroke="#f97316" strokeWidth={2}/>} 
        <text x={PAD} y={14} fill="#94a3b8" fontSize={12}>Novelty vs HOF (lower is better)</text>
      </g>
    </svg>
  )
}

export default NoveltyTrend

