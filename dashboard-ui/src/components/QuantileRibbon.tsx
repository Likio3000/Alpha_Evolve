import React, { useMemo } from 'react'

interface Props {
  gens: any[]
  width?: number
  height?: number
}

const QuantileRibbon: React.FC<Props> = ({ gens, width=640, height=220 }) => {
  const PAD = 24
  const data = useMemo(() => gens.map(g => ({
    gen: g.generation,
    best: g?.pop_quantiles?.best ?? g?.best?.fitness ?? null,
    p95: g?.pop_quantiles?.p95 ?? null,
    med: g?.pop_quantiles?.median ?? null,
    p25: g?.pop_quantiles?.p25 ?? null,
  })), [gens])
  const n = data.length
  if (n === 0) return <div style={{color:'#94a3b8', fontSize:12}}>No data</div>
  const xs = data.map((_, i) => i)
  const ys = data.flatMap(d => [d.p25, d.med, d.p95, d.best]).filter(v => typeof v === 'number') as number[]
  const xStep = (width - 2*PAD) / Math.max(1, n-1)
  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  const ySpan = yMax - yMin || 1
  const yScale = (v:number|null) => {
    if (v==null) return null
    return height - PAD - ((v - yMin)/ySpan) * (height - 2*PAD)
  }
  const xScale = (i:number) => PAD + i * xStep

  const medLine = data.map((d, i) => {
    const y = yScale(d.med)
    return y==null? null : `${xScale(i)},${y}`
  }).filter(Boolean).join(' ')

  // Ribbon path for p25→p95 area
  const upper = data.map((d,i)=>({x:xScale(i), y:yScale(d.p95)})).filter(p=>p.y!=null) as {x:number,y:number}[]
  const lower = data.map((d,i)=>({x:xScale(i), y:yScale(d.p25)})).filter(p=>p.y!=null) as {x:number,y:number}[]
  const areaPath = upper.length && lower.length ? `M ${upper.map(p=>`${p.x},${p.y}`).join(' L ')} L ${[...lower].reverse().map(p=>`${p.x},${p.y}`).join(' L ')} Z` : ''

  const bestLine = data.map((d,i)=>{
    const y = yScale(d.best)
    return y==null? null : `${xScale(i)},${y}`
  }).filter(Boolean).join(' ')

  return (
    <svg width={width} height={height} style={{background:'#0f172a', borderRadius:8}}>
      <g>
        {/* axes (minimal) */}
        <line x1={PAD} y1={height-PAD} x2={width-PAD} y2={height-PAD} stroke="#334155"/>
        <line x1={PAD} y1={PAD} x2={PAD} y2={height-PAD} stroke="#334155"/>

        {/* ribbon */}
        {areaPath && <path d={areaPath} fill="#22d3ee22" stroke="none"/>}

        {/* median */}
        {medLine && <polyline points={medLine} fill="none" stroke="#22d3ee" strokeWidth={2}/>}   
        {/* best */}
        {bestLine && <polyline points={bestLine} fill="none" stroke="#a78bfa" strokeWidth={2} strokeDasharray="4 3"/>}
      </g>
    </svg>
  )
}

export default QuantileRibbon

