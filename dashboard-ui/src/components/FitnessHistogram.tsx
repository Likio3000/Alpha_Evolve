import React, { useMemo } from 'react'

interface Props {
  hist?: { edges: number[]; counts: number[] }
  width?: number
  height?: number
}

const FitnessHistogram: React.FC<Props> = ({ hist, width=480, height=200 }) => {
  if (!hist || !hist.edges || !hist.counts || hist.edges.length < 2) return <div style={{color:'#94a3b8', fontSize:12}}>No histogram data</div>
  const PAD = 24
  const bins = useMemo(()=>{
    const out: { x:number; w:number; c:number }[] = []
    for (let i=0;i<hist.counts.length;i++) {
      const x0 = hist.edges[i]
      const x1 = hist.edges[i+1]
      out.push({ x:(x0+x1)/2, w: x1 - x0, c: hist.counts[i] })
    }
    return out
  }, [hist])
  const maxC = Math.max(...hist.counts, 1)
  const minX = hist.edges[0], maxX = hist.edges[hist.edges.length-1]
  const xScale = (x:number)=> PAD + (x - minX)/(maxX - minX || 1) * (width - 2*PAD)
  const yScale = (c:number)=> height - PAD - (c/maxC) * (height - 2*PAD)

  const bars = bins.map((b,i)=>{
    const x0 = xScale(b.x - b.w/2)
    const x1 = xScale(b.x + b.w/2)
    const w = Math.max(1, x1 - x0)
    const y = yScale(b.c)
    const h = (height - PAD) - y
    return <rect key={i} x={x0} y={y} width={w} height={h} fill="#38bdf8"/>
  })

  return (
    <svg width={width} height={height} style={{background:'#0f172a', borderRadius:8}}>
      <g>
        <line x1={PAD} y1={height-PAD} x2={width-PAD} y2={height-PAD} stroke="#334155"/>
        <line x1={PAD} y1={PAD} x2={PAD} y2={height-PAD} stroke="#334155"/>
        {bars}
        <text x={PAD} y={14} fill="#94a3b8" fontSize={12}>Population fitness histogram</text>
      </g>
    </svg>
  )
}

export default FitnessHistogram

