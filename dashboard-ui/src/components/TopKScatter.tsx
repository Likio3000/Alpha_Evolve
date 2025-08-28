import React, { useMemo } from 'react'

interface Props {
  topK?: { ops?: number; fitness?: number }[]
  width?: number
  height?: number
}

const TopKScatter: React.FC<Props> = ({ topK = [], width=480, height=220 }) => {
  if (!topK || topK.length === 0) return <div style={{color:'#94a3b8', fontSize:12}}>No top-K data</div>
  const PAD = 24
  const pts = useMemo(()=> topK.map(t => ({ x: (t.ops ?? 0), y: (t.fitness ?? 0) })), [topK])
  const xs = pts.map(p=>p.x), ys= pts.map(p=>p.y)
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  const xScale = (x:number)=> PAD + (x - xMin)/(xMax - xMin || 1) * (width - 2*PAD)
  const yScale = (y:number)=> height - PAD - (y - yMin)/(yMax - yMin || 1) * (height - 2*PAD)

  return (
    <svg width={width} height={height} style={{background:'#0f172a', borderRadius:8}}>
      <g>
        <line x1={PAD} y1={height-PAD} x2={width-PAD} y2={height-PAD} stroke="#334155"/>
        <line x1={PAD} y1={PAD} x2={PAD} y2={height-PAD} stroke="#334155"/>
        {pts.map((p,i)=> (
          <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={4} fill="#a78bfa"/>
        ))}
        <text x={PAD} y={14} fill="#94a3b8" fontSize={12}>Top-K fitness vs ops</text>
      </g>
    </svg>
  )
}

export default TopKScatter

