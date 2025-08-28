import React, { useMemo } from 'react'

interface Props {
  hofOpcodes?: string[][] // list of opcode lists (one per HOF entry)
  programs?: string[]     // fallback: raw program strings
  width?: number
  height?: number
}

function tokensFromProgram(p: string): string[] {
  // Extract opcodes like "out = opcode(" pattern
  const re = /(\w+)\s*=\s*(\w+)\s*\(/g
  const out: string[] = []
  let m: RegExpExecArray | null
  // eslint-disable-next-line no-cond-assign
  while (m = re.exec(p)) {
    const opcode = m[2]
    if (opcode) out.push(opcode)
  }
  return Array.from(new Set(out))
}

const JaccardHeatmap: React.FC<Props> = ({ hofOpcodes, programs, width=480, height=320 }) => {
  const sets = useMemo(()=>{
    if (hofOpcodes && hofOpcodes.length > 0) {
      return hofOpcodes.map(lst => new Set((lst||[]).map(String)))
    }
    const progs = (programs || []).filter(Boolean) as string[]
    if (progs.length === 0) return []
    return progs.map(p => new Set(tokensFromProgram(p)))
  }, [hofOpcodes, programs])
  const n = sets.length
  if (!n) return <div style={{color:'#94a3b8', fontSize:12}}>No structural data</div>

  const mat: number[][] = Array.from({length:n}, ()=> Array(n).fill(0))
  for (let i=0;i<n;i++) {
    for (let j=0;j<n;j++) {
      const a = sets[i], b = sets[j]
      const inter = new Set(Array.from(a).filter(x=> b.has(x)))
      const union = new Set<string>([...Array.from(a), ...Array.from(b)])
      const jacc = union.size === 0 ? 1 : inter.size / union.size
      mat[i][j] = 1 - jacc // distance (higher = more novel)
    }
  }

  const PAD = 30
  const cellW = (width - 2*PAD) / n
  const cellH = (height - 2*PAD) / n
  const cells: JSX.Element[] = []
  for (let i=0;i<n;i++) {
    for (let j=0;j<n;j++) {
      const v = mat[i][j] // 0..1
      // map to color (blue→yellow→red)
      const r = Math.round(255 * v)
      const g = Math.round(200 * (1 - Math.abs(v-0.5)*2))
      const b = Math.round(255 * (1 - v))
      const fill = `rgb(${r},${g},${b})`
      cells.push(<rect key={`${i}-${j}`} x={PAD + j*cellW} y={PAD + i*cellH} width={cellW} height={cellH} fill={fill} />)
    }
  }

  return (
    <svg width={width} height={height} style={{background:'#0f172a', borderRadius:8}}>
      <g>
        <text x={PAD} y={18} fill="#94a3b8" fontSize={12}>HOF structural distance (Jaccard, lower = similar)</text>
        {/* grid */}
        {cells}
        {/* ticks */}
        <line x1={PAD} y1={PAD} x2={PAD} y2={height-PAD} stroke="#334155"/>
        <line x1={PAD} y1={height-PAD} x2={width-PAD} y2={height-PAD} stroke="#334155"/>
      </g>
    </svg>
  )
}

export default JaccardHeatmap

