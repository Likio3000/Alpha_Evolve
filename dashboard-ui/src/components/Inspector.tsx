import React from 'react'

const Inspector: React.FC<{ item:any|null }> = ({ item }) => {
  if (!item) return <div style={{color:'#94a3b8', fontSize:12}}>Select a HOF entry to inspect</div>
  const fitness = item.fitness ?? item.metrics?.fitness
  const ic = item.mean_ic ?? item.metrics?.mean_ic
  const ops = item.ops ?? item.program?.size
  const program = item.program
  return (
    <div style={{fontSize:12, color:'#94a3b8'}}>
      <div><b>Fingerprint:</b> {(item.fp || item.fingerprint || '').slice(0,16)}</div>
      <div><b>Generation:</b> {item.gen || item.generation || '-'}</div>
      <div><b>Fitness:</b> {fitness?.toFixed?.(4) ?? '-'}</div>
      <div><b>IC:</b> {ic?.toFixed?.(4) ?? '-'}</div>
      <div><b>Ops:</b> {ops ?? '-'}</div>
      <div style={{marginTop:8}}><b>Program:</b></div>
      <div style={{whiteSpace:'pre-wrap', background:'#020617', color:'#e2e8f0', padding:8, borderRadius:6, maxHeight:180, overflow:'auto'}}>{program || '(not available)'}</div>
    </div>
  )
}

export default Inspector

