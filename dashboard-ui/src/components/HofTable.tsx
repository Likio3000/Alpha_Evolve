import React from 'react'

interface Props {
  items: any[]
  onSelect?: (item:any)=>void
}

const HofTable: React.FC<Props> = ({ items = [], onSelect }) => {
  if (!items || items.length === 0) return <div style={{color:'#94a3b8', fontSize:12}}>No HOF entries</div>
  return (
    <div style={{maxHeight:260, overflow:'auto', fontSize:12}}>
      <table style={{width:'100%', borderCollapse:'collapse'}}>
        <thead>
          <tr style={{color:'#94a3b8'}}>
            <th style={{textAlign:'left'}}>FP</th>
            <th>Gen</th>
            <th>Fitness</th>
            <th>IC</th>
            <th>Ops</th>
            <th>Program</th>
          </tr>
        </thead>
        <tbody>
          {items.map((h:any, i:number)=> (
            <tr key={h.fp || i} style={{cursor:'pointer'}} onClick={()=> onSelect && onSelect(h)}>
              <td>{(h.fp || h.fingerprint || '').slice(0,8)}</td>
              <td>{h.gen || h.generation || '-'}</td>
              <td>{(h.fitness ?? h.metrics?.fitness ?? '')?.toFixed?.(3) ?? '-'}</td>
              <td>{(h.mean_ic ?? h.metrics?.mean_ic ?? '')?.toFixed?.(3) ?? '-'}</td>
              <td>{h.ops ?? h.program?.size ?? '-'}</td>
              <td title={h.program || ''} style={{whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis'}}>{h.program || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default HofTable

