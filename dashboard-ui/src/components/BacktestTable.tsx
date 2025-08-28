import React from 'react'

interface Props {
  items: any[]
  onSelect?: (row: any) => void
}

const fmtPct = (v: any) => (typeof v === 'number' && isFinite(v) ? `${(v*100).toFixed(2)}%` : '-')
const fmtNum = (v: any, d=3) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '-')

const BacktestTable: React.FC<Props> = ({ items = [], onSelect }) => {
  if (!items || items.length === 0) return <div style={{color:'#94a3b8', fontSize:12}}>No backtest results yet</div>
  const rows = [...items].sort((a,b)=> (b.Sharpe??0) - (a.Sharpe??0))
  return (
    <div style={{maxHeight:300, overflow:'auto', fontSize:12}}>
      <table style={{width:'100%', borderCollapse:'collapse'}}>
        <thead>
          <tr style={{color:'#94a3b8'}}>
            <th style={{textAlign:'left'}}>Alpha</th>
            <th>Sharpe</th>
            <th>AnnRet</th>
            <th>MaxDD</th>
            <th>Turnover</th>
            <th>Ops</th>
            <th>TS</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r:any, i:number)=> (
            <tr key={r.AlphaID || i} style={{cursor:'pointer'}} onClick={()=> onSelect && onSelect(r)}>
              <td>{r.AlphaID || `#${i+1}`}</td>
              <td>{fmtNum(r.Sharpe, 3)}</td>
              <td>{fmtPct(r.AnnReturn)}</td>
              <td>{fmtPct(r.MaxDD)}</td>
              <td>{fmtNum(r.Turnover, 4)}</td>
              <td>{r.Ops ?? '-'}</td>
              <td title={r.TimeseriesFile || ''} style={{whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis'}}>{r.TS || (r.TimeseriesFile ? r.TimeseriesFile.split('/').pop() : '-')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default BacktestTable

