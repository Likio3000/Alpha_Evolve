import React from "react";
import { BacktestRow } from "../types";

interface BacktestTableProps {
  rows: BacktestRow[];
  selected?: string | null;
  onSelect?: (row: BacktestRow) => void;
}

const numberFormatter = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
});

const percentFormatter = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function formatNumber(value: number | null | undefined, formatter = numberFormatter): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return formatter.format(value);
}

export function BacktestTable({ rows, selected, onSelect }: BacktestTableProps): React.ReactElement {
  return (
    <div className="panel panel-main">
      <div className="panel-header">
        <h2>Backtest Summary</h2>
        {rows.length ? <span className="muted">{rows.length} alphas</span> : null}
      </div>
      <div className="table-scroll">
        <table className="bt-table">
          <thead>
            <tr>
              <th>Alpha</th>
              <th>Sharpe</th>
              <th>AnnRet</th>
              <th>AnnVol</th>
              <th>MaxDD</th>
              <th>Turnover</th>
              <th>Ops</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const key = row.TS || row.AlphaID || "";
              const isSelected = selected === key;
              return (
                <tr
                  key={key}
                  className={isSelected ? "bt-row bt-row--active" : "bt-row"}
                  onClick={() => onSelect?.(row)}
                >
                  <td>
                    <span className="bt-alpha-id">{row.AlphaID || "n/a"}</span>
                    <span className="bt-alpha-path">{row.TimeseriesFile || row.TS || ""}</span>
                  </td>
                  <td>{formatNumber(row.Sharpe)}</td>
                  <td>{formatNumber(row.AnnReturn, percentFormatter)}</td>
                  <td>{formatNumber(row.AnnVol, percentFormatter)}</td>
                  <td>{formatNumber(row.MaxDD, percentFormatter)}</td>
                  <td>{formatNumber(row.Turnover, percentFormatter)}</td>
                  <td>{row.Ops ?? "n/a"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.length === 0 ? <p className="muted">Select a run to view its backtest summary.</p> : null}
    </div>
  );
}
