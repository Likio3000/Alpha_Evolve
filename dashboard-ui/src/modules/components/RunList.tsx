import React from "react";
import { RunSummary } from "../types";

interface RunListProps {
  runs: RunSummary[];
  selected?: string | null;
  onSelect?: (run: RunSummary) => void;
  onRelabel?: (run: RunSummary, newLabel: string) => void;
  onRefresh?: () => void;
  loading?: boolean;
}

export function RunList({
  runs,
  selected,
  loading,
  onSelect,
  onRelabel,
  onRefresh,
}: RunListProps): React.ReactElement {
  const handleRelabel = (run: RunSummary) => {
    if (!onRelabel) return;
    const next = window.prompt("Enter label for run:", run.label ?? "");
    if (next === null) return;
    onRelabel(run, next.trim());
  };

  return (
    <section className="panel panel-controls">
      <div className="panel-header">
        <h2>Recent Runs</h2>
        <div className="panel-actions">
          <button className="btn" onClick={onRefresh} disabled={loading}>
            Refresh
          </button>
        </div>
      </div>
      {loading ? <p className="muted">Loading runs…</p> : null}
      {!loading && runs.length === 0 ? <p className="muted">No runs found.</p> : null}
      <ul className="run-list">
        {runs.map((run) => {
          const isSelected = selected === run.path;
          return (
            <li
              key={run.path}
              className={isSelected ? "run-list__item run-list__item--active" : "run-list__item"}
              onClick={() => onSelect?.(run)}
            >
              <div className="run-list__meta">
                <span className="run-list__name">{run.label || run.name}</span>
              </div>
              <button
                className="btn btn--tiny run-list__relabel"
                onClick={(evt) => {
                  evt.stopPropagation();
                  handleRelabel(run);
                }}
              >
                Rename
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
