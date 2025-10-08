import React, { useMemo } from "react";
import { ParamMetaItem, RunDetails, RunSummary } from "../types";

interface RunDetailsPanelProps {
  run: RunSummary | null;
  details: RunDetails | null;
  loading: boolean;
  error: string | null;
  onRefresh?: () => void;
  docs?: ReadonlyMap<string, ParamMetaItem>;
  docsLoading?: boolean;
  docsError?: string | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "n/a";
    }
    return Number.isInteger(value) ? value.toString() : value.toFixed(4).replace(/\.0+$/, ".0");
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (Array.isArray(value)) {
    return value.map((item) => formatValue(item)).join(", ");
  }
  if (isRecord(value)) {
    try {
      return JSON.stringify(value, null, 2);
    } catch (error) {
      return "[object]";
    }
  }
  return String(value);
}

function flattenRecord(record: Record<string, unknown>, prefix?: string): Array<{ key: string; value: unknown }> {
  const items: Array<{ key: string; value: unknown }> = [];
  Object.entries(record).forEach(([key, value]) => {
    const compoundKey = prefix ? `${prefix}.${key}` : key;
    if (isRecord(value)) {
      items.push(...flattenRecord(value, compoundKey));
    } else {
      items.push({ key: compoundKey, value });
    }
  });
  return items;
}

export function RunDetailsPanel({
  run,
  details,
  loading,
  error,
  onRefresh,
  docs,
  docsLoading,
  docsError,
}: RunDetailsPanelProps): React.ReactElement {
  const docLookup = docs ?? new Map<string, ParamMetaItem>();
  const payload = details?.uiContext?.payload;

  const [baseParams, overrideParams] = useMemo(() => {
    if (!isRecord(payload)) {
      return [[], []] as const;
    }
    const base: Array<{ key: string; value: unknown; help?: string; label?: string }> = [];
    const overrides: Array<{ key: string; value: unknown; help?: string; label?: string }> = [];
    Object.entries(payload).forEach(([key, value]) => {
      if (key === "overrides" && isRecord(value)) {
        flattenRecord(value).forEach((entry) => {
          const docKey = entry.key.split(".").pop() ?? entry.key;
          const meta = docLookup.get(entry.key) ?? docLookup.get(docKey);
          overrides.push({
            key: entry.key,
            value: entry.value,
            help: meta?.help,
            label: meta?.label,
          });
        });
      } else {
        const meta = docLookup.get(key);
        base.push({
          key,
          value,
          help: meta?.help,
          label: meta?.label,
        });
      }
    });
    return [base, overrides] as const;
  }, [payload, docLookup]);

  const pipelineArgs = Array.isArray(details?.uiContext?.pipeline_args)
    ? details?.uiContext?.pipeline_args ?? []
    : [];

  const submittedAt = details?.uiContext?.submitted_at ?? null;
  const sharpe = details?.sharpeBest ?? run?.sharpeBest ?? null;

  return (
    <section className="panel panel-run-details">
      <div className="panel-header">
        <h2>Run Parameters</h2>
        <div className="panel-actions">
          <button className="btn" onClick={onRefresh} disabled={!run || loading}>
            {loading ? "Refreshing…" : "Reload"}
          </button>
        </div>
      </div>

      {error ? <p className="muted error-text">{error}</p> : null}
      {docsError ? <p className="muted error-text">{docsError}</p> : null}
      {docsLoading ? <p className="muted">Loading parameter documentation…</p> : null}

      {!run ? (
        <p className="muted">Select a run to inspect its submission details.</p>
      ) : loading && !details ? (
        <p className="muted">Loading run metadata…</p>
      ) : !details ? (
        <p className="muted">No metadata captured for this run.</p>
      ) : (
        <>
          <div className="run-details__meta">
            <div>
              <span className="run-details__meta-label">Run directory</span>
              <span className="run-details__meta-value">{details.path || run.path}</span>
            </div>
            <div>
              <span className="run-details__meta-label">Label</span>
              <span className="run-details__meta-value">{run.label || "—"}</span>
            </div>
            <div>
              <span className="run-details__meta-label">Submitted</span>
              <span className="run-details__meta-value">{submittedAt ?? "unknown"}</span>
            </div>
            <div>
              <span className="run-details__meta-label">Best Sharpe</span>
              <span className="run-details__meta-value">
                {sharpe === undefined || sharpe === null ? "n/a" : sharpe.toFixed(3)}
              </span>
            </div>
          </div>

          <div className="run-param-section">
            <h3>Submitted options</h3>
            {baseParams.length === 0 ? (
              <p className="muted">No core parameters recorded.</p>
            ) : (
              <div className="run-param-grid">
                {baseParams.map((entry) => (
                  <div key={entry.key} className="run-param-item">
                    <div className="run-param-item__label">{entry.label ?? entry.key}</div>
                    <div className="run-param-item__value">{formatValue(entry.value)}</div>
                    {entry.help ? <div className="run-param-item__help">{entry.help}</div> : null}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="run-param-section">
            <h3>Overrides</h3>
            {overrideParams.length === 0 ? (
              <p className="muted">No overrides supplied.</p>
            ) : (
              <div className="run-param-grid">
                {overrideParams.map((entry) => (
                  <div key={entry.key} className="run-param-item">
                    <div className="run-param-item__label">{entry.label ?? entry.key}</div>
                    <div className="run-param-item__value">{formatValue(entry.value)}</div>
                    {entry.help ? <div className="run-param-item__help">{entry.help}</div> : null}
                  </div>
                ))}
              </div>
            )}
          </div>

          {pipelineArgs.length ? (
            <div className="run-param-section">
              <h3>CLI arguments</h3>
              <pre className="run-param-args">{pipelineArgs.join(" ")}</pre>
            </div>
          ) : null}

          {details.meta ? (
            <div className="run-param-section">
              <h3>Captured configs</h3>
              <div className="run-meta-grid">
                {Object.entries(details.meta).map(([key, value]) => (
                  <details key={key} className="run-meta-item">
                    <summary>{key}</summary>
                    <pre>{formatValue(value)}</pre>
                  </details>
                ))}
              </div>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}
