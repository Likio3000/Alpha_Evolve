import React, { useEffect, useState } from "react";
import { fetchConfigList } from "../api";
import { ConfigListItem, PipelineRunRequest } from "../types";

interface PipelineControlsProps {
  onSubmit: (payload: PipelineRunRequest) => Promise<void> | void;
  busy: boolean;
  defaultDataset?: string;
}

export function PipelineControls({
  onSubmit,
  busy,
  defaultDataset = "sp500",
}: PipelineControlsProps): React.ReactElement {
  const dataset = defaultDataset;
  const datasetLabel = dataset === "sp500" ? "S&P 500 (daily)" : dataset;
  const [generations, setGenerations] = useState(5);
  const [popSize, setPopSize] = useState(100);
  const [configPath, setConfigPath] = useState("");
  const [configs, setConfigs] = useState<ConfigListItem[]>([]);
  const [configLoading, setConfigLoading] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const loadConfigs = async () => {
      setConfigLoading(true);
      setConfigError(null);
      try {
        const items = await fetchConfigList();
        if (!cancelled) {
          setConfigs([...items].sort((a, b) => a.name.localeCompare(b.name)));
        }
      } catch (error) {
        if (!cancelled) {
          const detail = error instanceof Error ? error.message : String(error);
          setConfigError(detail);
        }
      } finally {
        if (!cancelled) {
          setConfigLoading(false);
        }
      }
    };
    void loadConfigs();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const sanitizedPopSize = Math.max(10, Math.min(2000, Math.round(Number.isFinite(popSize) ? popSize : 100)));
    const overrides = { pop_size: sanitizedPopSize };
    const payload: PipelineRunRequest = {
      generations,
      dataset: configPath ? undefined : dataset,
      config: configPath || undefined,
      overrides,
    };
    try {
      setMessage(null);
      await onSubmit(payload);
      setMessage("Submitted pipeline run.");
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      setMessage(detail);
    }
  };

  return (
    <section className="panel panel-controls">
      <h2>Pipeline Controls</h2>
      <form className="form-grid" onSubmit={handleSubmit}>
        <label className="form-field">
          <span className="form-label">Dataset preset</span>
          <input type="text" value={datasetLabel} readOnly />
        </label>

        <label className="form-field">
          <span className="form-label">Generations</span>
          <input
            type="number"
            min={1}
            max={1000}
            value={generations}
            onChange={(event) => setGenerations(Number(event.target.value) || 1)}
            disabled={busy}
          />
        </label>

        <label className="form-field">
          <span className="form-label">Population size</span>
          <input
            type="number"
            min={10}
            max={2000}
            step={10}
            value={popSize}
            onChange={(event) => {
              const raw = Number(event.target.value);
              if (Number.isFinite(raw)) {
                const clamped = Math.max(10, Math.min(2000, Math.round(raw)));
                setPopSize(clamped);
              } else {
                setPopSize(100);
              }
            }}
            disabled={busy}
          />
        </label>

        <label className="form-field">
          <span className="form-label">Config preset</span>
          <select
            value={configPath}
            onChange={(event) => setConfigPath(event.target.value)}
            disabled={busy || configLoading}
          >
            <option value="">None (use dataset defaults)</option>
            {configs.map((cfg) => (
              <option key={cfg.path} value={cfg.path}>
                {cfg.name}
              </option>
            ))}
          </select>
        </label>

        {configLoading ? <p className="muted">Loading configuration presets…</p> : null}
        {configError ? <p className="muted error-text">{configError}</p> : null}

        <div className="form-actions">
          <button className="btn btn-primary" type="submit" disabled={busy}>
            {busy ? "Running…" : "Launch Pipeline"}
          </button>
        </div>

        {message ? <div className="form-message">{message}</div> : null}
      </form>
    </section>
  );
}
