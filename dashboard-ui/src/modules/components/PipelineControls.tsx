import React, { useState } from "react";
import { PipelineRunRequest } from "../types";

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
  const [dataset, setDataset] = useState(defaultDataset);
  const [generations, setGenerations] = useState(5);
  const [configPath, setConfigPath] = useState("");
  const [dataDir, setDataDir] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const payload: PipelineRunRequest = {
      generations,
      dataset: configPath ? undefined : dataset,
      config: configPath || undefined,
      data_dir: dataDir || undefined,
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
          <select
            value={dataset}
            onChange={(event) => setDataset(event.target.value)}
            disabled={busy || Boolean(configPath)}
          >
            <option value="sp500">SP500 (daily)</option>
          </select>
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
          <span className="form-label">Config path (optional)</span>
          <input
            type="text"
            placeholder="configs/sp500.toml"
            value={configPath}
            onChange={(event) => setConfigPath(event.target.value)}
            disabled={busy}
          />
        </label>

        <label className="form-field">
          <span className="form-label">Data directory (optional)</span>
          <input
            type="text"
            placeholder="./data"
            value={dataDir}
            onChange={(event) => setDataDir(event.target.value)}
            disabled={busy}
          />
        </label>

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
