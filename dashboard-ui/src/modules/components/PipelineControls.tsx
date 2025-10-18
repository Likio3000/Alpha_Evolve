import React, { useEffect, useId, useState } from "react";
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
  const formId = useId();
  const dataset = defaultDataset;
  const datasetLabel = dataset === "sp500" ? "S&P 500 (daily)" : dataset;
  const [generationInput, setGenerationInput] = useState("5");
  const [popSizeInput, setPopSizeInput] = useState("100");
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

  const parseWholeNumber = (value: string): number | null => {
    const trimmed = value.trim();
    if (trimmed === "") {
      return null;
    }
    const numeric = Number(trimmed);
    if (!Number.isInteger(numeric)) {
      return null;
    }
    return numeric;
  };

  const parsedGenerations = parseWholeNumber(generationInput);
  const parsedPopSize = parseWholeNumber(popSizeInput);
  const hasInvalidInput = parsedGenerations === null || parsedPopSize === null;
  const hasZeroValue =
    (parsedGenerations !== null && parsedGenerations === 0) ||
    (parsedPopSize !== null && parsedPopSize === 0);
  const inlineNotice = hasZeroValue
    ? "Generations and population size must be greater than zero before launching."
    : hasInvalidInput
      ? "Enter whole numbers for generations and population size."
      : null;

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (parsedGenerations === null || parsedPopSize === null || parsedGenerations === 0 || parsedPopSize === 0) {
      setMessage("Please provide non-zero whole numbers for generations and population size.");
      return;
    }
    const overrides = { pop_size: parsedPopSize };
    const payload: PipelineRunRequest = {
      generations: parsedGenerations,
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
      <div className="panel-header">
        <h2>Pipeline Controls</h2>
        <div className="panel-actions">
          <button
            className="btn btn-primary"
            type="submit"
            form={formId}
            disabled={busy || hasInvalidInput || hasZeroValue}
          >
            {busy ? "Running…" : "Launch Pipeline"}
          </button>
        </div>
      </div>
      <form id={formId} className="form-grid" onSubmit={handleSubmit}>
        <label className="form-field">
          <span className="form-label">Dataset preset</span>
          <input type="text" value={datasetLabel} readOnly />
        </label>

        <label className="form-field">
          <span className="form-label">Generations</span>
          <input
            type="number"
            step={1}
            value={generationInput}
            onChange={(event) => setGenerationInput(event.target.value)}
            disabled={busy}
          />
        </label>

        <label className="form-field">
          <span className="form-label">Population size</span>
          <input
            type="number"
            step={1}
            value={popSizeInput}
            onChange={(event) => setPopSizeInput(event.target.value)}
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

        {inlineNotice ? <p className="form-warning">{inlineNotice}</p> : null}
        {message ? <div className="form-message">{message}</div> : null}
      </form>
    </section>
  );
}
