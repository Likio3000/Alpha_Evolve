import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchConfigDefaults,
  fetchConfigList,
  fetchConfigPresetValues,
  fetchConfigPresets,
  saveConfigPreset,
} from "../api";
import {
  ConfigDefaultsResponse,
  ConfigListItem,
  ConfigPresetValues,
  Scalar,
} from "../types";

type PresetKey = string;

type SettingsPanelProps = {
  onNotify?: (message: string) => void;
};

interface TableProps {
  title: string;
  defaults: Record<string, Scalar> | null;
  active: Record<string, Scalar> | null;
  filterText: string;
  showChangedOnly: boolean;
}

function formatValue(value: Scalar | undefined): string {
  if (value === undefined) {
    return "n/a";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4).replace(/\.0+$/, ".0");
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return value;
}

function ConfigTable({ title, defaults, active, filterText, showChangedOnly }: TableProps): React.ReactElement {
  const filter = filterText.trim().toLowerCase();

  const entries = useMemo(() => {
    const keys = new Set<string>();
    if (defaults) {
      Object.keys(defaults).forEach((key) => keys.add(key));
    }
    if (active) {
      Object.keys(active).forEach((key) => keys.add(key));
    }
    return Array.from(keys)
      .sort((a, b) => a.localeCompare(b))
      .map((key) => {
        const baseValue = defaults ? defaults[key] : undefined;
        const currentValue = active ? active[key] : undefined;
        const changed =
          currentValue !== undefined &&
          baseValue !== undefined &&
          String(currentValue) !== String(baseValue);
        return {
          key,
          baseValue,
          currentValue,
          changed,
        };
      })
      .filter((entry) => {
        if (showChangedOnly && !entry.changed) {
          return false;
        }
        if (!filter) {
          return true;
        }
        const haystack = `${entry.key} ${formatValue(entry.currentValue ?? entry.baseValue ?? "n/a")}`.toLowerCase();
        return haystack.includes(filter);
      });
  }, [active, defaults, filter, showChangedOnly]);

  return (
    <div className="settings-table">
      <div className="settings-table__header">
        <h3>{title}</h3>
        <span className="settings-table__count">{entries.length} {entries.length === 1 ? "field" : "fields"}</span>
      </div>
      <div className="settings-table__body">
        {entries.map(({ key, baseValue, currentValue, changed }) => {
          const display = currentValue ?? baseValue;
          return (
            <div key={key} className={changed ? "settings-card settings-card--changed" : "settings-card"}>
              <div className="settings-card__top">
                <div className="settings-card__key">{key}</div>
                {changed ? <span className="settings-card__badge">Changed</span> : null}
              </div>
              <div className="settings-card__value">{formatValue(display)}</div>
              {baseValue !== undefined ? (
                <div className={changed ? "settings-card__meta" : "settings-card__meta settings-card__meta--muted"}>
                  Default: {formatValue(baseValue)}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      {!entries.length ? <p className="muted">No parameters found.</p> : null}
    </div>
  );
}

export function SettingsPanel({ onNotify }: SettingsPanelProps): React.ReactElement {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [defaults, setDefaults] = useState<ConfigDefaultsResponse | null>(null);
  const [presets, setPresets] = useState<Record<string, string>>({});
  const [configs, setConfigs] = useState<ConfigListItem[]>([]);
  const [activePreset, setActivePreset] = useState<PresetKey | null>(null);
  const [activeValues, setActiveValues] = useState<ConfigPresetValues | null>(null);
  const [presetLoading, setPresetLoading] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [saving, setSaving] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [showChangedOnly, setShowChangedOnly] = useState(false);

  const notify = useCallback((message: string) => {
    if (onNotify) {
      onNotify(message);
    }
  }, [onNotify]);

  const resetActive = useCallback(() => {
    setActivePreset(null);
    setActiveValues(null);
    setSaveName("");
    setSearchText("");
    setShowChangedOnly(false);
  }, []);

  const loadInitialData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [defaultsResp, presetsResp, configsResp] = await Promise.all([
        fetchConfigDefaults(),
        fetchConfigPresets(),
        fetchConfigList(),
      ]);
      setDefaults(defaultsResp);
      setPresets(presetsResp);
      setConfigs(configsResp);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadInitialData();
  }, [loadInitialData]);

  const applyPreset = useCallback(async (label: PresetKey, params: { dataset?: string; path?: string }) => {
    setPresetLoading(true);
    setError(null);
    try {
      const values = await fetchConfigPresetValues(params);
      setActivePreset(label);
      setActiveValues(values);
      setSaveName(`${label}-copy-${new Date().toISOString().slice(0, 10)}`);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setError(detail);
      notify(`Failed to load preset: ${detail}`);
    } finally {
      setPresetLoading(false);
    }
  }, [notify]);

  const handleLoadDataset = useCallback((key: string) => {
    void applyPreset(key, { dataset: key });
  }, [applyPreset]);

  const handleLoadConfig = useCallback((item: ConfigListItem) => {
    void applyPreset(item.name, { path: item.path });
  }, [applyPreset]);

  const handleSave = useCallback(async () => {
    if (!defaults) {
      return;
    }
    const evolution = activeValues?.evolution ?? defaults.evolution;
    const backtest = activeValues?.backtest ?? defaults.backtest;
    const name = saveName.trim();
    if (!name) {
      setError("Enter a filename to save the configuration.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const payload = await saveConfigPreset({ name, evolution, backtest });
      notify(`Saved configuration to ${payload.saved}`);
      setConfigs((prev) => [{ name, path: payload.saved }, ...prev.filter((cfg) => cfg.name !== name)]);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setError(detail);
      notify(`Failed to save config: ${detail}`);
    } finally {
      setSaving(false);
    }
  }, [activeValues, defaults, notify, saveName]);

  const handleCopy = useCallback(() => {
    if (!defaults) {
      return;
    }
    const payload = {
      evolution: activeValues?.evolution ?? defaults.evolution,
      backtest: activeValues?.backtest ?? defaults.backtest,
    };
    const text = JSON.stringify(payload, null, 2);
    if (!navigator.clipboard) {
      notify("Clipboard API unavailable.");
      return;
    }
    void navigator.clipboard
      .writeText(text)
      .then(() => notify("Copied configuration JSON to clipboard."))
      .catch((err) => notify(err instanceof Error ? err.message : String(err)));
  }, [activeValues, defaults, notify]);

  const activeEvolution = activeValues?.evolution ?? defaults?.evolution ?? null;
  const activeBacktest = activeValues?.backtest ?? defaults?.backtest ?? null;

  return (
    <section className="panel settings-panel">
      <div className="panel-header">
        <h2>Settings & Presets</h2>
        <div className="panel-actions">
          <button className="btn" onClick={() => void loadInitialData()} disabled={loading}>
            Refresh
          </button>
          <button className="btn" onClick={resetActive} disabled={presetLoading}>
            Use defaults
          </button>
          <button className="btn" onClick={handleCopy} disabled={!defaults}>
            Copy JSON
          </button>
        </div>
      </div>

      {error ? <p className="muted error-text">{error}</p> : null}
      {loading ? <p className="muted">Loading configuration metadata…</p> : null}

      <div className="settings-grid">
        <aside className="settings-presets">
          <h3>Dataset Presets</h3>
          <div className="settings-presets__list">
            {Object.keys(presets).length === 0 ? (
              <p className="muted">No presets available.</p>
            ) : (
              Object.keys(presets)
                .sort()
                .map((key) => (
                  <button
                    key={key}
                    className={key === activePreset ? "settings-presets__btn settings-presets__btn--active" : "settings-presets__btn"}
                    onClick={() => handleLoadDataset(key)}
                    disabled={presetLoading}
                  >
                    {key}
                  </button>
                ))
            )}
          </div>

          <h3>Saved Configs</h3>
          <div className="settings-presets__list">
            {configs.length === 0 ? (
              <p className="muted">No stored configs yet.</p>
            ) : (
              configs.map((item) => (
                <button
                  key={item.path}
                  className={item.name === activePreset ? "settings-presets__btn settings-presets__btn--active" : "settings-presets__btn"}
                  onClick={() => handleLoadConfig(item)}
                  disabled={presetLoading}
                >
                  {item.name}
                </button>
              ))
            )}
          </div>
        </aside>

        <div className="settings-details">
          <div className="settings-toolbar">
            <div className="settings-toolbar__search">
              <input
                type="search"
                placeholder="Search settings…"
                value={searchText}
                onChange={(event) => setSearchText(event.target.value)}
              />
            </div>
            <label className="settings-toolbar__toggle">
              <input
                type="checkbox"
                checked={showChangedOnly}
                onChange={(event) => setShowChangedOnly(event.target.checked)}
              />
              Show changed only
            </label>
          </div>

          <ConfigTable
            title="Evolution"
            defaults={defaults?.evolution ?? null}
            active={activeEvolution}
            filterText={searchText}
            showChangedOnly={showChangedOnly}
          />
          <ConfigTable
            title="Backtest"
            defaults={defaults?.backtest ?? null}
            active={activeBacktest}
            filterText={searchText}
            showChangedOnly={showChangedOnly}
          />

          <div className="settings-save">
            <label className="form-field">
              <span className="form-label">Save as</span>
              <input
                type="text"
                value={saveName}
                onChange={(event) => setSaveName(event.target.value)}
                placeholder="configs/custom.toml"
                disabled={saving}
              />
            </label>
            <button className="btn btn-primary" onClick={handleSave} disabled={saving || loading}>
              {saving ? "Saving…" : "Save configuration"}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
