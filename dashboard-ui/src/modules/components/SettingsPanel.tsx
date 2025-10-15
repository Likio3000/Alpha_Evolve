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
  choices?: Record<string, string[]>;
  disabled?: boolean;
  onChange: (key: string, value: Scalar) => void;
  onReset: (key: string) => void;
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

function ConfigTable({
  title,
  defaults,
  active,
  filterText,
  showChangedOnly,
  choices,
  disabled,
  onChange,
  onReset,
}: TableProps): React.ReactElement {
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
        const resolvedValue = currentValue !== undefined ? currentValue : baseValue;
        const changed = currentValue !== undefined
          ? baseValue === undefined || currentValue !== baseValue
          : false;
        return {
          key,
          baseValue,
          currentValue,
          resolvedValue,
          changed,
        };
      })
      .filter((entry) => {
        if (!defaults) {
          return false;
        }
        if (showChangedOnly && !entry.changed) {
          return false;
        }
        if (!filter) {
          return true;
        }
        const haystack = `${entry.key} ${formatValue(entry.resolvedValue ?? entry.baseValue ?? "n/a")}`.toLowerCase();
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
        {entries.map(({ key, baseValue, resolvedValue, changed }) => {
          if (baseValue === undefined && resolvedValue === undefined) {
            return null;
          }
          const valueForType = resolvedValue ?? baseValue;
          const valueType = typeof valueForType;
          const choiceList = choices?.[key];
          const isBoolean = valueType === "boolean";
          const isNumber = valueType === "number" && !choiceList;
          const isString = valueType === "string" || choiceList;

          const handleNumberChange = (event: React.ChangeEvent<HTMLInputElement>) => {
            if (event.target.value === "") {
              onReset(key);
              return;
            }
            const next = event.target.valueAsNumber;
            if (Number.isNaN(next)) {
              return;
            }
            onChange(key, next);
          };

          const handleTextChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
            onChange(key, event.target.value);
          };

          const handleBooleanChange = (event: React.ChangeEvent<HTMLInputElement>) => {
            onChange(key, event.target.checked);
          };

          const metaClass = changed ? "settings-card__meta" : "settings-card__meta settings-card__meta--muted";

          return (
            <div key={key} className={changed ? "settings-card settings-card--changed" : "settings-card"}>
              <div className="settings-card__top">
                <div className="settings-card__key">{key}</div>
                {changed ? <span className="settings-card__badge">Changed</span> : null}
              </div>
              <div className="settings-card__value">
                {isBoolean ? (
                  <label className="settings-card__checkbox">
                    <input
                      type="checkbox"
                      checked={Boolean(valueForType)}
                      onChange={handleBooleanChange}
                      disabled={disabled}
                    />
                    <span>{valueForType ? "Enabled" : "Disabled"}</span>
                  </label>
                ) : null}
                {isNumber ? (
                  <input
                    type="number"
                    className="settings-card__input"
                    value={valueForType === undefined ? "" : Number(valueForType)}
                    onChange={handleNumberChange}
                    step={Number.isInteger(valueForType) ? 1 : "any"}
                    disabled={disabled}
                  />
                ) : null}
                {isString && !isBoolean && !isNumber ? (
                  choiceList ? (
                    (() => {
                      const current = String(valueForType ?? "");
                      const uniqueOptions = Array.from(new Set([current, ...choiceList]));
                      return (
                        <select
                          className="settings-card__select"
                          value={current}
                          onChange={handleTextChange}
                          disabled={disabled}
                        >
                          {uniqueOptions.map((option) => (
                            <option key={option} value={option}>{option}</option>
                          ))}
                        </select>
                      );
                    })()
                  ) : (
                    <input
                      type="text"
                      className="settings-card__input"
                      value={valueForType === undefined ? "" : String(valueForType)}
                      onChange={handleTextChange}
                      disabled={disabled}
                    />
                  )
                ) : null}
              </div>
              <div className="settings-card__footer">
                <button
                  type="button"
                  className="settings-card__reset"
                  onClick={() => onReset(key)}
                  disabled={disabled || baseValue === undefined || !changed}
                  aria-label={`Reset ${key} to default`}
                >
                  Reset
                </button>
                {baseValue !== undefined ? <div className={metaClass}>Default: {formatValue(baseValue)}</div> : null}
              </div>
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

  const updateParam = useCallback((scope: "evolution" | "backtest", key: string, value: Scalar) => {
    if (!defaults) {
      return;
    }
    setActivePreset((prev) => (prev ? null : prev));
    setActiveValues((prev) => {
      const baseEvolution = prev?.evolution ?? defaults.evolution;
      const baseBacktest = prev?.backtest ?? defaults.backtest;
      const nextEvolution = { ...baseEvolution };
      const nextBacktest = { ...baseBacktest };

      if (scope === "evolution") {
        nextEvolution[key] = value;
      } else {
        nextBacktest[key] = value;
      }

      const matchesDefaults =
        Object.keys(nextEvolution).length === Object.keys(defaults.evolution).length
        && Object.entries(defaults.evolution).every(([k, v]) => nextEvolution[k] === v)
        && Object.keys(nextBacktest).length === Object.keys(defaults.backtest).length
        && Object.entries(defaults.backtest).every(([k, v]) => nextBacktest[k] === v);

      if (matchesDefaults) {
        return null;
      }

      return {
        evolution: nextEvolution,
        backtest: nextBacktest,
      };
    });
  }, [defaults]);

  const resetParam = useCallback((scope: "evolution" | "backtest", key: string) => {
    if (!defaults) {
      return;
    }
    const baseline = scope === "evolution" ? defaults.evolution : defaults.backtest;
    if (!(key in baseline)) {
      return;
    }
    updateParam(scope, key, baseline[key]);
  }, [defaults, updateParam]);

  const activeEvolution = activeValues?.evolution ?? defaults?.evolution ?? null;
  const activeBacktest = activeValues?.backtest ?? defaults?.backtest ?? null;
  const choiceMap = defaults?.choices ?? {};
  const editingDisabled = loading || presetLoading || saving;

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
            choices={choiceMap}
            disabled={editingDisabled}
            onChange={(key, value) => updateParam("evolution", key, value)}
            onReset={(key) => resetParam("evolution", key)}
          />
          <ConfigTable
            title="Backtest"
            defaults={defaults?.backtest ?? null}
            active={activeBacktest}
            filterText={searchText}
            showChangedOnly={showChangedOnly}
            choices={choiceMap}
            disabled={editingDisabled}
            onChange={(key, value) => updateParam("backtest", key, value)}
            onReset={(key) => resetParam("backtest", key)}
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
