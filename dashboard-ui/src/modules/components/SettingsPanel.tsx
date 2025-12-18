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
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold tracking-tight">{title}</h3>
        <span className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded-full">{entries.length} {entries.length === 1 ? "field" : "fields"}</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
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

          return (
            <Card key={key} className={cn("flex flex-col", changed && "border-primary/50 bg-primary/5")}>
              <CardHeader className="p-4 pb-2">
                <div className="flex justify-between items-start gap-2">
                  <CardTitle className="text-xs font-mono font-medium truncate break-all" title={key}>{key}</CardTitle>
                  {changed ? <Badge variant="secondary" className="text-[10px] h-4 px-1">Changed</Badge> : null}
                </div>
              </CardHeader>

              <CardContent className="p-4 py-2 flex-grow">
                {isBoolean ? (
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-600"
                      checked={Boolean(valueForType)}
                      onChange={handleBooleanChange}
                      disabled={disabled}
                    />
                    <span className="text-sm">{valueForType ? "Enabled" : "Disabled"}</span>
                  </div>
                ) : null}
                {isNumber ? (
                  <Input
                    type="number"
                    className="h-8 text-sm"
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
                        <Select
                          value={current}
                          onValueChange={(value) => {
                            onChange(key, value);
                          }}
                          disabled={disabled}
                        >
                          <SelectTrigger className="h-8 w-full text-xs">
                            <SelectValue placeholder="Select value" />
                          </SelectTrigger>
                          <SelectContent>
                            {uniqueOptions.map((option) => (
                              <SelectItem key={option} value={option}>
                                {option}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      );
                    })()
                  ) : (
                    <Input
                      type="text"
                      className="h-8 text-sm"
                      value={valueForType === undefined ? "" : String(valueForType)}
                      onChange={handleTextChange}
                      disabled={disabled}
                    />
                  )
                ) : null}
              </CardContent>
              <CardFooter className="p-4 pt-2 flex justify-between items-center bg-muted/20">
                {baseValue !== undefined && changed ? (
                  <div className="text-[10px] text-muted-foreground truncate" title={`Default: ${formatValue(baseValue)}`}>
                    Def: {formatValue(baseValue)}
                  </div>
                ) : (
                  <div />
                )}

                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-[10px]"
                  onClick={() => onReset(key)}
                  disabled={disabled || baseValue === undefined || !changed}
                >
                  Reset
                </Button>
              </CardFooter>
            </Card>
          );
        })}
      </div>
      {!entries.length ? <p className="text-sm text-muted-foreground italic">No parameters found.</p> : null}
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
    <Card className="w-full h-full border-none shadow-none bg-transparent">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Settings &amp; Presets</h2>
          <p className="text-sm text-muted-foreground">Manage evolution configurations and presets.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => void loadInitialData()} disabled={loading}>
            Refresh
          </Button>
          <Button variant="outline" size="sm" onClick={resetActive} disabled={presetLoading}>
            Use defaults
          </Button>
          <Button variant="outline" size="sm" onClick={handleCopy} disabled={!defaults}>
            Copy JSON
          </Button>
        </div>
      </div>

      {error ? <div className="text-sm text-destructive bg-destructive/10 p-2 rounded mb-4">{error}</div> : null}
      {loading ? <div className="text-sm text-muted-foreground">Loading configuration metadata…</div> : null}

      {!loading && (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          <aside className="md:col-span-3 space-y-6">
            <div>
              <h3 className="text-sm font-semibold mb-3">Dataset Presets</h3>
              <div className="space-y-1">
                {Object.keys(presets).length === 0 ? (
                  <p className="text-xs text-muted-foreground">No presets available.</p>
                ) : (
                  Object.keys(presets)
                    .sort()
                    .map((key) => (
                      <Button
                        key={key}
                        variant={key === activePreset ? "secondary" : "ghost"}
                        className="w-full justify-start h-8 text-sm"
                        onClick={() => handleLoadDataset(key)}
                        disabled={presetLoading}
                      >
                        {key}
                      </Button>
                    ))
                )}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold mb-3">Saved Configs</h3>
              <div className="space-y-1">
                {configs.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No stored configs yet.</p>
                ) : (
                  configs.map((item) => (
                    <Button
                      key={item.path}
                      variant={item.name === activePreset ? "secondary" : "ghost"}
                      className="w-full justify-start h-8 text-sm truncate"
                      onClick={() => handleLoadConfig(item)}
                      disabled={presetLoading}
                      title={item.name}
                    >
                      <span className="truncate">{item.name}</span>
                    </Button>
                  ))
                )}
              </div>
            </div>
          </aside>

          <div className="md:col-span-9 space-y-8">
            <Card className="p-4 bg-muted/40 sticky top-0 z-10 backdrop-blur-sm">
              <div className="flex flex-col sm:flex-row gap-4 items-end">
                <div className="w-full space-y-1">
                  <Label htmlFor="search-settings">Filter Parameters</Label>
                  <Input
                    id="search-settings"
                    placeholder="Search settings…"
                    value={searchText}
                    onChange={(event) => setSearchText(event.target.value)}
                  />
                </div>
                <div className="flex items-center space-x-2 pb-2">
                  <input
                    id="show-changed"
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-600"
                    checked={showChangedOnly}
                    onChange={(event) => setShowChangedOnly(event.target.checked)}
                  />
                  <Label htmlFor="show-changed" className="font-normal cursor-pointer">Show changed only</Label>
                </div>
              </div>
            </Card>

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

            <Card className="p-6 bg-muted/20 mt-8">
              <div className="flex flex-col md:flex-row gap-4 items-end">
                <div className="w-full space-y-2">
                  <Label>Save Configuration As</Label>
                  <Input
                    type="text"
                    value={saveName}
                    onChange={(event) => setSaveName(event.target.value)}
                    placeholder="configs/custom.toml"
                    disabled={saving}
                  />
                </div>
                <Button onClick={handleSave} disabled={saving || loading}>
                  {saving ? "Saving…" : "Save configuration"}
                </Button>
              </div>
            </Card>
          </div>
        </div>
      )}
    </Card>
  );
}
