import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { fetchConfigList, fetchConfigPresets } from "../api";
import {
  ConfigListItem,
  MLModelSpec,
  MLRunDetails,
  MLRunRequest,
} from "../types";
import {
  useJobActivity,
  useMlModels,
  useMlRunDetails,
  useMlRuns,
  useStartMlRun,
  useStopMlRun,
  useRuns,
} from "@/hooks/use-dashboard";

type ModelFormState = {
  selected: boolean;
  preset: string;
  paramRows: ParamRow[];
  showAdvanced: boolean;
  advancedParams: string;
};

type MLLabPageProps = {
  onNotify?: (message: string) => void;
};

type ParamRow = {
  key: string;
  value: string;
};

const NUMERIC_PATTERN = /^-?\d+(?:\.\d+)?(?:e[-+]?\d+)?$/i;

function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const normalized = value <= 1 ? value * 100 : value;
  return `${normalized.toFixed(1)}%`;
}

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function parseNumeric(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseInteger(value: string): number | null {
  const parsed = parseNumeric(value);
  if (parsed === null || !Number.isInteger(parsed)) {
    return null;
  }
  return parsed;
}

function parseJsonParams(value: string): Record<string, unknown> | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = JSON.parse(trimmed);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Params must be a JSON object.");
  }
  return parsed as Record<string, unknown>;
}

function stringifyParamValue(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch (error) {
    return String(value);
  }
}

function buildParamRows(params?: Record<string, unknown> | null): ParamRow[] {
  if (!params) return [];
  return Object.keys(params)
    .sort((a, b) => a.localeCompare(b))
    .map((key) => ({
      key,
      value: stringifyParamValue(params[key]),
    }));
}

function parseParamValue(value: string): unknown {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const lower = trimmed.toLowerCase();
  if (lower === "true") return true;
  if (lower === "false") return false;
  if (lower === "null") return null;
  if (lower === "undefined") return undefined;
  if (NUMERIC_PATTERN.test(trimmed)) {
    return Number(trimmed);
  }
  const first = trimmed[0];
  if (first === "{" || first === "[" || first === "\"") {
    return JSON.parse(trimmed);
  }
  return trimmed;
}

function buildParamsFromRows(rows: ParamRow[]): Record<string, unknown> | null {
  const params: Record<string, unknown> = {};
  for (const row of rows) {
    const key = row.key.trim();
    if (!key) continue;
    const value = row.value.trim();
    if (!value) continue;
    const parsed = parseParamValue(value);
    if (parsed === undefined) continue;
    params[key] = parsed;
  }
  return Object.keys(params).length ? params : null;
}

function findPreset(model: MLModelSpec, presetName: string): MLModelSpec["presets"][number] | undefined {
  return model.presets.find((preset) => preset.name === presetName);
}

function progressPercent(progress: Record<string, unknown> | null | undefined): number | null {
  if (!progress) return null;
  const pctRaw = progress.pct_complete ?? progress.pctComplete;
  if (typeof pctRaw === "number" && Number.isFinite(pctRaw)) {
    return pctRaw <= 1 ? pctRaw * 100 : pctRaw;
  }
  const idx = progress.index ?? progress.current_index ?? progress.currentIndex;
  const total = progress.total ?? progress.total_models ?? progress.totalModels;
  if (typeof idx === "number" && typeof total === "number" && total > 0) {
    return (idx / total) * 100;
  }
  return null;
}

export function MLLabPage({ onNotify }: MLLabPageProps): React.ReactElement {
  const { data: models = [], isLoading: modelsLoading, error: modelsError } = useMlModels();
  const { data: mlRuns = [], isLoading: mlRunsLoading, error: mlRunsError } = useMlRuns();
  const { data: pipelineRuns = [] } = useRuns();

  const [selectedRunPath, setSelectedRunPath] = useState<string | null>(null);
  const { data: runDetails, isLoading: runDetailsLoading, error: runDetailsError } = useMlRunDetails(selectedRunPath);

  const startRun = useStartMlRun();
  const stopRun = useStopMlRun();

  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const { data: jobActivity } = useJobActivity(activeJobId, Boolean(activeJobId));

  const [dataset, setDataset] = useState("sp500");
  const [datasetOptions, setDatasetOptions] = useState<string[]>(["sp500"]);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState<string | null>(null);

  const [configPath, setConfigPath] = useState("");
  const [configOptions, setConfigOptions] = useState<ConfigListItem[]>([]);
  const [configLoading, setConfigLoading] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);

  const [trainFraction, setTrainFraction] = useState("0.7");
  const [seed, setSeed] = useState("42");
  const [formMessage, setFormMessage] = useState<string | null>(null);

  const [modelState, setModelState] = useState<Record<string, ModelFormState>>({});

  useEffect(() => {
    let cancelled = false;
    const loadPresets = async () => {
      setDatasetLoading(true);
      setDatasetError(null);
      try {
        const presets = await fetchConfigPresets();
        if (cancelled) return;
        const keys = Object.keys(presets).sort((a, b) => a.localeCompare(b));
        setDatasetOptions(keys);
        setDataset((prev) => (prev && keys.includes(prev) ? prev : keys[0] ?? ""));
      } catch (error) {
        if (!cancelled) {
          setDatasetError(formatError(error));
        }
      } finally {
        if (!cancelled) {
          setDatasetLoading(false);
        }
      }
    };
    void loadPresets();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadConfigs = async () => {
      setConfigLoading(true);
      setConfigError(null);
      try {
        const items = await fetchConfigList();
        if (!cancelled) {
          setConfigOptions([...items].sort((a, b) => a.name.localeCompare(b.name)));
        }
      } catch (error) {
        if (!cancelled) {
          setConfigError(formatError(error));
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

  useEffect(() => {
    if (!models.length) return;
    setModelState((prev) => {
      if (Object.keys(prev).length > 0) return prev;
      const next: Record<string, ModelFormState> = {};
      for (const model of models) {
        const defaultPreset = model.defaultPreset ?? model.presets[0]?.name ?? "";
        const preset = findPreset(model, defaultPreset);
        next[model.id] = {
          selected: true,
          preset: defaultPreset,
          paramRows: buildParamRows(preset?.params),
          showAdvanced: false,
          advancedParams: "",
        };
      }
      return next;
    });
  }, [models]);

  useEffect(() => {
    if (!selectedRunPath && mlRuns.length > 0) {
      setSelectedRunPath(mlRuns[0].path);
    }
  }, [mlRuns, selectedRunPath]);

  const modelError = modelsError ? formatError(modelsError) : null;
  const runError = mlRunsError ? formatError(mlRunsError) : null;
  const detailsError = runDetailsError ? formatError(runDetailsError) : null;

  const activeProgress = jobActivity?.progress as Record<string, unknown> | undefined;
  const activePercent = progressPercent(activeProgress);
  const activeLabel =
    (activeProgress?.model_label as string | undefined) ||
    (activeProgress?.model_id as string | undefined) ||
    null;
  const activeVariant = (activeProgress?.variant as string | undefined) || null;
  const activeStatus = jobActivity?.status ?? (jobActivity?.running ? "running" : null);
  const bestSharpe = jobActivity?.sharpe_best ?? null;

  const bestAlphaSharpe = useMemo(() => {
    if (!pipelineRuns.length) return null;
    return pipelineRuns.reduce((best, run) => {
      if (run.sharpeBest === null || run.sharpeBest === undefined) return best;
      return best === null ? run.sharpeBest : Math.max(best, run.sharpeBest);
    }, null as number | null);
  }, [pipelineRuns]);

  const bestMlSharpe = useMemo(() => {
    if (!mlRuns.length) return null;
    return mlRuns.reduce((best, run) => {
      if (run.bestSharpe === null || run.bestSharpe === undefined) return best;
      return best === null ? run.bestSharpe : Math.max(best, run.bestSharpe);
    }, null as number | null);
  }, [mlRuns]);

  const selectedModels = useMemo(() => {
    return models.filter((model) => modelState[model.id]?.selected);
  }, [models, modelState]);

  const modelSelections = useMemo(() => {
    const selections: Array<{ model: MLModelSpec; state: ModelFormState }> = [];
    for (const model of models) {
      const state = modelState[model.id];
      if (!state || !state.selected) continue;
      selections.push({ model, state });
    }
    return selections;
  }, [models, modelState]);

  const selectedRun: MLRunDetails | null = runDetails ?? null;
  const results = Array.isArray(selectedRun?.results) ? selectedRun?.results : [];
  const sortedResults = useMemo(() => {
    return [...results].sort((a, b) => {
      const sa = a.Sharpe ?? -Infinity;
      const sb = b.Sharpe ?? -Infinity;
      return sb - sa;
    });
  }, [results]);

  const selectedStatus = useMemo(() => {
    const status = selectedRun?.meta?.status;
    return typeof status === "string" && status ? status : "unknown";
  }, [selectedRun]);

  const handleToggleAll = useCallback(
    (value: boolean) => {
      setModelState((prev) => {
        const next = { ...prev };
        for (const model of models) {
          const preset = model.defaultPreset ?? model.presets[0]?.name ?? "";
          const current = next[model.id] ?? {
            selected: false,
            preset,
            paramRows: buildParamRows(findPreset(model, preset)?.params),
            showAdvanced: false,
            advancedParams: "",
          };
          next[model.id] = { ...current, selected: value };
        }
        return next;
      });
    },
    [models]
  );

  const handleRun = useCallback(async () => {
    if (selectedModels.length === 0) {
      setFormMessage("Select at least one model to run.");
      return;
    }
    const parsedSeed = parseInteger(seed);
    if (seed && parsedSeed === null) {
      setFormMessage("Seed must be an integer.");
      return;
    }
    const parsedTrainFraction = parseNumeric(trainFraction);
    if (trainFraction && parsedTrainFraction === null) {
      setFormMessage("Train fraction must be numeric.");
      return;
    }
    const modelsPayload: MLRunRequest["models"] = [];
    try {
      for (const { model, state } of modelSelections) {
        const params =
          state.advancedParams.trim().length > 0
            ? parseJsonParams(state.advancedParams)
            : buildParamsFromRows(state.paramRows);
        modelsPayload.push({
          id: model.id,
          preset: state.preset || undefined,
          params: params ?? undefined,
        });
      }
    } catch (error) {
      setFormMessage(formatError(error));
      return;
    }

    const payload: MLRunRequest = {
      dataset: configPath ? undefined : dataset || undefined,
      config: configPath || undefined,
      models: modelsPayload,
      train_fraction: parsedTrainFraction ?? undefined,
      seed: parsedSeed ?? undefined,
    };

    try {
      setFormMessage(null);
      const response = await startRun.mutateAsync(payload);
      setActiveJobId(response.job_id);
      setSelectedRunPath(response.run_dir);
      onNotify?.("ML run launched.");
    } catch (error) {
      setFormMessage(formatError(error));
    }
  }, [
    configPath,
    dataset,
    modelSelections,
    onNotify,
    seed,
    selectedModels.length,
    startRun,
    trainFraction,
  ]);

  const handleStop = useCallback(async () => {
    if (!activeJobId) return;
    try {
      await stopRun.mutateAsync(activeJobId);
      onNotify?.("Stop signal sent.");
    } catch (error) {
      onNotify?.(formatError(error));
    }
  }, [activeJobId, onNotify, stopRun]);

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-2 duration-700">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-heading font-bold tracking-tight">ML Lab</h2>
          <p className="text-sm text-muted-foreground">Train ML contenders side-by-side.</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <Card className="glass-panel border-white/10 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-transparent pointer-events-none" />
          <CardHeader className="relative space-y-1">
            <CardTitle className="text-lg font-semibold">Run Setup</CardTitle>
            <p className="text-xs text-muted-foreground">Pick a dataset and training split.</p>
          </CardHeader>
          <CardContent className="relative space-y-5">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="grid gap-2">
                <Label>Dataset preset</Label>
                <Select
                  value={dataset}
                  onValueChange={(value) => setDataset(value)}
                  disabled={datasetLoading}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select dataset" />
                  </SelectTrigger>
                  <SelectContent>
                    {datasetOptions.length === 0 ? (
                      <SelectItem value="none" disabled>
                        No presets found
                      </SelectItem>
                    ) : null}
                    {datasetOptions.map((value) => (
                      <SelectItem key={value} value={value}>
                        {value}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {datasetError && <p className="text-xs text-destructive">{datasetError}</p>}
              </div>
              <div className="grid gap-2">
                <Label>Config override</Label>
                <Select
                  value={configPath}
                  onValueChange={(value) => setConfigPath(value === "none" ? "" : value)}
                  disabled={configLoading}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Use dataset defaults" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None (use dataset defaults)</SelectItem>
                    {configOptions.map((cfg) => (
                      <SelectItem key={cfg.path} value={cfg.path}>
                        {cfg.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {configError && <p className="text-xs text-destructive">{configError}</p>}
              </div>
              <div className="grid gap-2">
                <Label>Seed</Label>
                <Input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="42" />
              </div>
              <div className="grid gap-2">
                <Label>Train fraction</Label>
                <Input value={trainFraction} onChange={(e) => setTrainFraction(e.target.value)} placeholder="0.7" />
              </div>
            </div>
            <div className="text-[10px] text-muted-foreground">
              Train fraction sets the split; test size is derived automatically.
            </div>

            {formMessage && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
                {formMessage}
              </div>
            )}
            <div className="flex items-center gap-3">
              <Button onClick={handleRun} disabled={startRun.isPending}>
                {startRun.isPending ? "Launching..." : "Run ML Models"}
              </Button>
              <Button variant="ghost" onClick={handleStop} disabled={!activeJobId}>
                Stop
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-panel border-white/10">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg font-semibold">Live Status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 pt-0">
            <div className="grid gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold">Active Run</span>
                <Badge variant="outline" className="text-[10px] uppercase tracking-widest">
                  {activeStatus ?? "idle"}
                </Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                {activeLabel ? `${activeLabel}${activeVariant ? ` (${activeVariant})` : ""}` : "No active run"}
              </div>
              <Progress value={activePercent ?? 0} className="h-1.5" />
              <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                <span>{activePercent !== null ? formatPercent(activePercent / 100) : "-"}</span>
                <span>Best Sharpe {formatNumber(bestSharpe, 3)}</span>
              </div>
            </div>

            <div className="grid gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold">Best Sharpe Benchmarks</span>
              </div>
              <div className="grid gap-2 text-xs text-muted-foreground">
                <div className="flex items-center justify-between">
                  <span>ML Lab (all runs)</span>
                  <span className="font-mono">{formatNumber(bestMlSharpe, 3)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Alpha Evolve (all runs)</span>
                  <span className="font-mono">{formatNumber(bestAlphaSharpe, 3)}</span>
                </div>
              </div>
            </div>

            {(modelsLoading || mlRunsLoading || runDetailsLoading) && (
              <div className="text-xs text-muted-foreground">Syncing ML artifacts...</div>
            )}
            {runError && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
                {runError}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="glass-panel border-white/10">
        <CardHeader className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle className="text-lg font-semibold">Models</CardTitle>
            <p className="text-xs text-muted-foreground">Choose contenders and tune presets.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="text-[10px] uppercase tracking-widest">
              {selectedModels.length} selected
            </Badge>
            <Button variant="secondary" size="sm" onClick={() => handleToggleAll(true)} disabled={modelsLoading}>
              Select All
            </Button>
            <Button variant="ghost" size="sm" onClick={() => handleToggleAll(false)} disabled={modelsLoading}>
              Clear
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {modelError && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
              {modelError}
            </div>
          )}
          {modelsLoading && (
            <div className="text-xs text-muted-foreground">Loading models...</div>
          )}
          <div className="grid gap-3">
            {models.map((model) => {
              const state = modelState[model.id];
              if (!state) return null;
              const selectedPreset = findPreset(model, state.preset);
              const updateState = (updater: (current: ModelFormState) => ModelFormState) =>
                setModelState((prev) => {
                  const current = prev[model.id] ?? state;
                  return {
                    ...prev,
                    [model.id]: updater(current),
                  };
                });
              return (
                <div
                  key={model.id}
                  className={cn(
                    "rounded-xl border p-4 transition-all",
                    state.selected
                      ? "border-primary/30 bg-primary/5"
                      : "border-white/10 bg-white/[0.02]"
                  )}
                >
                  <div className="flex flex-col gap-3">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={state.selected}
                            onChange={(e) => updateState((current) => ({ ...current, selected: e.target.checked }))}
                          />
                          <span className="text-sm font-semibold">{model.label}</span>
                        </div>
                        {model.description && (
                          <p className="text-xs text-muted-foreground mt-1">{model.description}</p>
                        )}
                      </div>
                      <Badge variant="secondary" className="text-[10px] uppercase tracking-widest">
                        {model.id}
                      </Badge>
                    </div>
                    <div className="grid gap-4 lg:grid-cols-[220px_1fr]">
                      <div className="grid gap-2">
                        <Label className="text-xs uppercase tracking-wider text-muted-foreground">Preset</Label>
                        <Select
                          value={state.preset}
                          onValueChange={(value) => {
                            const preset = findPreset(model, value);
                            updateState((current) => ({
                              ...current,
                              preset: value,
                              paramRows: buildParamRows(preset?.params),
                              advancedParams: "",
                            }));
                          }}
                          disabled={!state.selected}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Preset" />
                          </SelectTrigger>
                          <SelectContent>
                            {model.presets.map((preset) => (
                              <SelectItem key={preset.name} value={preset.name}>
                                {preset.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        {selectedPreset?.description && (
                          <p className="text-[10px] text-muted-foreground">
                            {selectedPreset.description}
                          </p>
                        )}
                        <div className="flex flex-wrap items-center gap-2 pt-1">
                          <Button
                            type="button"
                            variant="secondary"
                            size="sm"
                            disabled={!state.selected}
                            onClick={() => {
                              updateState((current) => ({
                                ...current,
                                paramRows: buildParamRows(selectedPreset?.params),
                                advancedParams: "",
                              }));
                            }}
                          >
                            Reset to preset
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            disabled={!state.selected}
                            onClick={() =>
                              updateState((current) => ({
                                ...current,
                                showAdvanced: !current.showAdvanced,
                              }))
                            }
                          >
                            {state.showAdvanced ? "Hide JSON" : "Advanced JSON"}
                          </Button>
                        </div>
                      </div>
                      <div className="grid gap-2">
                        <Label className="text-xs uppercase tracking-wider text-muted-foreground">Parameters</Label>
                        {state.paramRows.length === 0 ? (
                          <div className="text-[10px] text-muted-foreground">
                            No tunable params for this preset. Use Advanced JSON for overrides.
                          </div>
                        ) : (
                          <div className="grid gap-3 sm:grid-cols-2">
                            {state.paramRows.map((row, index) => (
                              <div key={`${model.id}-param-${index}`} className="grid gap-1">
                                <Label className="text-[10px] uppercase tracking-wider text-muted-foreground">
                                  {row.key}
                                </Label>
                                <Input
                                  value={row.value}
                                  onChange={(e) => {
                                    const value = e.target.value;
                                    updateState((current) => {
                                      const rows = [...current.paramRows];
                                      rows[index] = { ...rows[index], value };
                                      return { ...current, paramRows: rows };
                                    });
                                  }}
                                  placeholder="Value"
                                  disabled={!state.selected}
                                />
                              </div>
                            ))}
                          </div>
                        )}
                        <div className="text-[10px] text-muted-foreground">
                          Values accept numbers, true/false, JSON arrays/objects, or strings.
                        </div>
                        {state.showAdvanced && (
                          <div className="grid gap-2">
                            <Label className="text-xs uppercase tracking-wider text-muted-foreground">
                              Advanced JSON Override
                            </Label>
                            <textarea
                              value={state.advancedParams}
                              onChange={(e) =>
                                updateState((current) => ({
                                  ...current,
                                  advancedParams: e.target.value,
                                }))
                              }
                              placeholder='{"max_depth": 10, "learning_rate": 0.05}'
                              disabled={!state.selected}
                              className={cn(
                                "min-h-[88px] w-full rounded-md border border-input bg-background px-3 py-2 text-xs font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                              )}
                            />
                            <div className="text-[10px] text-muted-foreground">
                              When JSON is set, it overrides the row-based params above.
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <div className="grid lg:grid-cols-[320px_1fr] gap-6">
        <Card className="glass-panel border-white/10">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">ML Runs</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {mlRunsLoading && <div className="text-xs text-muted-foreground">Loading runs...</div>}
            {mlRuns.length === 0 && !mlRunsLoading && (
              <div className="text-xs text-muted-foreground">No ML runs yet.</div>
            )}
            <div className="flex flex-col gap-2">
              {mlRuns.map((run) => {
                const isSelected = run.path === selectedRunPath;
                return (
                  <button
                    key={run.path}
                    className={cn(
                      "text-left rounded-lg border p-3 transition-all",
                      isSelected
                        ? "border-primary/40 bg-primary/10"
                        : "border-white/10 bg-white/[0.02] hover:border-white/20"
                    )}
                    onClick={() => setSelectedRunPath(run.path)}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold">{run.name}</span>
                      <Badge variant="secondary" className="text-[9px] uppercase tracking-widest">
                        {run.status ?? "unknown"}
                      </Badge>
                    </div>
                    <div className="mt-2 flex items-center justify-between text-[10px] text-muted-foreground">
                      <span>Sharpe {formatNumber(run.bestSharpe, 3)}</span>
                      <span>
                        {run.completed ?? 0}/{run.total ?? "?"}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <Card className="glass-panel border-white/10">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Run Results</CardTitle>
          </CardHeader>
          <CardContent>
            {detailsError && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
                {detailsError}
              </div>
            )}
            {!selectedRun && !runDetailsLoading && (
              <div className="text-xs text-muted-foreground">Select a run to view results.</div>
            )}
            {selectedRun && (
              <div className="space-y-4">
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>{selectedRun.name}</span>
                  <span>{selectedStatus}</span>
                </div>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Model</TableHead>
                      <TableHead>Sharpe</TableHead>
                      <TableHead>Ann Ret</TableHead>
                      <TableHead>Max DD</TableHead>
                      <TableHead>Turnover</TableHead>
                      <TableHead>IC</TableHead>
                      <TableHead>Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sortedResults.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={7} className="text-xs text-muted-foreground">
                          No results yet.
                        </TableCell>
                      </TableRow>
                    )}
                    {sortedResults.map((row, index) => (
                      <TableRow key={`${row.modelId}-${row.variant}-${index}`}>
                        <TableCell className="text-xs">
                          <div className="font-semibold">{row.modelLabel ?? row.modelId}</div>
                          <div className="text-[10px] text-muted-foreground">{row.variant}</div>
                        </TableCell>
                        <TableCell className="text-xs font-mono">{formatNumber(row.Sharpe, 3)}</TableCell>
                        <TableCell className="text-xs font-mono">{formatNumber(row.AnnReturn, 3)}</TableCell>
                        <TableCell className="text-xs font-mono">{formatNumber(row.MaxDD, 3)}</TableCell>
                        <TableCell className="text-xs font-mono">{formatNumber(row.Turnover, 3)}</TableCell>
                        <TableCell className="text-xs font-mono">{formatNumber(row.IC, 3)}</TableCell>
                        <TableCell>
                          <Badge variant={row.status === "error" ? "destructive" : "secondary"} className="text-[9px] uppercase tracking-widest">
                            {row.status ?? "ok"}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
