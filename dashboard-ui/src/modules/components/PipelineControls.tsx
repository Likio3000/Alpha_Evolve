import React, { useEffect, useId, useState } from "react";
import { fetchConfigList, fetchConfigPresets } from "../api";
import { ConfigListItem, PipelineRunRequest } from "../types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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
  const datasetLabels: Record<string, string> = {
    sp500: "S&P 500 (daily)",
    sp500_small: "S&P 500 (subset)",
  };
  const [dataset, setDataset] = useState(defaultDataset);
  const [datasetOptions, setDatasetOptions] = useState<string[]>(defaultDataset ? [defaultDataset] : []);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [generationInput, setGenerationInput] = useState("5");
  const [popSizeInput, setPopSizeInput] = useState("100");
  const [runnerMode, setRunnerMode] = useState<PipelineRunRequest["runner_mode"]>("auto");
  const [configPath, setConfigPath] = useState("");
  const [configs, setConfigs] = useState<ConfigListItem[]>([]);
  const [configLoading, setConfigLoading] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [runBaselines, setRunBaselines] = useState(false);

  useEffect(() => {
    setDataset(defaultDataset);
  }, [defaultDataset]);

  useEffect(() => {
    let cancelled = false;
    const loadDatasets = async () => {
      setDatasetLoading(true);
      setDatasetError(null);
      try {
        const presets = await fetchConfigPresets();
        if (cancelled) return;
        const keys = Object.keys(presets).sort((a, b) => a.localeCompare(b));
        setDatasetOptions(keys);
        setDataset((prev) => {
          if (prev && keys.includes(prev)) {
            return prev;
          }
          if (defaultDataset && keys.includes(defaultDataset)) {
            return defaultDataset;
          }
          return keys[0] ?? "";
        });
      } catch (error) {
        if (!cancelled) {
          const detail = error instanceof Error ? error.message : String(error);
          setDatasetError(detail);
        }
      } finally {
        if (!cancelled) {
          setDatasetLoading(false);
        }
      }
    };
    void loadDatasets();
    return () => {
      cancelled = true;
    };
  }, [defaultDataset]);

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
      dataset: configPath ? undefined : dataset || undefined,
      config: configPath || undefined,
      overrides,
      runner_mode: runnerMode || undefined,
      run_baselines: runBaselines || undefined,
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
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-xl font-bold">Pipeline Controls</CardTitle>
        <Button
          type="submit"
          form={formId}
          disabled={busy || hasInvalidInput || hasZeroValue}
          className="w-auto"
        >
          {busy ? "Running…" : "Launch Pipeline"}
        </Button>
      </CardHeader>
      <CardContent>
        <form id={formId} className="grid gap-4 py-4" onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="grid gap-2">
              <Label htmlFor="dataset">Dataset preset</Label>
              <Select
                value={dataset}
                onValueChange={(value) => setDataset(value)}
                disabled={busy || datasetLoading}
              >
                <SelectTrigger id="dataset">
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
                      {datasetLabels[value] ?? value}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {datasetLoading && <p className="text-xs text-muted-foreground">Loading dataset presets…</p>}
              {datasetError && <p className="text-xs text-destructive">{datasetError}</p>}
            </div>

            <div className="grid gap-2">
              <Label htmlFor="config">Config preset</Label>
              <Select
                value={configPath}
                onValueChange={(value) => setConfigPath(value === "none" ? "" : value)}
                disabled={busy || configLoading}
              >
                <SelectTrigger id="config">
                  <SelectValue placeholder="Select config" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None (use dataset defaults)</SelectItem>
                  {configs.map((cfg) => (
                    <SelectItem key={cfg.path} value={cfg.path}>
                      {cfg.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {configLoading && <p className="text-xs text-muted-foreground">Loading configs…</p>}
              {configError && <p className="text-xs text-destructive">{configError}</p>}
              {configPath && <p className="text-xs text-muted-foreground">Overrides dataset defaults</p>}
            </div>

            <div className="grid gap-2">
              <Label htmlFor="generations">Generations</Label>
              <Input
                id="generations"
                type="number"
                step={1}
                value={generationInput}
                onChange={(event) => setGenerationInput(event.target.value)}
                disabled={busy}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="popSize">Population size</Label>
              <Input
                id="popSize"
                type="number"
                step={1}
                value={popSizeInput}
                onChange={(event) => setPopSizeInput(event.target.value)}
                disabled={busy}
              />
            </div>

            <div className="md:col-span-2 rounded-xl border border-white/10 bg-white/[0.04] p-4">
              <div className="flex items-center gap-2">
                <input
                  id="runBaselines"
                  type="checkbox"
                  className="h-4 w-4 rounded border-white/20 text-primary focus:ring-primary"
                  checked={runBaselines}
                  onChange={(event) => setRunBaselines(event.target.checked)}
                  disabled={busy}
                />
                <Label htmlFor="runBaselines" className="text-sm">Run ML baseline</Label>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Trains or reuses the ML baseline so you can compare Sharpe in the dashboard.
              </p>
            </div>

            <div className="grid gap-2 md:col-span-2">
              <Label htmlFor="runnerMode">Runner mode</Label>
              <Select
                value={runnerMode ?? "auto"}
                onValueChange={(value) => setRunnerMode(value as PipelineRunRequest["runner_mode"])}
                disabled={busy}
              >
                <SelectTrigger id="runnerMode">
                  <SelectValue placeholder="Select runner mode" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto (recommended)</SelectItem>
                  <SelectItem value="subprocess">Subprocess (sandbox-safe)</SelectItem>
                  <SelectItem value="multiprocessing">Multiprocessing (fastest)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {inlineNotice && <p className="text-sm font-medium text-destructive">{inlineNotice}</p>}
          {message && <div className="p-3 bg-muted rounded text-sm">{message}</div>}
        </form>
      </CardContent>
    </Card>
  );
}
