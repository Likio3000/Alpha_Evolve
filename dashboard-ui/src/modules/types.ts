export type Scalar = string | number | boolean;

export interface RunSummary {
  name: string;
  path: string;
  label?: string | null;
  sharpeBest?: number | null;
}

export interface LastRunPayload {
  run_dir: string | null;
  sharpe_best: number | null;
}

export interface BacktestRow {
  AlphaID: string | null;
  TS: string | null;
  TimeseriesFile: string | null;
  Sharpe: number | null;
  AnnReturn: number | null;
  AnnVol: number | null;
  MaxDD: number | null;
  Turnover: number | null;
  Ops: string | null;
  OriginalMetric: number | null;
  Program: string | null;
}

export interface AlphaTimeseries {
  date: string[];
  equity: number[];
  ret_net: number[];
}

export interface PipelineRunResponse {
  job_id: string;
}

export interface JobStatusResponse {
  exists: boolean;
  running: boolean;
}

export interface JobLogResponse {
  log: string;
}

export interface RunLabelPayload {
  path: string;
  label: string;
}

export interface PipelineRunRequest {
  generations: number;
  dataset?: string;
  config?: string;
  data_dir?: string;
  overrides?: Record<string, Scalar>;
}

export interface PipelineJobState {
  jobId: string;
  status: "idle" | "running" | "error" | "complete";
  lastMessage?: string;
  lastUpdated: number;
  log: string;
  sharpeBest?: number | null;
}

export interface RunUIContext {
  job_id?: string;
  submitted_at?: string;
  payload?: Record<string, unknown>;
  pipeline_args?: string[];
  [key: string]: unknown;
}

export interface RunDetails {
  name: string;
  path: string;
  label?: string | null;
  sharpeBest?: number | null;
  summary?: Record<string, unknown> | null;
  uiContext?: RunUIContext | null;
  meta?: Record<string, unknown> | null;
}

export interface ConfigDefaultsResponse {
  evolution: Record<string, Scalar>;
  backtest: Record<string, Scalar>;
  choices: Record<string, string[]>;
}

export interface ConfigListItem {
  name: string;
  path: string;
}

export interface ConfigListResponse {
  items: ConfigListItem[];
}

export interface ConfigPresetsResponse {
  presets: Record<string, string>;
}

export interface ConfigPresetValues {
  evolution: Record<string, Scalar>;
  backtest: Record<string, Scalar>;
}

export interface ConfigSavePayload {
  name: string;
  evolution: Record<string, Scalar>;
  backtest: Record<string, Scalar>;
}

export interface ParamMetaItem {
  key: string;
  label: string;
  type: string;
  default?: Scalar | null;
  min?: number;
  max?: number;
  step?: number;
  help?: string;
  choices?: string[];
}

export interface ParamMetaGroup {
  title: string;
  items: ParamMetaItem[];
}

export interface ParamMetaResponse {
  schema_version: number;
  groups: ParamMetaGroup[];
}
