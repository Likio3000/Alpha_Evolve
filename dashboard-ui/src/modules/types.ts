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
  pending?: boolean;
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
  runner_mode?: "auto" | "multiprocessing" | "subprocess";
}

export interface PipelineJobState {
  jobId: string;
  status: "idle" | "running" | "error" | "complete";
  lastMessage?: string;
  lastUpdated: number;
  log: string;
  logPath?: string | null;
  sharpeBest?: number | null;
  runDir?: string | null;
  runName?: string | null;
  progress?: GenerationProgressState | null;
  summaries: GenerationSummary[];
}

export interface JobActivityResponse {
  exists: boolean;
  running: boolean;
  status?: PipelineJobState["status"] | string | null;
  last_message?: string | null;
  log?: string | null;
  log_path?: string | null;
  sharpe_best?: number | null;
  run_dir?: string | null;
  progress?: unknown;
  summaries?: unknown[];
  updated_at?: number | string | null;
}

export interface GenerationProgressState {
  generation: number;
  generationsTotal?: number | null;
  pctComplete?: number | null;
  completed: number;
  totalIndividuals?: number | null;
  bestFitness?: number | null;
  medianFitness?: number | null;
  elapsedSeconds?: number | null;
  etaSeconds?: number | null;
}

export interface GenerationSummaryBest {
  fitness: number;
  fitnessStatic: number | null;
  meanIc: number;
  icStd: number;
  turnover: number;
  sharpeProxy: number;
  sortino: number;
  drawdown: number;
  downsideDeviation: number;
  cvar: number;
  factorPenalty: number;
  fingerprint?: string | null;
  programSize: number;
  program: string;
  horizonMetrics: Record<string, Record<string, number>>;
  factorExposures: Record<string, number>;
  regimeExposures: Record<string, number>;
  transactionCosts: Record<string, number>;
  stressMetrics: Record<string, number>;
}

export interface GenerationSummary {
  generation: number;
  generationsTotal: number;
  pctComplete: number;
  best: GenerationSummaryBest;
  penalties: Record<string, number>;
  fitnessBreakdown: Record<string, number | null>;
  timing: {
    generationSeconds: number;
    averageSeconds: number | null;
    etaSeconds: number | null;
  };
  population: {
    size: number;
    uniqueFingerprints: number;
  };
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
