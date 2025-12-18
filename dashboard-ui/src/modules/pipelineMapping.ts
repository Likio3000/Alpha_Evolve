import { GenerationProgressState, GenerationSummary } from "./types";

function toNumber(value: unknown, fallback = 0): number {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function toNullableNumber(value: unknown): number | null {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

function mapNumberRecord(source: unknown): Record<string, number> {
  const record = asRecord(source);
  if (!record) {
    return {};
  }
  const result: Record<string, number> = {};
  for (const [key, val] of Object.entries(record)) {
    const num = Number(val);
    if (Number.isFinite(num)) {
      result[key] = num;
    }
  }
  return result;
}

function mapNestedNumberRecord(source: unknown): Record<string, Record<string, number>> {
  const record = asRecord(source);
  if (!record) {
    return {};
  }
  const result: Record<string, Record<string, number>> = {};
  for (const [key, val] of Object.entries(record)) {
    result[key] = mapNumberRecord(val);
  }
  return result;
}

export function mapGenerationProgress(raw: unknown): GenerationProgressState | null {
  const record = asRecord(raw);
  if (!record) {
    return null;
  }
  const generation = toNumber(record.gen ?? record.generation, NaN);
  const completed = toNumber(record.completed, NaN);
  if (!Number.isFinite(generation) || !Number.isFinite(completed)) {
    return null;
  }
  const total = toNullableNumber(record.total ?? record.population ?? record.total_individuals);
  const pctComplete = toNullableNumber(record.pct_complete ?? record.pct);
  let derivedPct = pctComplete;
  if (derivedPct === null && total && total > 0) {
    derivedPct = Math.min(1, generation / total);
  }
  return {
    generation,
    generationsTotal: toNullableNumber(record.generations_total ?? record.total_gens),
    pctComplete: derivedPct,
    completed,
    totalIndividuals: total,
    bestFitness: toNullableNumber(record.best),
    medianFitness: toNullableNumber(record.median),
    elapsedSeconds: toNullableNumber(record.elapsed_sec ?? record.elapsed),
    etaSeconds: toNullableNumber(record.eta_sec ?? record.eta),
  };
}

export function mapGenerationSummary(raw: unknown): GenerationSummary | null {
  const record = asRecord(raw);
  if (!record) {
    return null;
  }
  const generation = toNumber(record.generation, NaN);
  const total = toNumber(record.generations_total, NaN);
  if (!Number.isFinite(generation) || !Number.isFinite(total)) {
    return null;
  }
  const pctCompleteRaw = Number(record.pct_complete);
  const pctComplete = Number.isFinite(pctCompleteRaw) ? pctCompleteRaw : generation / Math.max(1, total);

  const bestRecord = asRecord(record.best);
  if (!bestRecord) {
    return null;
  }

  const timingRecord = asRecord(record.timing);
  const populationRecord = asRecord(record.population);
  const penalties = mapNumberRecord(record.penalties);
  const fitnessBreakdownRaw = asRecord(record.fitness_breakdown);
  const fitnessBreakdown: Record<string, number | null> = {};
  if (fitnessBreakdownRaw) {
    for (const [key, val] of Object.entries(fitnessBreakdownRaw)) {
      const num = Number(val);
      fitnessBreakdown[key] = Number.isFinite(num) ? num : null;
    }
  }

  const best: GenerationSummary["best"] = {
    fitness: toNumber(bestRecord.fitness),
    fitnessStatic: toNullableNumber(bestRecord.fitness_static),
    meanIc: toNumber(bestRecord.mean_ic),
    icStd: toNumber(bestRecord.ic_std),
    turnover: toNumber(bestRecord.turnover),
    sharpeProxy: toNumber(bestRecord.sharpe_proxy),
    sortino: toNumber(bestRecord.sortino),
    drawdown: toNumber(bestRecord.drawdown),
    downsideDeviation: toNumber(bestRecord.downside_deviation),
    cvar: toNumber(bestRecord.cvar),
    factorPenalty: toNumber(bestRecord.factor_penalty),
    fingerprint: typeof bestRecord.fingerprint === "string" ? bestRecord.fingerprint : null,
    programSize: Math.trunc(toNumber(bestRecord.program_size)),
    program: typeof bestRecord.program === "string" ? bestRecord.program : "",
    horizonMetrics: mapNestedNumberRecord(bestRecord.horizon_metrics),
    factorExposures: mapNumberRecord(bestRecord.factor_exposures),
    regimeExposures: mapNumberRecord(bestRecord.regime_exposures),
    transactionCosts: mapNumberRecord(bestRecord.transaction_costs),
    stressMetrics: mapNumberRecord(bestRecord.stress_metrics),
  };

  const timing = {
    generationSeconds: toNumber(timingRecord?.generation_seconds),
    averageSeconds: toNullableNumber(timingRecord?.average_seconds),
    etaSeconds: toNullableNumber(timingRecord?.eta_seconds),
  };

  const population = {
    size: Math.trunc(toNumber(populationRecord?.size)),
    uniqueFingerprints: Math.trunc(toNumber(populationRecord?.unique_fingerprints)),
  };

  return {
    generation,
    generationsTotal: total,
    pctComplete,
    best,
    penalties,
    fitnessBreakdown,
    timing,
    population,
  };
}

