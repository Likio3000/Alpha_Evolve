import React, { useEffect, useMemo, useState } from "react";
import { fetchRunAssets } from "../api";
import { GenerationSummary } from "../types";
import { mapGenerationSummary } from "../pipelineMapping";

interface RunForensicsPanelProps {
  runDir: string | null;
}

type PreviewKind = "none" | "text" | "image";

function buildRunAssetUrl(runDir: string, file: string): string {
  const params = new URLSearchParams({ run_dir: runDir, file });
  return `/api/run-asset?${params.toString()}`;
}

function slicePreview(text: string, limit = 12_000): string {
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit)}\n… (truncated, ${text.length - limit} chars omitted)`;
}

export function RunForensicsPanel({ runDir }: RunForensicsPanelProps): React.ReactElement {
  const [assets, setAssets] = useState<string[]>([]);
  const [assetsLoading, setAssetsLoading] = useState(false);
  const [assetsError, setAssetsError] = useState<string | null>(null);
  const [selectedAsset, setSelectedAsset] = useState<string>("");
  const [previewKind, setPreviewKind] = useState<PreviewKind>("none");
  const [previewText, setPreviewText] = useState<string>("");

  const [timeline, setTimeline] = useState<GenerationSummary[]>([]);
  const [timelineError, setTimelineError] = useState<string | null>(null);
  const [compareA, setCompareA] = useState<number | null>(null);
  const [compareB, setCompareB] = useState<number | null>(null);

  useEffect(() => {
    if (!runDir) {
      setAssets([]);
      setSelectedAsset("");
      setPreviewKind("none");
      setPreviewText("");
      setTimeline([]);
      setCompareA(null);
      setCompareB(null);
      return;
    }
    let cancelled = false;
    setAssetsLoading(true);
    setAssetsError(null);
    (async () => {
      try {
        const items = await fetchRunAssets(runDir);
        if (cancelled) return;
        setAssets(items);
        setSelectedAsset((prev) => {
          if (prev && items.includes(prev)) return prev;
          return items.includes("SUMMARY.json") ? "SUMMARY.json" : items[0] ?? "";
        });
      } catch (error) {
        if (!cancelled) {
          const detail = error instanceof Error ? error.message : String(error);
          setAssetsError(detail);
          setAssets([]);
        }
      } finally {
        if (!cancelled) {
          setAssetsLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runDir]);

  useEffect(() => {
    if (!runDir) {
      setTimeline([]);
      setTimelineError(null);
      return;
    }
    const timelinePath = "meta/gen_summary.jsonl";
    if (!assets.includes(timelinePath)) {
      setTimeline([]);
      setTimelineError(null);
      return;
    }
    let cancelled = false;
    setTimelineError(null);
    setTimeline([]);
    (async () => {
      const url = buildRunAssetUrl(runDir, timelinePath);
      try {
        const resp = await fetch(url);
        if (!resp.ok) {
          if (!cancelled) {
            setTimeline([]);
          }
          return;
        }
        const text = await resp.text();
        if (cancelled) return;
        const summaries: GenerationSummary[] = [];
        for (const line of text.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const raw = JSON.parse(trimmed) as unknown;
            const parsed = mapGenerationSummary(raw);
            if (parsed) {
              summaries.push(parsed);
            }
          } catch {
            continue;
          }
        }
        summaries.sort((a, b) => a.generation - b.generation);
        setTimeline(summaries);
        const last = summaries[summaries.length - 1];
        const prev = summaries.length >= 2 ? summaries[summaries.length - 2] : null;
        setCompareA(last?.generation ?? null);
        setCompareB(prev?.generation ?? null);
      } catch (error) {
        if (!cancelled) {
          const detail = error instanceof Error ? error.message : String(error);
          setTimelineError(detail);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [assets, runDir]);

  useEffect(() => {
    if (!runDir || !selectedAsset) {
      setPreviewKind("none");
      setPreviewText("");
      return;
    }
    const lower = selectedAsset.toLowerCase();
    if (lower.endsWith(".png")) {
      setPreviewKind("image");
      setPreviewText("");
      return;
    }
    let cancelled = false;
    setPreviewKind("text");
    setPreviewText("Loading…");
    (async () => {
      try {
        const resp = await fetch(buildRunAssetUrl(runDir, selectedAsset));
        if (!resp.ok) {
          throw new Error(`Request failed (${resp.status})`);
        }
        const text = await resp.text();
        if (!cancelled) {
          setPreviewText(slicePreview(text));
        }
      } catch (error) {
        if (!cancelled) {
          const detail = error instanceof Error ? error.message : String(error);
          setPreviewText(`Failed to preview: ${detail}`);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runDir, selectedAsset]);

  const comparePair = useMemo(() => {
    if (!compareA || !compareB) return null;
    const a = timeline.find((entry) => entry.generation === compareA) ?? null;
    const b = timeline.find((entry) => entry.generation === compareB) ?? null;
    if (!a || !b) return null;
    return { a, b };
  }, [compareA, compareB, timeline]);

  const diffMetrics = useMemo(() => {
    if (!comparePair) return [];
    const { a, b } = comparePair;
    const rows: Array<{ key: string; a: number; b: number; delta: number }> = [];
    const add = (key: string, aVal: number, bVal: number) => {
      if (!Number.isFinite(aVal) || !Number.isFinite(bVal)) return;
      rows.push({ key, a: aVal, b: bVal, delta: aVal - bVal });
    };
    add("fitness", a.best.fitness, b.best.fitness);
    add("mean_ic", a.best.meanIc, b.best.meanIc);
    add("sharpe_proxy", a.best.sharpeProxy, b.best.sharpeProxy);
    add("turnover", a.best.turnover, b.best.turnover);
    add("drawdown", a.best.drawdown, b.best.drawdown);
    return rows;
  }, [comparePair]);

  return (
    <div className="panel panel-run-details">
      <div className="panel-header">
        <h2>Run Forensics</h2>
        {runDir ? <span className="muted">{runDir}</span> : null}
      </div>

      {!runDir ? <p className="muted">Select a run to inspect its generation timeline and artefacts.</p> : null}

      {assetsError ? <p className="muted error-text">{assetsError}</p> : null}
      {assetsLoading ? <p className="muted">Scanning run artefacts…</p> : null}

      {runDir ? (
        <div className="run-param-grid">
          <div className="run-param-item">
            <div className="run-param-item__label">Artefacts</div>
            {assets.length ? (
              <>
                <select value={selectedAsset} onChange={(event) => setSelectedAsset(event.target.value)}>
                  {assets.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
                {selectedAsset ? (
                  <a className="btn btn--link" href={buildRunAssetUrl(runDir, selectedAsset)} target="_blank" rel="noreferrer">
                    Download / open
                  </a>
                ) : null}
              </>
            ) : (
              <p className="muted">No artefacts found.</p>
            )}

            {selectedAsset && previewKind === "image" ? (
              <img
                src={buildRunAssetUrl(runDir, selectedAsset)}
                alt={selectedAsset}
                style={{ width: "100%", borderRadius: 10, marginTop: 12 }}
              />
            ) : null}
            {selectedAsset && previewKind === "text" ? <pre className="log-viewer">{previewText}</pre> : null}
          </div>

          <div className="run-param-item">
            <div className="run-param-item__label">Generation timeline</div>
            {timelineError ? <p className="muted error-text">{timelineError}</p> : null}
            {!timeline.length ? (
              <p className="muted">No gen_summary history found (meta/gen_summary.jsonl).</p>
            ) : (
              <>
                <div className="table-scroll">
                  <table className="bt-table">
                    <thead>
                      <tr>
                        <th>Gen</th>
                        <th>Fitness</th>
                        <th>Mean IC</th>
                        <th>Sharpe proxy</th>
                        <th>Turnover</th>
                        <th>Drawdown</th>
                      </tr>
                    </thead>
                    <tbody>
                      {timeline.map((entry) => (
                        <tr
                          key={entry.generation}
                          style={{
                            cursor: "pointer",
                            background:
                              entry.generation === compareA || entry.generation === compareB
                                ? "rgba(52, 88, 140, 0.25)"
                                : undefined,
                          }}
                          onClick={() => {
                            setCompareB(compareA);
                            setCompareA(entry.generation);
                          }}
                        >
                          <td>{entry.generation}</td>
                          <td>{entry.best.fitness.toFixed(4)}</td>
                          <td>{entry.best.meanIc.toFixed(4)}</td>
                          <td>{entry.best.sharpeProxy.toFixed(3)}</td>
                          <td>{entry.best.turnover.toFixed(4)}</td>
                          <td>{entry.best.drawdown.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {comparePair ? (
                  <div style={{ marginTop: 12 }}>
                    <div className="muted" style={{ marginBottom: 8 }}>
                      Comparing Gen {comparePair.a.generation} vs Gen {comparePair.b.generation}
                    </div>
                    {diffMetrics.length ? (
                      <div className="table-scroll">
                        <table className="bt-table">
                          <thead>
                            <tr>
                              <th>Metric</th>
                              <th>A</th>
                              <th>B</th>
                              <th>Δ</th>
                            </tr>
                          </thead>
                          <tbody>
                            {diffMetrics.map((row) => (
                              <tr key={row.key}>
                                <td>{row.key}</td>
                                <td>{row.a.toFixed(4)}</td>
                                <td>{row.b.toFixed(4)}</td>
                                <td>{row.delta >= 0 ? "+" : ""}{row.delta.toFixed(4)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p className="muted">No comparable metrics available for this pair.</p>
                    )}
                    <div style={{ marginTop: 12 }}>
                      <div className="muted" style={{ marginBottom: 6 }}>
                        Program diff (raw)
                      </div>
                      <div className="run-param-grid">
                        <pre className="log-viewer">{comparePair.a.best.program || "n/a"}</pre>
                        <pre className="log-viewer">{comparePair.b.best.program || "n/a"}</pre>
                      </div>
                    </div>
                  </div>
                ) : null}
              </>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
