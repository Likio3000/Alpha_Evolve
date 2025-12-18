import React, { useEffect, useMemo, useState } from "react";
import { fetchRunAssets } from "../api";
import { GenerationSummary } from "../types";
import { mapGenerationSummary } from "../pipelineMapping";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

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
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-xl font-bold">Run Forensics</CardTitle>
        {runDir ? <span className="text-xs font-mono text-muted-foreground">{runDir}</span> : null}
      </CardHeader>

      <CardContent>
        {!runDir ? (
          <div className="text-center py-6 text-muted-foreground text-sm">Select a run to inspect its generation timeline and artefacts.</div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-sm font-semibold border-b pb-2">Artefacts</h3>

              {assetsError ? <p className="text-xs text-destructive">{assetsError}</p> : null}
              {assetsLoading ? <p className="text-xs text-muted-foreground">Scanning run artefacts…</p> : null}

              {assets.length ? (
                <div className="flex gap-2">
                  <select
                    className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                    value={selectedAsset}
                    onChange={(event) => setSelectedAsset(event.target.value)}
                  >
                    {assets.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                  {selectedAsset ? (
                    <Button variant="outline" size="sm" asChild>
                      <a href={buildRunAssetUrl(runDir, selectedAsset)} target="_blank" rel="noreferrer">
                        Open
                      </a>
                    </Button>
                  ) : null}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">No artefacts found.</p>
              )}

              {selectedAsset && previewKind === "image" ? (
                <img
                  src={buildRunAssetUrl(runDir, selectedAsset)}
                  alt={selectedAsset}
                  className="w-full rounded-lg border mt-4"
                />
              ) : null}
              {selectedAsset && previewKind === "text" ? (
                <pre className="mt-4 h-[400px] overflow-y-auto rounded-lg border bg-muted/30 p-4 font-mono text-xs whitespace-pre-wrap">
                  {previewText}
                </pre>
              ) : null}
            </div>

            <div className="space-y-4">
              <h3 className="text-sm font-semibold border-b pb-2">Generation timeline</h3>

              {timelineError ? <p className="text-xs text-destructive">{timelineError}</p> : null}
              {!timeline.length ? (
                <p className="text-xs text-muted-foreground">No gen_summary history found (meta/gen_summary.jsonl).</p>
              ) : (
                <>
                  <div className="border rounded-md overflow-hidden max-h-[300px] overflow-y-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[60px]">Gen</TableHead>
                          <TableHead>Fit</TableHead>
                          <TableHead>IC</TableHead>
                          <TableHead>Sharpe</TableHead>
                          <TableHead>TO</TableHead>
                          <TableHead>DD</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {timeline.map((entry) => (
                          <TableRow
                            key={entry.generation}
                            className={cn(
                              "cursor-pointer hover:bg-muted/50",
                              (entry.generation === compareA || entry.generation === compareB) && "bg-muted"
                            )}
                            onClick={() => {
                              setCompareB(compareA);
                              setCompareA(entry.generation);
                            }}
                          >
                            <TableCell className="font-mono">{entry.generation}</TableCell>
                            <TableCell>{entry.best.fitness.toFixed(4)}</TableCell>
                            <TableCell>{entry.best.meanIc.toFixed(4)}</TableCell>
                            <TableCell>{entry.best.sharpeProxy.toFixed(3)}</TableCell>
                            <TableCell>{entry.best.turnover.toFixed(4)}</TableCell>
                            <TableCell>{entry.best.drawdown.toFixed(4)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>

                  {comparePair ? (
                    <div className="space-y-4 pt-4 border-t">
                      <div className="text-xs font-medium text-muted-foreground">
                        Comparing Gen {comparePair.a.generation} vs Gen {comparePair.b.generation}
                      </div>
                      {diffMetrics.length ? (
                        <div className="border rounded-md">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Metric</TableHead>
                                <TableHead>Gen {comparePair.a.generation}</TableHead>
                                <TableHead>Gen {comparePair.b.generation}</TableHead>
                                <TableHead>Δ</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {diffMetrics.map((row) => (
                                <TableRow key={row.key}>
                                  <TableCell className="font-medium">{row.key}</TableCell>
                                  <TableCell>{row.a.toFixed(4)}</TableCell>
                                  <TableCell>{row.b.toFixed(4)}</TableCell>
                                  <TableCell className={cn(row.delta > 0 ? "text-green-600" : row.delta < 0 ? "text-red-600" : "")}>
                                    {row.delta >= 0 ? "+" : ""}{row.delta.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      ) : (
                        <p className="text-xs text-muted-foreground">No comparable metrics available for this pair.</p>
                      )}

                      <div className="grid grid-cols-2 gap-2">
                        <div className="space-y-1">
                          <div className="text-xs text-muted-foreground">Program Gen {comparePair.a.generation}</div>
                          <pre className="h-[150px] overflow-y-auto rounded border bg-muted/30 p-2 font-mono text-[10px] whitespace-pre-wrap">
                            {comparePair.a.best.program || "n/a"}
                          </pre>
                        </div>
                        <div className="space-y-1">
                          <div className="text-xs text-muted-foreground">Program Gen {comparePair.b.generation}</div>
                          <pre className="h-[150px] overflow-y-auto rounded border bg-muted/30 p-2 font-mono text-[10px] whitespace-pre-wrap">
                            {comparePair.b.best.program || "n/a"}
                          </pre>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
