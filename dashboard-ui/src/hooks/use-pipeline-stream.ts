import { useState, useRef, useCallback, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { fetchJobActivity } from "../modules/api"; // We might want to use react-query invalidation instead of direct API call
import { mapGenerationProgress, mapGenerationSummary } from "../modules/pipelineMapping";
import { GenerationSummary, PipelineJobState } from "../modules/types";
import { keys } from "./use-dashboard";

const SUMMARY_HISTORY_LIMIT = 400;
const LOG_HISTORY_LIMIT_CHARS = 50_000;
const CODENAME_RE = /Run codename:\s*([A-Za-z0-9-]+)/;

function extractCodenameFromRunDir(runDir: string): string | null {
    const base = runDir.split(/[\\/]/).pop() ?? "";
    const match = /^run_([^_]+)_g\d+_seed/.exec(base);
    return match?.[1] ?? null;
}

type StreamState = "connected" | "retrying" | "stale";

function formatError(error: unknown): string {
    if (error instanceof Error) return error.message;
    return String(error);
}

export function usePipelineStream(activeJobId: string | null) {
    const [job, setJob] = useState<PipelineJobState | null>(null);
    const [streamState, setStreamState] = useState<StreamState | null>(null);
    const eventSourceRef = useRef<EventSource | null>(null);
    const lastStreamEventAtRef = useRef<number>(0);
    const queryClient = useQueryClient();

    const closeEventStream = useCallback(() => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        setStreamState(null);
    }, []);

    const refreshJob = useCallback(
        async (jobIdOverride?: string) => {
            const jobId = jobIdOverride ?? activeJobId;
            if (!jobId) {
                return;
            }
            try {
                const snapshot = await fetchJobActivity(jobId);
                const summaryPayload = Array.isArray(snapshot.summaries) ? snapshot.summaries : [];
                const mappedSummaries = summaryPayload
                    .map((entry) => mapGenerationSummary(entry))
                    .filter((entry): entry is GenerationSummary => Boolean(entry));
                const trimmedSummaries =
                    mappedSummaries.length > SUMMARY_HISTORY_LIMIT
                        ? mappedSummaries.slice(mappedSummaries.length - SUMMARY_HISTORY_LIMIT)
                        : mappedSummaries;

                let progressState = mapGenerationProgress(snapshot.progress);
                if (!progressState && trimmedSummaries.length) {
                    const latestSummary = trimmedSummaries[trimmedSummaries.length - 1];
                    progressState = {
                        generation: latestSummary.generation,
                        generationsTotal: latestSummary.generationsTotal,
                        pctComplete: latestSummary.pctComplete,
                        completed: latestSummary.population.size,
                        totalIndividuals: latestSummary.population.size,
                        bestFitness: latestSummary.best.fitness,
                        medianFitness: latestSummary.best.meanIc,
                        elapsedSeconds: latestSummary.timing.generationSeconds,
                        etaSeconds: latestSummary.timing.etaSeconds,
                    };
                }
                const updatedAtMs = (() => {
                    const raw = snapshot.updated_at;
                    const numeric = typeof raw === "number" ? raw : Number(raw);
                    if (Number.isFinite(numeric)) {
                        return Math.max(Date.now(), Math.trunc(numeric > 1e12 ? numeric : numeric * 1000));
                    }
                    return Date.now();
                })();

                setJob((current) => {
                    const base: PipelineJobState =
                        current && current.jobId === jobId
                            ? current
                            : {
                                jobId,
                                status: "running",
                                lastMessage: "Pipeline running…",
                                lastUpdated: Date.now(),
                                log: "",
                                sharpeBest: null,
                                runDir: null,
                                runName: null,
                                progress: null,
                                summaries: [],
                            };
                    const next: PipelineJobState = {
                        ...base,
                        lastUpdated: updatedAtMs,
                    };
                    if (!snapshot.exists) {
                        next.status = "error";
                        next.lastMessage = "Job no longer exists.";
                        return next;
                    }
                    const statusRaw = snapshot.status;
                    if (typeof statusRaw === "string" && ["idle", "running", "error", "complete"].includes(statusRaw)) {
                        next.status = statusRaw as PipelineJobState["status"];
                    } else if (snapshot.running) {
                        next.status = "running";
                    } else if (next.status === "running") {
                        next.status = "complete";
                        if (!next.lastMessage) {
                            next.lastMessage = "Pipeline finished.";
                        }
                    }
                    const messageRaw = snapshot.last_message;
                    if (typeof messageRaw === "string" && messageRaw.trim()) {
                        next.lastMessage = messageRaw;
                    } else if (!next.lastMessage && next.status === "running") {
                        next.lastMessage = "Pipeline running…";
                    }
                    if (typeof snapshot.log === "string") {
                        next.log = snapshot.log;
                    }
                    const sharpeRaw = snapshot.sharpe_best;
                    if (sharpeRaw !== undefined && sharpeRaw !== null) {
                        const value = Number(sharpeRaw);
                        if (Number.isFinite(value)) {
                            next.sharpeBest = value;
                        }
                    }
                    const runDirRaw = snapshot.run_dir;
                    if (typeof runDirRaw === "string" && runDirRaw.trim()) {
                        next.runDir = runDirRaw;
                        if (!next.runName) {
                            next.runName = extractCodenameFromRunDir(runDirRaw);
                        }
                    }
                    if (progressState) {
                        next.progress = progressState;
                    }
                    if (trimmedSummaries.length) {
                        next.summaries = trimmedSummaries;
                    }
                    const logPath = snapshot.log_path;
                    if (typeof logPath === "string") {
                        next.logPath = logPath;
                    }
                    return next;
                });

                if (!snapshot.running || !snapshot.exists) {
                    // queryClient.invalidateQueries({ queryKey: keys.runs }); // Removed as per instruction
                }
            } catch (error) {
                setJob((current) => {
                    if (!current || current.jobId !== jobId) {
                        return current;
                    }
                    return {
                        ...current,
                        status: "error",
                        lastMessage: formatError(error),
                    };
                });
            }
        },
        [activeJobId, queryClient],
    );

    const applyPipelineEvent = useCallback(
        (activeJobId: string, raw: string) => {
            let payload: unknown;
            try {
                payload = JSON.parse(raw);
            } catch {
                return;
            }
            if (!payload || typeof payload !== "object") {
                return;
            }
            const event = payload as Record<string, unknown>;
            const eventType = event.type;

            if (eventType === "final" || (eventType === "status" && event.msg === "exit")) {
                queryClient.invalidateQueries({ queryKey: keys.runs });
            }

            setJob((current) => {
                if (!current || current.jobId !== activeJobId) {
                    return current;
                }
                const next: PipelineJobState = { ...current, lastUpdated: Date.now() };

                if (eventType === "log") {
                    const rawLine = typeof event.raw === "string" ? event.raw : "";
                    if (rawLine) {
                        if (!next.runName) {
                            const match = CODENAME_RE.exec(rawLine);
                            if (match?.[1]) {
                                next.runName = match[1];
                            }
                        }
                        const normalized = rawLine.endsWith("\n") ? rawLine : `${rawLine}\n`;
                        const combined = `${next.log || ""}${normalized}`;
                        next.log =
                            combined.length > LOG_HISTORY_LIMIT_CHARS
                                ? combined.slice(combined.length - LOG_HISTORY_LIMIT_CHARS)
                                : combined;
                    }
                    return next;
                }

                if (eventType === "progress") {
                    const progress = mapGenerationProgress(event.data);
                    if (progress) {
                        next.progress = progress;
                    }
                    return next;
                }

                if (eventType === "gen_summary") {
                    const summary = mapGenerationSummary(event.data);
                    if (summary) {
                        const summaries = next.summaries ?? [];
                        const idx = summaries.findIndex((entry) => entry.generation === summary.generation);
                        const updated = idx >= 0 ? summaries.map((entry, i) => (i === idx ? summary : entry)) : [...summaries, summary];
                        updated.sort((a, b) => a.generation - b.generation);
                        next.summaries =
                            updated.length > SUMMARY_HISTORY_LIMIT ? updated.slice(updated.length - SUMMARY_HISTORY_LIMIT) : updated;
                    }
                    return next;
                }

                if (eventType === "score") {
                    const sharpeRaw = event.sharpe_best ?? event.sharpeBest;
                    const sharpe = Number(sharpeRaw);
                    if (Number.isFinite(sharpe)) {
                        next.sharpeBest = sharpe;
                    }
                    return next;
                }

                if (eventType === "final") {
                    const sharpe = Number(event.sharpe_best ?? event.sharpeBest);
                    if (Number.isFinite(sharpe)) {
                        next.sharpeBest = sharpe;
                    }
                    const runDir = typeof event.run_dir === "string" ? event.run_dir : null;
                    if (runDir) {
                        next.runDir = runDir;
                        if (!next.runName) {
                            next.runName = extractCodenameFromRunDir(runDir);
                        }
                    }
                    next.status = "complete";
                    next.lastMessage = "Pipeline finished.";
                    return next;
                }

                if (eventType === "status") {
                    const msg = event.msg;
                    if (msg === "stop_requested") {
                        next.lastMessage = "Stop requested…";
                        return next;
                    }
                    if (msg === "exit") {
                        const code = Number(event.code);
                        next.status = code === 0 ? "complete" : "error";
                        next.lastMessage = code === 0 ? "Pipeline finished." : "Pipeline stopped.";
                        return next;
                    }
                    if (typeof msg === "string" && msg.trim()) {
                        next.lastMessage = msg;
                    }
                    return next;
                }

                if (eventType === "error") {
                    next.status = "error";
                    const detail = typeof event.detail === "string" ? event.detail : null;
                    next.lastMessage = detail && detail.trim() ? detail : "Pipeline error.";
                    return next;
                }

                return next;
            });
        },
        [queryClient],
    );

    useEffect(() => {
        const jobRunning = Boolean(job && job.status === "running");
        if (!activeJobId || !jobRunning) {
            closeEventStream();
            return;
        }
        if (typeof EventSource === "undefined") {
            setStreamState("stale");
            return;
        }

        const jobId = activeJobId;
        closeEventStream();
        setStreamState("retrying");
        lastStreamEventAtRef.current = Date.now();

        const url = `/api/pipeline/events/${encodeURIComponent(jobId)}`;
        const es = new EventSource(url);
        eventSourceRef.current = es;

        const touch = () => {
            lastStreamEventAtRef.current = Date.now();
        };

        const markConnected = () => {
            touch();
            setStreamState("connected");
        };

        es.onopen = () => {
            markConnected();
        };

        es.onmessage = (event) => {
            touch();
            applyPipelineEvent(jobId, event.data);
        };

        es.onerror = () => {
            touch();
            setStreamState((prev) => (prev === "connected" ? "retrying" : prev ?? "retrying"));
            void refreshJob(jobId);
        };

        es.addEventListener("ping", markConnected as EventListener);

        const staleInterval = window.setInterval(() => {
            const ageMs = Date.now() - lastStreamEventAtRef.current;
            setStreamState((prev) => {
                if (ageMs > 15_000) {
                    return prev === "connected" ? "stale" : prev;
                }
                return prev === "stale" ? "connected" : prev;
            });
        }, 2_000);

        return () => {
            window.clearInterval(staleInterval);
            es.close();
            if (eventSourceRef.current === es) {
                eventSourceRef.current = null;
            }
            setStreamState(null);
        };
    }, [activeJobId, job, applyPipelineEvent, closeEventStream, refreshJob]);

    // Polling fallback
    useEffect(() => {
        const jobRunning = Boolean(job && job.status === "running");
        if (!jobRunning || streamState === "connected") return;

        const interval = setInterval(() => {
            void refreshJob();
        }, 2500);
        return () => clearInterval(interval);
    }, [job, streamState, refreshJob]);

    return {
        job,
        streamState,
        setJob,
        refreshJob
    };
}
