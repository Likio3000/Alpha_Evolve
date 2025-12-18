import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as api from "../modules/api"; // Adjust path if needed
import { RunSummary, BacktestRow, AlphaTimeseries, PipelineRunRequest } from "../modules/types";

// Keys
export const keys = {
    runs: ["runs"],
    runDetails: (path: string) => ["run", path],
    backtest: (path: string) => ["backtest", path],
    timeseries: (path: string, alphaId?: string, file?: string) => ["timeseries", path, alphaId, file],
    job: (id: string) => ["job", id],
};

// Hooks
export function useRuns(limit = 50) {
    return useQuery({
        queryKey: keys.runs,
        queryFn: () => api.fetchRuns(limit),
        staleTime: 10 * 1000,
        refetchInterval: 30 * 1000,
    });
}

export function useRunDetails(path: string | null) {
    return useQuery({
        queryKey: keys.runDetails(path!),
        queryFn: () => api.fetchRunDetails(path!),
        enabled: !!path,
    });
}

export function useBacktestSummary(path: string | null) {
    return useQuery({
        queryKey: keys.backtest(path!),
        queryFn: () => api.fetchBacktestSummary(path!),
        enabled: !!path,
    });
}

export function useAlphaTimeseries(path: string | null, alphaId?: string, file?: string) {
    return useQuery({
        queryKey: keys.timeseries(path!, alphaId, file),
        queryFn: () => api.fetchAlphaTimeseries(path!, alphaId, file),
        enabled: !!path && (!!alphaId || !!file),
    });
}

// Mutations
export function useStartPipeline() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (data: PipelineRunRequest) => api.startPipelineRun(data),
        onSuccess: () => {
            // Invalidate runs shortly after start, or relying on SSE
            queryClient.invalidateQueries({ queryKey: keys.runs });
        },
    });
}

export function useStopJob() {
    return useMutation({
        mutationFn: (jobId: string) => api.stopPipelineJob(jobId),
    });
}

export function useUpdateRunLabel() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (vars: { path: string, label: string }) => api.updateRunLabel(vars),
        onSuccess: (_, vars) => {
            queryClient.invalidateQueries({ queryKey: keys.runs });
            queryClient.invalidateQueries({ queryKey: keys.runDetails(vars.path) });
        }
    })
}
