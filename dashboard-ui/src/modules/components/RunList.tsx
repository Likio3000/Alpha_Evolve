import React, { useCallback, useState, useEffect } from "react";
import { RunSummary } from "../types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RefreshCw, Edit2 } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface RunListProps {
  runs: RunSummary[];
  selected?: string | null;
  onSelect?: (run: RunSummary) => void;
  onRelabel?: (run: RunSummary, newLabel: string) => void;
  onRefresh?: () => void;
  loading?: boolean;
}

export function RunList({
  runs,
  selected,
  loading,
  onSelect,
  onRelabel,
  onRefresh,
}: RunListProps): React.ReactElement {
  const [editingRun, setEditingRun] = useState<RunSummary | null>(null);
  const [editLabel, setEditLabel] = useState("");

  const handleEditClick = useCallback((run: RunSummary) => {
    setEditingRun(run);
    setEditLabel(run.label ?? run.name ?? "");
  }, []);

  const handleSaveLabel = useCallback(() => {
    if (editingRun && onRelabel) {
      onRelabel(editingRun, editLabel.trim());
      setEditingRun(null);
    }
  }, [editLabel, editingRun, onRelabel]);

  return (
    <>
      <div className="h-full flex flex-col overflow-hidden bg-transparent">
        <div className="flex flex-row items-center justify-between py-4 px-6 border-b border-white/5">
          <div className="flex flex-col">
            <h3 className="text-xs font-heading font-bold uppercase tracking-[0.2em] text-primary glow-text">Recent Runs</h3>
            <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider mt-0.5">Experiment History</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onRefresh}
            disabled={loading}
            className="h-8 w-8 hover:bg-white/5 text-muted-foreground hover:text-foreground transition-all duration-300"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
            <span className="sr-only">Refresh</span>
          </Button>
        </div>

        <div className="flex-1 overflow-auto p-0 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
          {loading && runs.length === 0 && (
            <div className="flex flex-col items-center justify-center p-12 gap-4">
              <div className="w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
              <div className="text-[10px] text-muted-foreground uppercase tracking-widest font-mono">Hydrating...</div>
            </div>
          )}

          {!loading && runs.length === 0 && (
            <div className="p-12 text-center text-xs text-muted-foreground uppercase tracking-widest font-mono">Archive Empty</div>
          )}

          <div className="flex flex-col gap-2 p-4">
            {runs.map((run, index) => {
              const isSelected = selected === run.path;
              return (
                <div
                  key={run.path}
                  className={cn(
                    "group flex items-center justify-between gap-4 p-4 rounded-xl text-sm transition-all duration-500 cursor-pointer border relative overflow-hidden",
                    isSelected
                      ? "bg-primary/10 border-primary/30 shadow-[0_4px_20px_-4px_rgba(59,130,246,0.3)]"
                      : "bg-white/[0.02] border-white/5 hover:bg-white/[0.05] hover:border-white/10 hover:translate-x-1"
                  )}
                  style={{ animationDelay: `${index * 50}ms` }}
                  onClick={() => onSelect?.(run)}
                >
                  <div className="flex flex-col min-w-0 flex-1 relative z-10">
                    <span className={cn("font-bold truncate transition-colors duration-500", isSelected ? "text-primary" : "text-foreground/90")}>
                      {run.label || run.name}
                    </span>
                    <span className="text-[10px] text-muted-foreground/60 truncate font-mono uppercase tracking-tighter mt-1">
                      {run.name}
                    </span>
                  </div>

                  <Button
                    variant="ghost"
                    size="icon"
                    className={cn(
                      "h-8 w-8 transition-all duration-500 shrink-0 relative z-10",
                      isSelected ? "opacity-100 text-primary bg-primary/20" : "opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground hover:bg-white/10"
                    )}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleEditClick(run);
                    }}
                    title="Rename"
                  >
                    <Edit2 className="h-3 w-3" />
                  </Button>

                  {/* Glass highlight effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/[0.02] to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-in-out pointer-events-none" />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <Dialog open={!!editingRun} onOpenChange={(open) => !open && setEditingRun(null)}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Edit Label</DialogTitle>
            <DialogDescription>
              Add a descriptive label to identify this experiment run.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="run-label" className="text-right">
                Label
              </Label>
              <Input
                id="run-label"
                value={editLabel}
                onChange={(e) => setEditLabel(e.target.value)}
                className="col-span-3"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSaveLabel();
                }}
              />
            </div>
            {editingRun && (
              <div className="grid grid-cols-4 items-center gap-4">
                <span className="text-right text-xs text-muted-foreground ml-auto col-span-1">ID</span>
                <span className="text-xs text-muted-foreground font-mono truncate col-span-3">{editingRun.name}</span>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button type="submit" onClick={handleSaveLabel}>Save changes</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
