import React from "react";
import { cn } from "@/lib/utils";

export type TabId = "introduction" | "controls" | "overview" | "settings" | "experiments";

interface HeaderNavProps {
  active: TabId;
  onChange: (tab: TabId) => void;
}

const TABS: Array<{ id: TabId; label: string }> = [
  { id: "introduction", label: "Introduction" },
  { id: "controls", label: "Pipeline Controls" },
  { id: "overview", label: "Backtest Analysis" },
  { id: "experiments", label: "Experiments" },
  { id: "settings", label: "Settings & Presets" },
];

export function HeaderNav({ active, onChange }: HeaderNavProps): React.ReactElement {
  return (
    <nav className="flex items-center space-x-1 bg-white/5 backdrop-blur-sm p-1 rounded-xl border border-white/5" data-test="header-nav">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          className={cn(
            "relative px-4 py-2 text-sm font-medium transition-all duration-500 rounded-lg group",
            tab.id === active
              ? "text-primary-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
          data-test={`nav-${tab.id}`}
          onClick={() => onChange(tab.id)}
        >
          {tab.id === active && (
            <div className="absolute inset-0 bg-primary shadow-[0_0_15px_-3px_rgba(59,130,246,0.6)] rounded-lg animate-in fade-in zoom-in-95 duration-300" />
          )}
          <span className="relative z-10">{tab.label}</span>
          {tab.id !== active && (
            <div className="absolute inset-0 bg-white/0 group-hover:bg-white/5 rounded-lg transition-colors duration-300" />
          )}
        </button>
      ))}
    </nav>
  );
}
