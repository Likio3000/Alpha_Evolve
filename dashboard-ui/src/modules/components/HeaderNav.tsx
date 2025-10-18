import React from "react";

export type TabId = "introduction" | "overview" | "controls" | "settings";

interface HeaderNavProps {
  active: TabId;
  onChange: (tab: TabId) => void;
}

const TABS: Array<{ id: TabId; label: string }> = [
  { id: "introduction", label: "Introduction" },
  { id: "overview", label: "Backtest Analysis" },
  { id: "controls", label: "Pipeline Controls" },
  { id: "settings", label: "Settings & Presets" },
];

export function HeaderNav({ active, onChange }: HeaderNavProps): React.ReactElement {
  return (
    <div className="header-nav" data-test="header-nav">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          className={tab.id === active ? "header-nav__btn header-nav__btn--active" : "header-nav__btn"}
          data-test={`nav-${tab.id}`}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
