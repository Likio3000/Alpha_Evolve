import React from "react";
import { BacktestRow } from "../types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

interface BacktestTableProps {
  rows: BacktestRow[];
  selected?: string | null;
  onSelect?: (row: BacktestRow) => void;
}

const numberFormatter = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
});

const percentFormatter = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function formatNumber(value: number | null | undefined, formatter = numberFormatter): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return formatter.format(value);
}

export function BacktestTable({ rows, selected, onSelect }: BacktestTableProps): React.ReactElement {
  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-xl font-bold">Backtest Summary</CardTitle>
        {rows.length ? <span className="text-sm text-muted-foreground">{rows.length} alphas</span> : null}
      </CardHeader>
      <CardContent>
        {rows.length === 0 ? (
          <div className="text-center py-6 text-muted-foreground text-sm">Select a run to view its backtest summary.</div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Alpha</TableHead>
                <TableHead>Sharpe</TableHead>
                <TableHead>AnnRet</TableHead>
                <TableHead>AnnVol</TableHead>
                <TableHead>MaxDD</TableHead>
                <TableHead>Turnover</TableHead>
                <TableHead>Ops</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const key = row.TimeseriesFile || row.TS || row.AlphaID || "";
                const isSelected = selected === key;
                return (
                  <TableRow
                    key={key}
                    data-state={isSelected ? "selected" : undefined}
                    onClick={() => onSelect?.(row)}
                    className="cursor-pointer"
                  >
                    <TableCell>
                      <div className="flex flex-col">
                        <span className="font-medium">{row.AlphaID || "n/a"}</span>
                        <span className="text-xs text-muted-foreground">{row.TimeseriesFile || row.TS || ""}</span>
                      </div>
                    </TableCell>
                    <TableCell>{formatNumber(row.Sharpe)}</TableCell>
                    <TableCell>{formatNumber(row.AnnReturn, percentFormatter)}</TableCell>
                    <TableCell>{formatNumber(row.AnnVol, percentFormatter)}</TableCell>
                    <TableCell>{formatNumber(row.MaxDD, percentFormatter)}</TableCell>
                    <TableCell>{formatNumber(row.Turnover, percentFormatter)}</TableCell>
                    <TableCell>{row.Ops ?? "n/a"}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
