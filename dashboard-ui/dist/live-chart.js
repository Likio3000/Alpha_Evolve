(function(){
  "use strict";

  const GEN_LINE_COLORS = ['#5ab4f0', '#6ee7b7', '#f6c560', '#c792ea', '#ef6d7a'];
  const GEN_BAND_COLORS = [
    'rgba(90,180,240,0.10)',
    'rgba(110,231,183,0.10)',
    'rgba(246,197,96,0.10)',
    'rgba(199,146,234,0.10)',
    'rgba(239,109,122,0.10)',
  ];

  function clampNumber(value){
    if (value === null || value === undefined) return null;
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function formatMetric(value){
    const num = clampNumber(value);
    if (num === null) return "n/a";
    return num >= 0 ? num.toFixed(4) : num.toFixed(4);
  }

  class LiveChart {
    constructor(canvas, metaElement){
      this.canvas = canvas;
      this.ctx = canvas && canvas.getContext ? canvas.getContext("2d") : null;
      this.metaEl = metaElement || null;
      this.bestSeries = [];
      this.medianSeries = [];
      this.lastProgress = null;
      this.lastSharpe = null;
      this.state = "idle";
      this._metaTemplate = "No active run.";
      this.resize();
      this._draw();
    }

    resize(){
      if (!this.canvas) return;
      const rect = this.canvas.getBoundingClientRect();
      const width = Math.max(420, rect.width | 0);
      const height = Math.max(200, rect.height | 0);
      if (this.canvas.width !== width) this.canvas.width = width;
      if (this.canvas.height !== height) this.canvas.height = height;
      this._draw();
    }

    setIdle(){
      this.state = "idle";
      this.bestSeries = [];
      this.medianSeries = [];
      this.lastProgress = null;
      this.lastSharpe = null;
      this._metaTemplate = "No active run.";
      this._draw();
      this._updateMeta();
    }

    startRun(){
      this.state = "running";
      this.bestSeries = [];
      this.medianSeries = [];
      this.lastProgress = null;
      this.lastSharpe = null;
      this._metaTemplate = "Run initialising…";
      this._draw();
      this._updateMeta();
    }

    addProgress(progress){
      if (!progress || this.state === "idle") return;
      const gen = clampNumber(progress.gen) || 0;
      const total = clampNumber(progress.total) || 1;
      const completed = clampNumber(progress.completed) || 0;
      const best = clampNumber(progress.best);
      const median = clampNumber(progress.median);
      const frac = total > 0 ? completed / total : 0;
      const x = gen > 0 ? (gen - 1) + Math.max(0, Math.min(1, frac)) : completed;
      if (best !== null) this._pushPoint(this.bestSeries, x, best, gen);
      if (median !== null) this._pushPoint(this.medianSeries, x, median, gen);
      this.lastProgress = {
        gen: gen || 0,
        completed: completed || 0,
        total: total || 0,
        best,
        median,
      };
      this._metaTemplate = `Gen ${gen} • Eval ${completed}/${total}`;
      this._draw();
      this._updateMeta();
    }

    setSharpe(sharpe){
      const value = clampNumber(sharpe);
      if (value === null) return;
      this.lastSharpe = value;
      this._updateMeta();
    }

    finish(opts){
      this.state = "finished";
      if (opts && opts.exitCode !== undefined && opts.exitCode !== 0){
        this._metaTemplate = `Run exited with code ${opts.exitCode}`;
      } else {
        this._metaTemplate = "Run completed.";
      }
      if (opts && typeof opts.sharpeBest === "number") {
        this.lastSharpe = clampNumber(opts.sharpeBest);
      }
      this._draw();
      this._updateMeta();
    }

    _pushPoint(series, x, y, gen){
      if (!series) return;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      const last = series.length ? series[series.length - 1] : null;
      const nextX = last && x <= last.x ? last.x + 1e-6 : x;
      series.push({ x: nextX, y, gen: typeof gen === "number" ? gen : null });
    }

    _draw(){
      const ctx = this.ctx;
      if (!ctx || !this.canvas) return;
      const w = this.canvas.width;
      const h = this.canvas.height;
      ctx.save();
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#0b1014";
      ctx.fillRect(0, 0, w, h);

      const seriesList = [];
      if (this.bestSeries.length > 1) {
        seriesList.push({ pts: this.bestSeries, dash: [] });
      }
      if (this.medianSeries.length > 1) {
        seriesList.push({ pts: this.medianSeries, dash: [6, 4] });
      }

      if (!seriesList.length) {
        ctx.strokeStyle = "#1f242a";
        ctx.lineWidth = 1;
        for (let i = 0; i < 6; i++) {
          const y = ((h - 20) * i) / 5 + 10;
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(w, y);
          ctx.stroke();
        }
        ctx.fillStyle = "#2b3440";
        ctx.font = "12px monospace";
        const placeholder = this.state === "running" ? "Waiting for evaluations…" : "No chart data";
        ctx.fillText(placeholder, 12, h - 16);
        ctx.restore();
        return;
      }

      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      seriesList.forEach(series => {
        series.pts.forEach(pt => {
          if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return;
          if (pt.x < minX) minX = pt.x;
          if (pt.x > maxX) maxX = pt.x;
          if (pt.y < minY) minY = pt.y;
          if (pt.y > maxY) maxY = pt.y;
        });
      });

      if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
        minX = 0;
        maxX = 1;
      }
      if (minX === maxX) {
        maxX = minX + 1;
      }
      if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
        minY = -1;
        maxY = 1;
      }
      if (minY === maxY) {
        const delta = Math.max(1, Math.abs(minY) * 0.1 + 0.1);
        minY -= delta;
        maxY += delta;
      }

      const sx = v => 10 + ((w - 20) * (v - minX)) / (maxX - minX || 1);
      const sy = v => h - 10 - ((h - 20) * (v - minY)) / (maxY - minY || 1);

      const genStart = Math.floor(minX);
      let genEnd = Math.ceil(maxX);
      if (genEnd <= genStart) genEnd = genStart + 1;

      for (let g = genStart; g < genEnd; g++) {
        const start = sx(g);
        const end = sx(g + 1);
        const bandColor = GEN_BAND_COLORS[(g - genStart) % GEN_BAND_COLORS.length];
        ctx.fillStyle = bandColor;
        ctx.fillRect(start, 0, Math.max(1, end - start), h);
      }

      ctx.strokeStyle = "#1f242a";
      ctx.lineWidth = 1;
      for (let i = 0; i < 6; i++) {
        const y = ((h - 20) * i) / 5 + 10;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      ctx.save();
      ctx.strokeStyle = "rgba(90,180,240,0.18)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 6]);
      for (let g = genStart; g < genEnd; g++) {
        const xPos = sx(g);
        ctx.beginPath();
        ctx.moveTo(xPos, 10);
        ctx.lineTo(xPos, h - 10);
        ctx.stroke();
      }
      ctx.restore();

      if (minY < 0 && maxY > 0) {
        ctx.strokeStyle = "#2a3e50";
        ctx.beginPath();
        const zeroY = sy(0);
        ctx.moveTo(0, zeroY);
        ctx.lineTo(w, zeroY);
        ctx.stroke();
      }

      const colorCount = GEN_LINE_COLORS.length;
      seriesList.forEach(series => {
        ctx.save();
        ctx.setLineDash(series.dash || []);
        ctx.lineWidth = series.dash && series.dash.length ? 1.5 : 2;
        let prevPoint = null;
        let prevGen = null;
        let prevX = 0;
        let prevY = 0;
        let pathActive = false;
        for (let i = 0; i < series.pts.length; i++) {
          const pt = series.pts[i];
          if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) {
            if (pathActive) {
              ctx.stroke();
              pathActive = false;
            }
            prevPoint = null;
            prevGen = null;
            continue;
          }
          const x = sx(pt.x);
          const y = sy(pt.y);
          const genIdx = Math.max(0, Math.floor(pt.x + 1e-6));
          const color = GEN_LINE_COLORS[genIdx % colorCount];
          if (!prevPoint) {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.moveTo(x, y);
            pathActive = true;
          } else {
            if (genIdx !== prevGen) {
              ctx.lineTo(prevX, prevY);
              ctx.stroke();
              ctx.beginPath();
              ctx.strokeStyle = color;
              ctx.moveTo(prevX, prevY);
            }
            ctx.lineTo(x, y);
          }
          prevPoint = pt;
          prevGen = genIdx;
          prevX = x;
          prevY = y;
        }
        if (pathActive) {
          ctx.stroke();
        }
        ctx.restore();
      });

      ctx.setLineDash([]);
      ctx.restore();
    }

    _updateMeta(){
      if (!this.metaEl) return;
      let text = this._metaTemplate;
      if (this.state === "running" && this.lastProgress) {
        const { gen, completed, total, best, median } = this.lastProgress;
        const parts = [
          `Gen ${gen || 0}`,
          `Eval ${completed || 0}/${total || 0}`,
        ];
        if (best !== null) parts.push(`Best ${formatMetric(best)}`);
        if (median !== null) parts.push(`Median ${formatMetric(median)}`);
        if (this.lastSharpe !== null) parts.push(`Sharpe ${formatMetric(this.lastSharpe)}`);
        text = parts.join(" • ");
      } else if (this.state === "finished") {
        const parts = [text];
        if (this.lastProgress && this.lastProgress.best !== null) {
          parts.push(`Live best ${formatMetric(this.lastProgress.best)}`);
        }
        if (this.lastSharpe !== null) {
          parts.push(`Sharpe ${formatMetric(this.lastSharpe)}`);
        }
        text = parts.join(" • ");
      } else if (this.state === "idle") {
        text = this._metaTemplate;
      }
      this.metaEl.textContent = text;
    }
  }

  window.LiveChart = LiveChart;
})();
