import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Beaker, Cpu, Gavel, Microscope, Network, Zap } from "lucide-react";

export function IntroductionPage(): React.ReactElement {
  return (
    <article className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-1000" data-test="introduction-layout">
      {/* Hero Section */}
      <section className="text-center max-w-4xl mx-auto space-y-6 pt-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold uppercase tracking-wider mb-2">
          <Zap className="w-3 h-3" />
          <span>Research Platform v0.1.0</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-heading font-bold tracking-tight bg-gradient-to-b from-foreground to-foreground/70 bg-clip-text text-transparent">
          Discovery via Evolutionary Synthesis
        </h2>
        <p className="text-xl text-muted-foreground leading-relaxed max-w-2xl mx-auto">
          Alpha Evolve couples high-performance genetic programming with a reproducible backtesting stack for industrial-grade alpha discovery.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left pt-10">
          <Card className="glass-card premium-border group">
            <CardHeader className="pb-2">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform duration-500">
                <Microscope className="w-5 h-5 text-primary" />
              </div>
              <CardTitle className="text-lg">Research Mandate</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground leading-relaxed">
              Produce interpretable, cross-sectional alpha expressions that survive held-out validation while satisfying strict turnover and risk constraints.
            </CardContent>
          </Card>

          <Card className="glass-card premium-border group">
            <CardHeader className="pb-2">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform duration-500">
                <Cpu className="w-5 h-5 text-primary" />
              </div>
              <CardTitle className="text-lg">Methodological Spine</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground leading-relaxed">
              Genetic programming over a domain-specific DSL, evaluated on rolling datasets with rigorous alignment and execution-lag controls.
            </CardContent>
          </Card>

          <Card className="glass-card premium-border group">
            <CardHeader className="pb-2">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform duration-500">
                <Gavel className="w-5 h-5 text-primary" />
              </div>
              <CardTitle className="text-lg">Evidence Standard</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground leading-relaxed">
              Multi-horizon fitness (Sharpe, IC, Turnover) with temporal separation, QD archives, and hall-of-fame correlation guards.
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Architecture Section */}
      <section className="space-y-6 pt-8">
        <div className="flex flex-col gap-2 items-center text-center">
          <h3 className="text-2xl font-bold tracking-tight">System Architecture</h3>
          <p className="text-muted-foreground max-w-xl">
            A decoupled three-layer stack designed for research velocity and execution fidelity.
          </p>
        </div>

        <div className="grid gap-4">
          <div className="glass-panel p-6 rounded-2xl flex gap-6 items-start hover:bg-white/[0.02] transition-colors border-white/5">
            <div className="w-12 h-12 rounded-xl bg-white/5 flex-shrink-0 flex items-center justify-center border border-white/10">
              <span className="text-xl font-bold text-primary">01</span>
            </div>
            <div className="space-y-2">
              <h4 className="font-bold text-lg">Data &amp; Context</h4>
              <p className="text-sm text-muted-foreground leading-relaxed max-w-3xl">
                Evaluation context built via <code className="bg-white/5 px-2 py-0.5 rounded font-mono text-primary text-xs">make_eval_context_from_dir</code>. Loads aligned OHLC bars and engineered features into shared-memory matrices. Guarantees evaluation lags and enforces history windows.
              </p>
            </div>
          </div>

          <div className="glass-panel p-6 rounded-2xl flex gap-6 items-start hover:bg-white/[0.02] transition-colors border-white/5">
            <div className="w-12 h-12 rounded-xl bg-white/5 flex-shrink-0 flex items-center justify-center border border-white/10">
              <span className="text-xl font-bold text-primary">02</span>
            </div>
            <div className="space-y-2">
              <h4 className="font-bold text-lg">Evolution Engine</h4>
              <p className="text-sm text-muted-foreground leading-relaxed max-w-3xl">
                High-performance engine in <code className="bg-white/5 px-2 py-0.5 rounded font-mono text-primary text-xs">alpha_evolve.evolution.engine</code>. Evaluates populations in multiprocessing pools, maintains elite archives, and exposes all hyperparameters via <code className="bg-white/5 px-2 py-0.5 rounded font-mono text-primary text-xs">EvoConfig</code>.
              </p>
            </div>
          </div>

          <div className="glass-panel p-6 rounded-2xl flex gap-6 items-start hover:bg-white/[0.02] transition-colors border-white/5">
            <div className="w-12 h-12 rounded-xl bg-white/5 flex-shrink-0 flex items-center justify-center border border-white/10">
              <span className="text-xl font-bold text-primary">03</span>
            </div>
            <div className="space-y-2">
              <h4 className="font-bold text-lg">Dashboard Orchestration</h4>
              <p className="text-sm text-muted-foreground leading-relaxed max-w-3xl">
                Isolated dashboard server streams SSE updates (<code className="bg-white/5 px-2 py-0.5 rounded font-mono text-primary text-xs">status</code>, <code className="bg-white/5 px-2 py-0.5 rounded font-mono text-xs">progress</code>), persists artefacts under <code className="bg-white/5 px-2 py-0.5 rounded font-mono text-primary text-xs">pipeline_runs_cs/</code>, and exposes REST endpoints for closed-loop self-evolution.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* search dynamics */}
      <section className="grid md:grid-cols-2 gap-12 pt-8">
        <div className="space-y-4">
          <h3 className="text-2xl font-bold tracking-tight border-b border-white/5 pb-2">Search Dynamics</h3>
          <div className="space-y-6">
            <div className="flex gap-4">
              <Network className="w-6 h-6 text-primary flex-shrink-0" />
              <div className="space-y-1">
                <h4 className="font-bold">Domain-specific language</h4>
                <p className="text-sm text-muted-foreground">Candidates use three-phase logic (setup, predict, update) with an opcode library tuned for cross-sectional factor construction.</p>
              </div>
            </div>
            <div className="flex gap-4">
              <Beaker className="w-6 h-6 text-primary flex-shrink-0" />
              <div className="space-y-1">
                <h4 className="font-bold">Initialisation &amp; variation</h4>
                <p className="text-sm text-muted-foreground">Depth-limited tree init combined with subtree crossover and point mutation ensures efficient traversals of the program space.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-2xl font-bold tracking-tight border-b border-white/5 pb-2">Risk &amp; Validation</h3>
          <div className="space-y-4 p-4 rounded-xl bg-primary/[0.03] border border-primary/10">
            <h4 className="font-bold text-sm uppercase tracking-wider text-primary">Penalty Stack</h4>
            <ul className="space-y-3 text-sm">
              <li className="flex gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5" />
                <span><strong>Parsimony.</strong> Complexity-weighted penalties to prevent over-parametrisation.</span>
              </li>
              <li className="flex gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5" />
                <span><strong>Turnover.</strong> Real-world cost constraints enforced during evaluation.</span>
              </li>
              <li className="flex gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5" />
                <span><strong>HOF Correlation.</strong> Ensures new elites provide orthogonal alpha.</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Quick Start/CTA */}
      <section className="p-8 rounded-[2rem] bg-gradient-to-tr from-primary/20 via-primary/5 to-transparent border border-white/10 text-center space-y-4 overflow-hidden relative">
        <div className="absolute top-0 right-0 -mr-20 -mt-20 w-64 h-64 bg-primary/20 blur-[100px] pointer-events-none" />
        <h3 className="text-xl font-bold">Ready to evolve?</h3>
        <p className="text-sm text-muted-foreground max-w-md mx-auto">
          Navigate to <strong>Pipeline Controls</strong> to launch a new run, or <strong>Backtest Analysis</strong> to review results.
        </p>
      </section>
    </article>
  );
}
