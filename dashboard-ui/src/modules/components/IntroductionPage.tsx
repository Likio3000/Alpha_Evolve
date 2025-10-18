import React from "react";

export function IntroductionPage(): React.ReactElement {
  return (
    <article className="intro-page" data-test="introduction-layout">
      <section className="intro-page__hero">
        <h2>Alpha Evolve in Context</h2>
        <p>
          Alpha Evolve is a research platform for discovering systematic equity alphas via evolutionary
          program synthesis. The platform couples a configurable genetic programming engine with a
          reproducible backtesting stack, rich diagnostics, and real-time observability. This page
          summarizes the scientific framing so every user understands what the pipeline optimizes,
          how it measures evidence, and which safeguards are in place before deploying a factor.
        </p>
        <div className="intro-page__hero-grid">
          <div>
            <h3>Research mandate</h3>
            <p>
              Produce interpretable, cross-sectional alpha expressions that survive held-out validation
              while satisfying turnover and risk constraints.
            </p>
          </div>
          <div>
            <h3>Methodological spine</h3>
            <p>
              Genetic programming over a domain-specific language of factor operators, evaluated on
              rolling, cross-sectional datasets with strict alignment and risk controls.
            </p>
          </div>
          <div>
            <h3>Evidence standard</h3>
            <p>
              Multi-horizon fitness (Sharpe, IC, turnover, drawdown) with train/validation separation,
              QD archives, and hall-of-fame correlation guards to prioritise robust signals over noise.
            </p>
          </div>
        </div>
      </section>

      <section className="intro-page__section">
        <header>
          <h3>System Architecture</h3>
          <p>
            Alpha Evolve is organised as a three-layer stack that keeps research concerns decoupled:
          </p>
        </header>
        <ol className="intro-page__list intro-page__list--numbered">
          <li>
            <strong>Data &amp; Context.</strong> The evaluation context is built by{" "}
            <code>make_eval_context_from_dir</code>; it loads aligned OHLC bars, engineered features,
            optional sector mapping, and caches them in shared-memory matrices. Alignment guarantees
            that every candidate observes the same training universe, honours evaluation lags, and
            enforces a minimum history window before any fitness is computed.
          </li>
          <li>
            <strong>Evolution Engine.</strong> The engine in <code>alpha_evolve.evolution.engine</code>{" "}
            generates candidate programs, evaluates them in a multiprocessing pool, updates the hall of
            fame, and emits diagnostics. It exposes every hyperparameter via <code>EvoConfig</code> so
            you can tune population sizes, operator distributions, novelty weighting, or cross-validation
            settings without touching core code.
          </li>
          <li>
            <strong>Experiment Orchestration.</strong> The dashboard server spawns the pipeline in an
            isolated process, streams structured SSE updates (<code>status</code>, <code>progress</code>,
            <code>score</code>, <code>diag</code>), persists artefacts under <code>pipeline_runs_cs/</code>,
            and exposes REST endpoints plus the self-evolution controller for closed-loop research.
          </li>
        </ol>
      </section>

      <section className="intro-page__section">
        <header>
          <h3>Alpha Representation &amp; Search Dynamics</h3>
        </header>
        <div className="intro-page__grid">
          <div>
            <h4>Domain-specific language</h4>
            <p>
              Candidates are <code>AlphaProgram</code> instances with three phases:{" "}
              <em>setup</em> (state initialisation), <em>predict</em> (cross-sectional score
              computation), and <em>update</em> (state roll-forward). The opcode library combines price
              transforms, rolling statistics, rank operators, boolean gates, and vector arithmetic tuned
              for daily equity universes. Operator selection is biased toward vector-returning
              expressions to stabilise cross-sectional ranking.
            </p>
          </div>
          <div>
            <h4>Initialisation &amp; variation</h4>
            <p>
              Tree initialisation respects depth limits per phase and samples features from{" "}
              <code>CROSS_SECTIONAL_FEATURE_VECTOR_NAMES</code>. Evolution uses a mix of subtree
              crossover, point mutation, shrink, and insert operators. Ramp/fixed selection policies,
              softmax tournament parameters, and novelty boosts are all configurable to balance
              exploitation versus exploration.
            </p>
          </div>
          <div>
            <h4>Quality-diversity tooling</h4>
            <p>
              The engine maintains (optional) QD archives over turnover and structural complexity in
              addition to the classical hall of fame. Multi-objective optimisation (NSGA-II-lite) is
              available to trade off Sharpe, risk, and structural cost simultaneously. Correlation-aware
              hall-of-fame updates ensure that new champions contribute orthogonal information instead of
              rediscovering existing profiles.
            </p>
          </div>
        </div>
      </section>

      <section className="intro-page__section">
        <header>
          <h3>Fitness, Validation, and Risk Controls</h3>
        </header>
        <div className="intro-page__grid intro-page__grid--two">
          <div>
            <h4>Primary objectives</h4>
            <p>
              Fitness aggregates multi-horizon Sharpe, information coefficient (IC), and downside
              statistics. Evaluation horizons are derived from <code>evaluation_horizons</code> or the
              configured lag, enabling simultaneous assessment of short and medium-term efficacy. Train /
              validation splits, CPCV-style folds, and embargo windows are supported to detect overfit
              programs before they enter the hall of fame.
            </p>
          </div>
          <div>
            <h4>Penalty stack</h4>
            <ul className="intro-page__list">
              <li>
                <strong>Parsimony.</strong> Weighted by <code>parsimony_penalty</code> and injected
                jitter to discourage over-complex expressions.
              </li>
              <li>
                <strong>Turnover &amp; IC variance.</strong> Penalties for excessive trading costs or
                unstable cross-sectional relationships (<code>turnover_penalty_w</code>,{" "}
                <code>ic_std_penalty_w</code>).
              </li>
              <li>
                <strong>Correlation.</strong> <code>corr_penalty_w</code> and <code>hof_corr_mode</code>{" "}
                ensure new elites add orthogonal alpha. Temporal decay half-lives and factor exposure
                penalties further align with risk budgeting constraints.
              </li>
              <li>
                <strong>Early abort guards.</strong> Temporal/cross-sectional flatness checks and
                rolling window sanity tests abort evaluations early when a program degenerates, saving
                compute and avoiding NaNs.
              </li>
            </ul>
          </div>
        </div>
        <p>
          Every evaluation emits structured JSON summaries (per-generation champion metrics, penalty
          breakdowns, horizon table) that the dashboard replays in real time. Final champions, ranked
          by adjusted fitness, flow into <code>SUMMARY.json</code>, <code>backtest_summary_topN.csv</code>,
          and the hall-of-fame export.
        </p>
      </section>

      <section className="intro-page__section">
        <header>
          <h3>Backtesting &amp; Diagnostics</h3>
        </header>
        <p>
          After evolution, the pipeline performs a deterministic backtest over the selected champions.
          It computes equity curves, realised turnover, drawdowns, and a full attribution table. Artefacts
          under <code>backtest_portfolio_csvs/</code> mirror the UI tables, while{" "}
          <code>meta/gen_summary.jsonl</code> captures the entire evolutionary trajectory for offline
          analysis. Diagnostics modules surface candidate-level tracebacks, feature usage statistics,
          QD archive snapshots, and optional MOEA fronts to guide follow-up research.
        </p>
        <div className="intro-page__callouts">
          <div>
            <h4>Key artefacts</h4>
            <ul className="intro-page__list">
              <li>
                <code>SUMMARY.json</code> — run-wide metrics (best Sharpe, IC, max drawdown, turnover).
              </li>
              <li>
                <code>meta/ui_context.json</code> — submission payload, config snapshot, timestamps.
              </li>
              <li>
                <code>generated_configs/</code> — TOML copies of saved presets and overrides.
              </li>
              <li>
                <code>logs/pipeline_*.log</code> — full text logs mirrored in the Job Console.
              </li>
            </ul>
          </div>
          <div>
            <h4>Observability</h4>
            <p>
              Real-time progress streams through server-sent events; the UI plots generation summaries,
              best-of-run equity curves, and exposes log excerpts. Replayability is preserved: the same
              JSON lines powering the live console are written to disk for regression comparisons or
              scientific reporting.
            </p>
          </div>
        </div>
      </section>

      <section className="intro-page__section">
        <header>
          <h3>Automation &amp; Outer-Loop Experimentation</h3>
        </header>
        <p>
          Beyond single runs, Alpha Evolve embeds automation hooks for large studies:
        </p>
        <ul className="intro-page__list">
          <li>
            <strong>Self-Evolution Sessions.</strong> <code>scripts/self_evolve.py</code> perturbs config
            parameters according to a declarative search space, launches inner pipeline runs, and pauses
            for human or agent approval. Each iteration logs a briefing, objective score, and suggested
            follow-up to <code>agent_briefings.jsonl</code>.
          </li>
          <li>
            <strong>Server Manager &amp; Iteration harness.</strong> The dashboard server is managed via{" "}
            <code>scripts/dev/server_manager.py</code>, while{" "}
            <code>scripts/dev/run_iteration.py</code> rotates artefact slots, rebuilds the UI, and captures
            Playwright screenshots for auditability.
          </li>
          <li>
            <strong>API surface.</strong> Everything visible in the UI is exposed through REST endpoints
            (<code>/api/pipeline/run</code>, <code>/api/runs</code>, <code>/ui-meta/*</code>) so you can
            orchestrate studies programmatically or integrate Alpha Evolve into a broader research stack.
          </li>
        </ul>
      </section>

      <section className="intro-page__section">
        <header>
          <h3>How to Use the Dashboard</h3>
        </header>
        <div className="intro-page__grid intro-page__grid--three">
          <div>
            <h4>Backtest Analysis</h4>
            <p>
              Explore historical runs, inspect champion tables, and review cross-sectional equity curves.
              Use the run list to relabel experiments, compare Sharpe trajectories, or download artefacts
              straight from the backend.
            </p>
          </div>
          <div>
            <h4>Pipeline Controls</h4>
            <p>
              Launch fresh evolution jobs, monitor real-time logs, and react when diagnostics flag an
              issue. The console streams both stdout and structured SSE events, while the control panel
              mirrors <code>EvolutionConfig</code> defaults and presets.
            </p>
          </div>
          <div>
            <h4>Settings &amp; Presets</h4>
            <p>
              Browse the entire configuration surface area, import presets, or persist curated TOML
              snapshots. This serves as the knowledge base linking UI choices to concrete pipeline
              arguments.
            </p>
          </div>
        </div>
        <p className="intro-page__closing">
          Continue into the Backtest Analysis tab to study prior discoveries, or open Pipeline Controls to
          launch your own experiment. All artefacts generated downstream remain linked back to this
          overview so collaborators can trace assumptions and reproduce results.
        </p>
      </section>
    </article>
  );
}
