EvolutionParams: generation and mutation knobs
================================================

AlphaProgram exposes optional heuristics to bias random generation and mutation
without changing the core operator set. These are grouped into a small
dataclass that you can pass into `random_program`, `mutate`, and `crossover`.

Fields
- vector_ops_bias: Probability to require vector‑producing ops when sampling.
- relation_ops_weight: Weight multiplier for `relation_*` ops.
- cs_ops_weight: Weight multiplier for `cs_*` ops.
- default_op_weight: Baseline weight for all ops.
- max_setup_ops / max_predict_ops / max_update_ops: Informative only; limits are still enforced by explicit args.
- ops_split_base: Tuple of base fractions for (setup, predict, update).
- ops_split_jitter: Jitter level added around the base split when seeding fresh programs.

Usage
```python
from alpha_framework.utils import EvolutionParams
params = EvolutionParams(
    vector_ops_bias=0.3,
    relation_ops_weight=3.0,
    cs_ops_weight=1.5,
    ops_split_base=(0.2, 0.6, 0.2),
    ops_split_jitter=0.2,
)
prog = AlphaProgram.random_program(feature_vars, state_vars, max_total_ops=32, params=params)
child = prog.mutate(feature_vars, state_vars, params=params)
offspring = prog.crossover(other_prog, params=params)
```

CLI/Config
The same knobs are exposed in `EvolutionConfig` and can be set via file/env/CLI:

```toml
[evolution]
vector_ops_bias = 0.3
relation_ops_weight = 3.0
cs_ops_weight = 1.5
default_op_weight = 1.0
ops_split_base_setup = 0.15
ops_split_base_predict = 0.70
ops_split_base_update = 0.15
ops_split_jitter = 0.0
```

```bash
uv run run_pipeline.py 5 --vector_ops_bias 0.3 --ops_split_base_predict 0.6
```

