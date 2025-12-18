from __future__ import annotations

from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


Scalar = Union[str, int, float, bool]


class PipelineRunRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    generations: int = Field(default=5, ge=1)
    dataset: Optional[str] = None
    config: Optional[str] = None
    data_dir: Optional[str] = None
    overrides: Optional[Dict[str, Scalar]] = None
    runner_mode: Optional[str] = Field(
        default=None,
        description="Execution mode for dashboard jobs: auto|multiprocessing|subprocess",
    )


class AutoImproveRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    iters: Optional[int] = Field(default=None, ge=1)
    gens: Optional[int] = Field(default=None, ge=1)
    base_config: Optional[str] = None
    data_dir: Optional[str] = None
    bt_top: Optional[int] = Field(default=None, ge=1)
    no_clean: Optional[bool] = None
    dry_run: Optional[bool] = None
    sweep_capacity: Optional[bool] = None
    seeds: Optional[str] = None
    out_summary: Optional[bool] = None


class SelfplayApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., min_length=1)
    notes: Optional[str] = None
    approved_candidate: Optional[Dict[str, Any]] = None
    approved_updates: Optional[Dict[str, Any]] = None
    session: Optional[str] = None


class SelfplayRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_space: str
    config: Optional[str] = None
    iterations: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None
    objective: Optional[str] = None
    minimize: Optional[bool] = False
    exploration_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    auto_approve: Optional[bool] = None
    pipeline_output_dir: Optional[str] = None
    session_root: Optional[str] = None
    approval_poll_interval: Optional[float] = Field(default=None, gt=0.0)
    approval_timeout: Optional[float] = Field(default=None, gt=0.0)
    pipeline_log_level: Optional[str] = None
    debug_prints: Optional[bool] = None
    run_baselines: Optional[bool] = None
    retrain_baselines: Optional[bool] = None


class ExperimentStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_space: str
    config: Optional[str] = None
    iterations: int = Field(default=5, ge=1)
    seed: int = Field(default=0, ge=0)
    objective: str = Field(default="Sharpe", min_length=1)
    minimize: bool = False
    exploration_prob: float = Field(default=0.35, ge=0.0, le=1.0)
    auto_approve: bool = False
    approval_poll_interval: float = Field(default=5.0, gt=0.0)
    approval_timeout: Optional[float] = Field(default=None, gt=0.0)
    pipeline_output_dir: Optional[str] = None
    pipeline_log_level: Optional[str] = None
    pipeline_log_file: Optional[str] = None
    debug_prints: bool = False
    run_baselines: bool = False
    retrain_baselines: bool = False
    disable_align_cache: bool = False
    align_cache_dir: Optional[str] = None

    corr_gate_sharpe: float = Field(default=1.0)
    sharpe_close_epsilon: float = Field(default=0.05, ge=0.0)
    max_sharpe_sacrifice: float = Field(default=0.05, ge=0.0)
    min_corr_improvement: float = Field(default=0.05, ge=0.0)


class ExperimentProposalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str = Field(..., min_length=1, description="approved|rejected")
    decided_by: Optional[str] = None
    notes: Optional[str] = None
