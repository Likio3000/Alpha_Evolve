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
