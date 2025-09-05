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

