"""Pydantic models shared across the tracker, analyzer, digest, and MCP server."""

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class UsageEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=_now_iso)
    model: str
    provider: str = "unknown"
    project_tag: str = "default"
    input_tokens: int
    output_tokens: int
    # Disjoint subsets of input_tokens: served from cache (cheap) and newly
    # written to cache (a premium on some models).
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float
    latency_ms: float
    prompt_preview: str = ""
    success: bool = True
    error: str | None = None

    @classmethod
    def make_preview(cls, prompt: str, limit: int = 80) -> str:
        """Truncate a prompt to its first `limit` characters. Never log full prompts."""
        return prompt[:limit]


class BudgetConfig(BaseModel):
    period: Literal["daily", "weekly", "monthly"]
    limit_usd: float


class AnomalyFlag(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    usage_event_id: str
    reason: str
    severity: Literal["low", "medium", "high"]
    detected_at: str = Field(default_factory=_now_iso)


class CostReport(BaseModel):
    period: str
    total_cost_usd: float
    total_calls: int
    breakdown_by_model: dict[str, float]
    breakdown_by_project: dict[str, float]
    breakdown_by_provider: dict[str, float] = Field(default_factory=dict)
    # What prompt caching saved vs. pricing every input token at the full rate.
    cache_savings_usd: float = 0.0
    failed_calls: int = 0
    anomalies: list[AnomalyFlag] = Field(default_factory=list)
