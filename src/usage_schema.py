"""Pydantic models shared across the tracker, analyzer, digest, and MCP server."""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

# Where a row came from. A cost tracker whose numbers can't be traced back to a
# real API call is worse than no tracker — it reports confident fiction. Every
# row carries its provenance so "total spend" can always be narrowed to money
# that was actually charged.
#   live         - a real HTTP request through call_llm(); metered and billed per token
#   demo         - seeded sample data, for populating an empty dashboard
#   manual       - hand-entered via log_manual_entry / the CLI; real spend, but unverified
#   subscription - real tokens, real usage, but consumed under a flat-fee plan
#                  (Claude Pro/Max) rather than metered per-token billing
#
# `subscription` exists because conflating it with `manual` was a lie this
# project told about its own headline number. Claude Code build usage on a Pro
# plan is real work at real token counts, but **no per-token charge occurred**
# — the cost figure is list-price-equivalent *value*, not money that left an
# account. Reporting it as billed spend would be exactly the "confident
# fiction" this provenance system exists to prevent. See tracker.BILLED_SOURCES.
Source = Literal["live", "demo", "manual", "subscription"]

# Provider service tiers. `batch` bills at 50%; see pricing.SERVICE_TIER_MULTIPLIERS.
ServiceTier = Literal["standard", "batch", "priority"]


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
    # Subset of cache_write_tokens written with a 1-hour TTL, which bills at
    # 2.0x input instead of the 5-minute 1.25x. Tracked separately because the
    # difference is 14% of this repo's own real build cost.
    cache_write_1h_tokens: int = 0
    cost_usd: float
    latency_ms: float
    prompt_preview: str = ""
    # SHA-256 of the full prompt. The 80-char `prompt_preview` cannot tell two
    # calls apart when they share a long fixed instruction template — a real
    # false-positive this project hit on civil-prep's duplicate detection. The
    # hash identifies a repeated prompt exactly without ever storing one.
    prompt_hash: str | None = None
    service_tier: ServiceTier = "standard"
    success: bool = True
    error: str | None = None
    # Defaults to "live" because the wrapper is the overwhelmingly common
    # writer, and a row that lies about being real is the dangerous direction
    # of that default — seeders and manual entry both set this explicitly.
    source: Source = "live"

    @classmethod
    def make_preview(cls, prompt: str, limit: int = 80) -> str:
        """Truncate a prompt to its first `limit` characters. Never log full prompts."""
        return prompt[:limit]

    @classmethod
    def make_hash(cls, prompt: str) -> str | None:
        """SHA-256 of the full prompt, for exact duplicate detection.

        A hash rather than the text itself: it identifies a repeat exactly
        while storing nothing recoverable, so duplicate detection gets stronger
        without weakening the never-log-full-prompts rule.
        """
        if not prompt:
            return None
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


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
    # Provenance of the rows behind these totals. `source_filter` is None when
    # the report spans everything; `breakdown_by_source` lets any consumer see
    # at a glance how much of the headline number was actually billed.
    source_filter: str | None = None
    breakdown_by_source: dict[str, float] = Field(default_factory=dict)
    calls_by_source: dict[str, int] = Field(default_factory=dict)
