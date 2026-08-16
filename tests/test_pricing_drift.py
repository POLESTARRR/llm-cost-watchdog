"""Pricing drift: reconciling the local table against a public price map.

The tool must never rewrite a rate, and must never fail a caller when the
network is unavailable, a reporting nicety that can break cost calculation
would be a bad trade.
"""

import json

import pytest

from src.pricing import PRICING_TABLE
from src.pricing_drift import check_drift, context_windows, load_price_map


@pytest.fixture
def fake_map(tmp_path, monkeypatch):
    """Point the module at a temp cache file with a known map."""
    path = tmp_path / "map.json"
    monkeypatch.setattr("src.pricing_drift.CACHE_PATH", path)
    return path


def _write(path, payload):
    path.write_text(json.dumps(payload))


def test_agreeing_rates_report_no_drift(fake_map, monkeypatch):
    local = PRICING_TABLE["claude-haiku-4-5"]
    _write(fake_map, {
        "claude-haiku-4-5": {
            "input_cost_per_token": local["input"] / 1000,
            "cache_read_input_token_cost": local["cached_input"] / 1000,
            "output_cost_per_token": local["output"] / 1000,
        }
    })
    monkeypatch.setattr("src.pricing_drift.MODEL_ALIASES", {"claude-haiku-4-5": "claude-haiku-4-5"})

    report = check_drift()
    assert report["checked"] is True
    assert report["drifted"] == []
    assert "claude-haiku-4-5" in report["agreed"]


def test_disagreeing_rate_is_reported_with_both_numbers(fake_map, monkeypatch):
    _write(fake_map, {"claude-haiku-4-5": {"input_cost_per_token": 0.000002}})  # 0.002/1k
    monkeypatch.setattr("src.pricing_drift.MODEL_ALIASES", {"claude-haiku-4-5": "claude-haiku-4-5"})

    report = check_drift()
    assert len(report["drifted"]) == 1
    fields = report["drifted"][0]["fields"]
    assert fields["input"]["ours"] == PRICING_TABLE["claude-haiku-4-5"]["input"]
    assert fields["input"]["theirs"] == pytest.approx(0.002)


def test_drift_check_never_mutates_the_local_table(fake_map, monkeypatch):
    before = dict(PRICING_TABLE["claude-haiku-4-5"])
    _write(fake_map, {"claude-haiku-4-5": {"input_cost_per_token": 0.999}})
    monkeypatch.setattr("src.pricing_drift.MODEL_ALIASES", {"claude-haiku-4-5": "claude-haiku-4-5"})

    check_drift()
    assert PRICING_TABLE["claude-haiku-4-5"] == before


def test_models_absent_from_the_public_map_are_unmapped_not_drifted(fake_map, monkeypatch):
    _write(fake_map, {})
    monkeypatch.setattr("src.pricing_drift.MODEL_ALIASES", {"claude-haiku-4-5": "nope"})

    report = check_drift()
    assert report["drifted"] == []
    assert "claude-haiku-4-5" in report["unmapped"]


def test_missing_map_reports_it_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pricing_drift.CACHE_PATH", tmp_path / "absent.json")
    monkeypatch.setattr("src.pricing_drift.load_price_map", lambda refresh=False: None)

    report = check_drift()
    assert report["checked"] is False
    assert "unavailable" in report["reason"]


def test_corrupt_cache_is_treated_as_unavailable(fake_map):
    fake_map.write_text("{not json")
    assert load_price_map() is None


def test_context_windows_reads_only_positive_integers(fake_map, monkeypatch):
    _write(fake_map, {
        "a": {"max_input_tokens": 200_000},
        "b": {"max_input_tokens": None},
        "c": {},
    })
    monkeypatch.setattr(
        "src.pricing_drift.MODEL_ALIASES",
        {"model-a": "a", "model-b": "b", "model-c": "c"},
    )
    assert context_windows() == {"model-a": 200_000}
