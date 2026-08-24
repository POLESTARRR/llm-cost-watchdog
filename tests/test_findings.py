"""The findings are the page's headline, so they get the page's strictest tests.

The failure this file exists to prevent is a specific one with a history here:
a number rendered large and confidently that the database underneath does not
support. Every assertion below therefore checks a stated claim against the rows
it was derived from, rather than checking that the function returned something.
"""

import pytest

from src.findings import PORTFOLIO_SOURCES, compute_findings, counterfactual
from src.pricing import calculate_cost
from src.tracker import init_db, log_usage_many
from src.usage_schema import UsageEvent


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A small ledger whose stored costs are priced from its own token counts.

    Deliberately not hand-written dollar figures. An earlier version of this
    fixture set `cost_usd` by hand while the token counts implied something
    else, which is a state the real pipeline cannot produce (the tracker and the
    import endpoint both price from tokens) and which made the split percentages
    meaningless. Building the fixture the way production builds a row keeps the
    test measuring the module instead of the fixture.
    """
    monkeypatch.setenv("WATCHDOG_DB_PATH", str(tmp_path / "t.db"))
    init_db()

    def ev(scale=1, **kw):
        model = kw.pop("model", "claude-sonnet-5")
        toks = dict(
            input_tokens=100_000 * scale, output_tokens=1_000 * scale,
            cached_input_tokens=90_000 * scale, cache_write_tokens=5_000 * scale,
            cache_write_1h_tokens=5_000 * scale,
        )
        base = dict(
            model=model, provider="anthropic", project_tag="alpha",
            latency_ms=100.0, success=True, source="subscription",
            cost_usd=calculate_cost(
                model, toks["input_tokens"], toks["output_tokens"],
                toks["cached_input_tokens"], toks["cache_write_tokens"],
                toks["cache_write_1h_tokens"],
            ),
            **toks,
        )
        base.update(kw)
        return UsageEvent(**base)

    log_usage_many([
        ev(scale=3, project_tag="alpha"),          # priciest
        ev(scale=1, project_tag="alpha"),
        ev(scale=1, project_tag="beta", model="claude-opus-5"),
        # Gateway traffic. Must never reach the portfolio findings. Priced on an
        # expensive model on purpose: a leak-detection test whose leaked row is
        # cheaper than the rows it would contaminate cannot fail when it should.
        ev(scale=9, project_tag="gw", source="live", model="claude-opus-5"),
        # A failure. Costs nothing and is not a measured turn.
        ev(scale=1, project_tag="alpha", success=False, cost_usd=0.0),
    ])
    return tmp_path


def test_counts_only_successful_portfolio_turns(ledger):
    f = compute_findings().as_dict()
    # Three successful subscription rows. Not the live one, not the failure.
    assert f["headline"]["turns"] == 3
    assert f["headline"]["projects"] == 2
    # alpha ran 4 units of work (3 + 1), beta ran 1, so the total is the sum of
    # the per-project rows and nothing else leaked in.
    assert f["headline"]["cost_usd"] == pytest.approx(
        sum(p["cost_usd"] for p in f["by_project"]), abs=0.01
    )


def test_gateway_traffic_never_enters_the_finding(ledger):
    """The defect that reported '16 projects' by folding demo calls into a study."""
    f = compute_findings().as_dict()
    assert "gw" not in [p["project"] for p in f["by_project"]]
    # The live row is nine units of work against alpha+beta's five, so if it
    # leaked it would dominate rather than perturb.
    live_only = compute_findings(source="live").as_dict()
    assert live_only["headline"]["cost_usd"] > f["headline"]["cost_usd"]
    assert f["headline"]["turns"] == 3, "a live row leaked into the portfolio total"


def test_cost_split_accounts_for_every_dollar(ledger):
    """Reading, cache-TTL premium and output must sum to the whole bill.

    A split that leaves an unexplained remainder is how the 10.6% cache-write
    surcharge went unnamed on the real data for as long as it did.
    """
    rw = compute_findings().as_dict()["read_vs_write"]
    total = rw["read_percent"] + rw["cache_ttl_premium_percent"] + rw["write_percent"]
    assert total == pytest.approx(100.0, abs=0.6)


def test_read_dominates_a_cache_heavy_workload(ledger):
    rw = compute_findings().as_dict()["read_vs_write"]
    assert rw["ratio"] == 100  # 100k in / 1k out
    assert rw["read_percent"] > rw["write_percent"]


def test_caching_is_measured_against_the_uncached_price(ledger):
    c = compute_findings().as_dict()["caching"]
    assert c["uncached_cost_usd"] > c["saved_usd"] > 0
    assert 0 < c["percent_saved"] < 100
    assert c["hit_rate_percent"] == pytest.approx(90.0, abs=0.1)


def test_priciest_turn_is_the_priciest_turn(ledger):
    f = compute_findings().as_dict()
    p = f["read_vs_write"]["priciest_turn"]
    assert p["project"] == "alpha"       # the scale-3 row
    assert p["input_tokens"] == 300_000
    assert p["cost_usd"] < f["headline"]["cost_usd"], "one turn cannot exceed the total"


def test_projects_are_ranked_by_cost(ledger):
    rows = compute_findings().as_dict()["by_project"]
    assert [r["project"] for r in rows] == ["alpha", "beta"]
    assert rows[0]["turns"] == 2
    assert rows[0]["cost_usd"] > rows[1]["cost_usd"]
    assert rows[0]["cost_per_turn_usd"] == pytest.approx(rows[0]["cost_usd"] / 2, abs=0.01)


def test_counterfactual_prices_a_switch_and_refuses_to_call_it_a_saving(ledger):
    cf = counterfactual("claude-haiku-4-5")
    actual = compute_findings().as_dict()["headline"]["cost_usd"]
    assert cf["actual_usd"] == pytest.approx(actual, abs=0.01)
    # Haiku is the cheapest tier, so re-pricing this traffic on it must come out
    # below what Sonnet and Opus actually charged.
    assert 0 < cf["cost_usd"] < cf["actual_usd"]
    assert cf["delta_percent"] < 0
    # The word "saving" must not appear: this priced a substitution, it did not
    # judge one. See src/shadow.py for the measurement that would.
    assert "saving" not in cf["caveat"].lower()
    assert "does not judge" in cf["caveat"]


def test_empty_ledger_states_nothing(tmp_path, monkeypatch):
    """A fresh install must not render a study it never ran."""
    monkeypatch.setenv("WATCHDOG_DB_PATH", str(tmp_path / "empty.db"))
    init_db()
    f = compute_findings().as_dict()
    assert f["headline"]["turns"] == 0
    assert f["headline"]["projects"] == 0
    assert f["by_project"] == []
    assert counterfactual("claude-haiku-4-5")["cost_usd"] == 0.0


def test_portfolio_sources_excludes_live_and_demo():
    assert "live" not in PORTFOLIO_SOURCES
    assert "demo" not in PORTFOLIO_SOURCES
