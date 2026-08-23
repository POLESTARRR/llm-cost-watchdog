"""Shadow comparison: collect first, judge second, and never break a real call.

The behaviours worth pinning are the safety ones. A quality experiment that can
fail a production request, or that quietly reports unscored data as proven
savings, is worse than having no experiment.
"""

import pytest

from src import shadow


@pytest.fixture
def db(temp_db, monkeypatch):
    monkeypatch.setattr(shadow, "SHADOW_RATE", 1.0)
    return temp_db


def _run(db, prompt="reformat this JSON", **kw):
    return shadow.run_shadow(
        prompt=prompt, real_model="claude-opus-5", real_response="the real answer",
        real_cost_usd=0.05, real_latency_ms=1200.0, db_path=db, **kw
    )


@pytest.fixture
def fake_local(monkeypatch):
    """A local model that answers instantly, so tests never touch Ollama."""
    from src.providers.base import LLMResponse

    class Stub:
        def complete(self, prompt, model, temperature):
            return LLMResponse(text="the cheap answer", input_tokens=10, output_tokens=5)

    monkeypatch.setattr("src.providers.get_provider", lambda m: Stub())
    return Stub


class TestSampling:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.setattr(shadow, "SHADOW_RATE", 0.0)
        assert shadow.enabled() is False
        assert shadow.should_shadow("anything") is False

    def test_rate_one_always_samples(self, monkeypatch):
        monkeypatch.setattr(shadow, "SHADOW_RATE", 1.0)
        assert all(shadow.should_shadow("hi") for _ in range(20))

    def test_oversized_prompts_are_skipped(self, monkeypatch):
        """A 3B model on a 30k-token context costs minutes and teaches nothing."""
        monkeypatch.setattr(shadow, "SHADOW_RATE", 1.0)
        assert shadow.should_shadow("x" * (shadow.MAX_SHADOW_PROMPT_CHARS + 1)) is False


class TestSafety:
    def test_a_failing_shadow_never_raises(self, db, monkeypatch):
        """The property the whole design rests on: the real call already
        succeeded, so nothing here may turn that into a failure."""
        def boom(model):
            raise RuntimeError("ollama is on fire")

        monkeypatch.setattr("src.providers.get_provider", boom)
        result = _run(db)
        assert result is not None
        assert "ollama is on fire" in result.shadow_error

    def test_a_failed_shadow_is_still_recorded(self, db, monkeypatch):
        monkeypatch.setattr(
            "src.providers.get_provider",
            lambda m: (_ for _ in ()).throw(RuntimeError("down")),
        )
        _run(db)
        summary = shadow.shadow_summary(db_path=db)
        assert summary["total_comparisons"] == 1
        assert sum(t["shadow_failures"] for t in summary["by_tier"].values()) == 1


class TestCollection:
    def test_stores_both_answers_and_the_classification(self, db, fake_local):
        _run(db, prompt="reformat this JSON")
        rows = shadow.pending_review(db_path=db)
        assert len(rows) == 1
        row = rows[0]
        assert row["real_response"] == "the real answer"
        assert row["shadow_response"] == "the cheap answer"
        assert row["complexity_tier"] == "trivial"
        # Prompt text is deliberately retained here and nowhere else.
        assert row["prompt"] == "reformat this JSON"

    def test_local_shadow_costs_nothing(self, db, fake_local):
        _run(db)
        assert shadow.pending_review(db_path=db)[0]["shadow_cost_usd"] == 0.0

    def test_prompts_never_leak_into_the_usage_ledger(self, db, fake_local):
        """The ledger's never-store-prompts rule still holds; shadows are a
        separate, opt-in table."""
        from src.tracker import _connect

        _run(db, prompt="a very secret prompt")
        with _connect(db) as conn:
            rows = conn.execute("SELECT prompt_preview FROM usage_events").fetchall()
        assert all("a very secret prompt" not in (r["prompt_preview"] or "") for r in rows)


class TestScoring:
    def test_unscored_data_is_not_reported_as_proven_savings(self, db, fake_local):
        """The claim this project must not make: 'we saved $X' from data nobody
        has graded."""
        _run(db)
        s = shadow.shadow_summary(db_path=db)
        assert s["unverified_savings_usd"] == 0.05
        assert "NOT a saving" in s["note"]
        tier = s["by_tier"]["trivial"]
        assert tier["scored"] == 0
        assert tier["acceptance_rate"] is None

    def test_recording_a_verdict_produces_an_acceptance_rate(self, db, fake_local):
        _run(db)
        row_id = shadow.pending_review(db_path=db)[0]["id"]
        shadow.record_verdict(row_id, "acceptable", db_path=db)
        tier = shadow.shadow_summary(db_path=db)["by_tier"]["trivial"]
        assert tier["scored"] == 1
        assert tier["acceptance_rate"] == 1.0

    def test_rejects_an_unknown_verdict(self, db):
        with pytest.raises(ValueError):
            shadow.record_verdict("any-id", "pretty good", db_path=db)

    def test_scored_rows_leave_the_review_queue(self, db, fake_local):
        _run(db)
        row_id = shadow.pending_review(db_path=db)[0]["id"]
        shadow.record_verdict(row_id, "inadequate", db_path=db)
        assert shadow.pending_review(db_path=db) == []


def test_purge_removes_stored_prompt_text(db, fake_local):
    _run(db)
    assert shadow.purge_shadows(db_path=db) == 1
    assert shadow.shadow_summary(db_path=db)["total_comparisons"] == 0


def test_a_shadow_against_the_same_model_is_skipped(db, fake_local):
    """Comparing a model with itself is not evidence about cheap-vs-expensive.

    Regression from the first real run: the primary call had failed over to the
    local model, and the shadow then re-ran that same local model. Two rows of
    pure noise landed in the acceptance rate.
    """
    result = shadow.run_shadow(
        prompt="anything", real_model="ollama/llama3.2:3b",
        real_response="x", real_cost_usd=0.0, real_latency_ms=1.0,
        shadow_model="ollama/llama3.2:3b", db_path=db,
    )
    assert result is None
    assert shadow.shadow_summary(db_path=db)["total_comparisons"] == 0
