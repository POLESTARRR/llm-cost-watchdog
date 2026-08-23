"""Grading shadow pairs. The deterministic checks must abstain, not guess."""

import pytest

from src.judge import ACCEPTABLE, INADEQUATE, deterministic_verdict, llm_verdict


class TestDeterministic:
    def test_empty_cheap_answer_is_inadequate(self):
        assert deterministic_verdict("a real answer", "")[0] == INADEQUATE

    def test_broken_python_loses_to_working_python(self):
        real = "```python\ndef f():\n    return 1\n```"
        shadow = "```python\ndef f(\n    return 1\n```"
        verdict, reason = deterministic_verdict(real, shadow)
        assert verdict == INADEQUATE
        assert "Python" in reason

    def test_both_broken_is_not_a_difference_between_the_models(self):
        """If the expensive model also emitted garbage, that is not evidence
        about the cheap one."""
        broken = "```python\ndef f(\n```"
        assert deterministic_verdict(broken, broken) is None

    def test_broken_json_loses(self):
        assert deterministic_verdict('```json\n{"a":1}\n```', '```json\n{a:1\n```')[0] == INADEQUATE

    def test_abstains_on_ordinary_prose(self):
        """The common case. Inventing a check here would be worse than deferring."""
        assert deterministic_verdict("Paris is the capital.", "The capital is Paris.") is None

    def test_a_drastically_shorter_answer_loses(self):
        assert deterministic_verdict("x" * 1000, "short")[0] == INADEQUATE

    def test_a_terse_correct_answer_is_not_punished(self):
        """Efficiency is not a defect; the threshold has a floor for this."""
        assert deterministic_verdict("The answer is 4.", "4") is None

    def test_valid_code_in_both_abstains(self):
        good = "```python\nx = 1\n```"
        assert deterministic_verdict(good, good) is None


class TestLLMJudge:
    @pytest.fixture
    def reply(self, monkeypatch):
        holder = {}

        def _set(text):
            holder["prompt_seen"] = None

            def _fake(prompt, **kw):
                holder["prompt_seen"] = prompt
                return text

            monkeypatch.setattr("src.utils.call_llm", _fake)
            return holder

        return _set

    def test_equivalent_means_acceptable(self, reply):
        reply("EQUIVALENT")
        assert llm_verdict("q", "real", "shadow")[0] == ACCEPTABLE

    def test_worse_means_inadequate(self, reply):
        reply("WORSE")
        assert llm_verdict("q", "real", "shadow")[0] == INADEQUATE

    def test_an_unclear_verdict_does_not_count_as_a_pass(self, reply):
        """Defaulting to acceptable would inflate the rate every time it rambled."""
        reply("Well, it depends on what you mean by useful...")
        verdict, reason = llm_verdict("q", "real", "shadow")
        assert verdict == INADEQUATE
        assert "no clear verdict" in reason

    def test_the_judge_is_blind_to_which_model_wrote_what(self, reply):
        holder = reply("EQUIVALENT")
        llm_verdict("the question", "EXPENSIVE_TEXT", "CHEAP_TEXT")
        seen = holder["prompt_seen"]
        assert "EXPENSIVE_TEXT" in seen and "CHEAP_TEXT" in seen
        # No label anywhere identifies the source of either answer.
        for leak in ("expensive", "cheap", "local", "frontier", "opus", "gpt"):
            assert leak not in seen.lower().replace("expensive_text", "").replace("cheap_text", "")

    def test_presentation_order_is_randomised(self, reply, monkeypatch):
        """Position bias is the best-documented LLM-judge failure mode."""
        holder = reply("EQUIVALENT")
        monkeypatch.setattr("src.judge.random.random", lambda: 0.1)
        llm_verdict("q", "REAL", "SHADOW")
        first = holder["prompt_seen"].index("SHADOW") < holder["prompt_seen"].index("REAL")

        monkeypatch.setattr("src.judge.random.random", lambda: 0.9)
        llm_verdict("q", "REAL", "SHADOW")
        second = holder["prompt_seen"].index("SHADOW") < holder["prompt_seen"].index("REAL")
        assert first != second


class TestGradePending:
    def test_deterministic_wins_without_calling_a_model(self, temp_db, monkeypatch):
        from src import shadow
        from src.judge import grade_pending
        from src.providers.base import LLMResponse

        class Stub:
            def complete(self, prompt, model, temperature):
                return LLMResponse(text="", input_tokens=1, output_tokens=1)

        monkeypatch.setattr("src.providers.get_provider", lambda m: Stub())
        shadow.run_shadow(prompt="write code", real_model="claude-opus-5",
                          real_response="```python\nx=1\n```", real_cost_usd=0.05,
                          real_latency_ms=100.0, db_path=temp_db)

        def _boom(*a, **k):
            raise AssertionError("the LLM judge should not have been called")

        monkeypatch.setattr("src.judge.llm_verdict", _boom)
        out = grade_pending(db_path=temp_db)
        assert out["graded"] == 1
        assert out["results"][0]["scored_by"] == "deterministic"
        assert out["results"][0]["verdict"] == INADEQUATE
