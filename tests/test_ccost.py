"""Tests for the ccost CLI.

The tool reads files it does not own, written by a program that changes, so the
things worth pinning are the ones that break silently: a transcript shape it
cannot parse, a model it cannot price, and a directory name it turns into
nonsense. Each of those fails by printing something plausible and wrong, which
is the worst failure mode a cost tool has.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ccost  # noqa: E402


# --- project naming -------------------------------------------------------


@pytest.mark.parametrize("encoded,expected", [
    ("-Users-dhruvsharma-Desktop-RES-PROJECTS-P2-JSW", "P2-JSW"),
    ("-Users-dhruvsharma-Desktop-FEYMAN", "FEYMAN"),
    ("-Users-someone-Desktop-Mum-Rag-assistant", "Mum-Rag-assistant"),
    ("-Users-x-Downloads-uigen", "uigen"),
    ("-Users-dhruvsharma", "~"),
])
def test_project_name_keeps_the_identifying_part(encoded, expected):
    assert ccost.project_name(encoded) == expected


def test_project_name_drops_the_username_positionally():
    """Must work on a machine that is not this one."""
    assert ccost.project_name("-Users-alice-Desktop-thing") == "thing"
    assert ccost.project_name("-Users-bob-Desktop-thing") == "thing"


# --- reading transcripts --------------------------------------------------


def _write(tmp_path, name, records):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "session.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return f


def _turn(model="claude-sonnet-5", fresh=100, cached=0, written=0, out=50):
    return {
        "type": "assistant",
        "timestamp": "2026-08-20T10:00:00Z",
        "message": {
            "model": model,
            "usage": {
                "input_tokens": fresh,
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": written,
                "output_tokens": out,
            },
        },
    }


def test_reads_and_prices_a_session(tmp_path, monkeypatch):
    _write(tmp_path, "-Users-x-Desktop-demo", [
        _turn(fresh=1000, out=100),
        _turn(fresh=500, cached=5000, out=200),
    ])
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)

    sessions = ccost.read_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s["project"] == "demo"
    assert len(s["turns"]) == 2
    assert s["turns"] == [1000, 5500]   # context per turn, cache included
    assert s["cost"] > 0


def test_non_assistant_records_are_ignored(tmp_path, monkeypatch):
    """A transcript is mostly user turns and tool results, which have no usage."""
    _write(tmp_path, "-Users-x-Desktop-demo", [
        {"type": "user", "message": {"content": "hello"}},
        {"type": "summary"},
        _turn(),
        {"type": "assistant", "message": {"model": "claude-sonnet-5"}},  # no usage
    ])
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)
    assert len(ccost.read_sessions()[0]["turns"]) == 1


def test_a_corrupt_line_does_not_lose_the_session(tmp_path, monkeypatch):
    """Transcripts are appended live and the last line can be half-written."""
    f = _write(tmp_path, "-Users-x-Desktop-demo", [_turn(), _turn()])
    f.write_text(f.read_text() + '{"type": "assis')
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)
    assert len(ccost.read_sessions()[0]["turns"]) == 2


def test_an_unpriceable_model_is_skipped_rather_than_guessed(tmp_path, monkeypatch):
    """Inventing a rate is how a cost tool starts lying."""
    _write(tmp_path, "-Users-x-Desktop-demo", [
        _turn(model="claude-sonnet-5"),
        _turn(model="some-model-nobody-has-priced"),
    ])
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)
    assert len(ccost.read_sessions()[0]["turns"]) == 1


def test_dated_model_ids_still_price(tmp_path, monkeypatch):
    """Transcripts carry claude-sonnet-5-20260114; the price table carries the family."""
    _write(tmp_path, "-Users-x-Desktop-demo", [_turn(model="claude-sonnet-5-20260114")])
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)
    s = ccost.read_sessions()
    assert len(s) == 1 and s[0]["cost"] > 0


# --- the advice -----------------------------------------------------------


def test_restart_advice_appears_only_once_context_is_large(tmp_path, monkeypatch, capsys):
    small = [_turn(fresh=1000) for _ in range(5)]
    _write(tmp_path, "-Users-x-Desktop-small", small)
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)
    ccost.cmd_now(ccost.read_sessions())
    assert "Consider starting a new session" not in capsys.readouterr().out


def test_restart_advice_fires_on_a_grown_session(tmp_path, monkeypatch, capsys):
    grown = [_turn(fresh=1000), _turn(fresh=10_000, cached=500_000)]
    _write(tmp_path, "-Users-x-Desktop-grown", grown)
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)
    ccost.cmd_now(ccost.read_sessions())
    out = capsys.readouterr().out
    assert "Consider starting a new session" in out
    assert "510K" in out          # context reported, not just a warning


def test_human_reads_billions():
    assert ccost.human(2_259_700_000) == "2.26B"
    assert ccost.human(1_500_000) == "1.5M"
    assert ccost.human(2_500) == "2K"
    assert ccost.human(42) == "42"


def test_empty_directory_is_not_a_crash(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)
    assert ccost.read_sessions() == []
    ccost.cmd_now([])
    assert "No Claude Code sessions" in capsys.readouterr().out
