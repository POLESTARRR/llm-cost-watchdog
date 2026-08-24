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


# --- the shipped example --------------------------------------------------


def test_snapshot_carries_no_prompt_text_or_paths(tmp_path, monkeypatch):
    """The snapshot is published, so it must reveal nothing about the work.

    Project names, counts and prices only. A snapshot that leaked a prompt or a
    file path would be a privacy failure shipped to a public URL, which is worse
    than having no example at all.
    """
    _write(tmp_path, "-Users-x-Desktop-secretproject", [_turn(), _turn(fresh=999_999)])
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)

    snap = ccost.snapshot(ccost.read_sessions())
    blob = json.dumps(snap)

    assert "secretproject" in blob            # the name is intentionally kept
    assert "/Users/" not in blob              # absolute paths are not
    assert "session.jsonl" not in blob        # nor filenames

    # Checked against the key names, not by substring: "context_now" contains
    # "text", so a naive scan of the serialised blob fails on its own data.
    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    leaky = {"prompt", "content", "message", "preview", "file", "path"}
    assert not (leaky & set(keys(snap)))


def test_snapshot_reports_growth_and_the_restart_flag(tmp_path, monkeypatch):
    _write(tmp_path, "-Users-x-Desktop-demo", [
        _turn(fresh=1000), _turn(fresh=1000, cached=500_000),
    ])
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path)

    snap = ccost.snapshot(ccost.read_sessions())
    assert snap["current"]["should_restart"] is True
    assert snap["current"]["growth"] > 100
    assert snap["generated_at"]


# --- other assistants -----------------------------------------------------


def _codex_turn(model="gpt-5.6-terra", inp=10_000, cached=8_000, written=0, out=200, reasoning=50):
    return {
        "timestamp": "2026-08-09T17:37:02.171Z",
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "last_token_usage": {
                    "input_tokens": inp,
                    "cached_input_tokens": cached,
                    "cache_write_input_tokens": written,
                    "output_tokens": out,
                    "reasoning_output_tokens": reasoning,
                },
                # Running total for the session. Summing THIS instead of
                # last_token_usage double-counts every earlier turn.
                "total_token_usage": {"input_tokens": 999_999, "output_tokens": 999_999},
            },
        },
    }


def test_codex_sessions_are_read_and_priced(tmp_path, monkeypatch):
    d = tmp_path / "2026" / "08" / "09"
    d.mkdir(parents=True)
    (d / "rollout-x.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "session_meta", "payload": {"cwd": "/Users/x/Desktop/myproj",
                                             "model": "gpt-5.6-terra"}},
        _codex_turn(),
        _codex_turn(inp=20_000, cached=18_000),
    ]) + "\n")
    monkeypatch.setattr(ccost, "CODEX_SESSIONS", tmp_path)

    got = ccost.read_codex_sessions()
    assert len(got) == 1
    s = got[0]
    assert s["tool"] == "codex"
    assert s["project"] == "myproj"
    assert s["priced"] is True
    assert len(s["turns"]) == 2
    assert s["cost"] > 0


def test_codex_uses_the_per_turn_count_not_the_running_total(tmp_path, monkeypatch):
    """total_token_usage accumulates; summing it overstates a long session wildly."""
    d = tmp_path / "2026" / "08" / "09"
    d.mkdir(parents=True)
    (d / "r.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "session_meta", "payload": {"cwd": "/Users/x/p", "model": "gpt-5.6-terra"}},
        _codex_turn(inp=10_000, cached=0),
    ]) + "\n")
    monkeypatch.setattr(ccost, "CODEX_SESSIONS", tmp_path)
    assert ccost.read_codex_sessions()[0]["turns"] == [10_000]   # not 999,999


def test_copilot_sessions_are_counted_but_never_priced(tmp_path, monkeypatch):
    """Copilot publishes no token counts. Inventing one would defeat the tool."""
    import sqlite3

    db = tmp_path / "session-store.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT);
        CREATE TABLE turns (id INTEGER PRIMARY KEY, session_id TEXT,
                            user_message TEXT, assistant_response TEXT, timestamp TEXT);
        INSERT INTO sessions VALUES ('s1', '/Users/x/Desktop/thing');
        INSERT INTO turns (session_id, user_message, timestamp)
            VALUES ('s1', 'hi', '2026-08-20T10:00:00Z'), ('s1', 'again', '2026-08-20T10:05:00Z');
    """)
    conn.commit()
    conn.close()
    monkeypatch.setattr(ccost, "COPILOT_DB", db)

    got = ccost.read_copilot_sessions()
    assert len(got) == 1
    assert got[0]["tool"] == "copilot"
    assert got[0]["priced"] is False
    assert got[0]["cost"] == 0.0
    assert got[0]["message_count"] == 2
    assert got[0]["turns"] == []     # no context figures exist to report


def test_missing_tools_are_simply_absent(tmp_path, monkeypatch):
    """Not having Codex or Copilot installed is normal, not an error."""
    monkeypatch.setattr(ccost, "CODEX_SESSIONS", tmp_path / "nope")
    monkeypatch.setattr(ccost, "COPILOT_DB", tmp_path / "nope.db")
    assert ccost.read_codex_sessions() == []
    assert ccost.read_copilot_sessions() == []


def test_now_refuses_to_advise_on_an_unpriced_session(tmp_path, monkeypatch, capsys):
    """Copilot has no context size, so it can never be the subject of the advice."""
    monkeypatch.setattr(ccost, "TRANSCRIPTS", tmp_path / "none")
    ccost.cmd_now([{"tool": "copilot", "project": "x", "turns": [], "cost": 0.0,
                    "priced": False, "message_count": 3, "mtime": 0}])
    assert "No session with token counts" in capsys.readouterr().out


# --- the shareable report -------------------------------------------------


def _sessions_for_report():
    return [
        {"tool": "claude-code", "project": "alpha", "turns": [1000, 500_000],
         "cost": 12.5, "priced": True, "mtime": 200},
        {"tool": "codex", "project": "beta", "turns": [2000, 3000],
         "cost": 1.25, "priced": True, "mtime": 100},
        {"tool": "copilot", "project": "gamma", "turns": [], "cost": 0.0,
         "priced": False, "message_count": 7, "mtime": 50},
    ]


def test_report_is_self_contained(tmp_path):
    """It must open with the network off, years from now, wherever it lands."""
    import re

    html = ccost.build_report(_sessions_for_report())
    assert "<style>" in html                      # css inlined
    assert not re.findall(r'(?:src|href)="https?://', html), "report reaches out to the network"
    assert "<script" not in html                  # nothing to execute, nothing to break


def test_report_shows_every_project_and_marks_the_assistant(tmp_path):
    html = ccost.build_report(_sessions_for_report())
    assert "alpha" in html and "beta" in html
    assert "claude-code" in html and "codex" in html


def test_report_counts_copilot_without_pricing_it(tmp_path):
    html = ccost.build_report(_sessions_for_report())
    assert "7 messages" in html
    assert "never priced" in html or "not priced" in html


def test_report_carries_the_restart_advice_when_the_session_has_grown(tmp_path):
    html = ccost.build_report(_sessions_for_report())
    assert "Start a new session" in html


def test_report_never_calls_list_price_a_bill(tmp_path):
    """Same honesty rule the site is held to, in a file that outlives the site."""
    html = ccost.build_report(_sessions_for_report())
    assert "not a bill" in html


def test_report_handles_a_machine_with_only_copilot(tmp_path):
    """No priced session means no context figures. It must still produce a page."""
    only = [s for s in _sessions_for_report() if not s["priced"]]
    html = ccost.build_report(only)
    assert "<title>" in html
    assert "Start a new session" not in html
