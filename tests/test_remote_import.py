"""Remote push behaviour: batching, retry, and incremental checkpointing.

Nothing here hits the network — every request goes through an
httpx.MockTransport wired into the module's httpx.Client. This is the exact
path that failed in production: a real push to Render, mid-transcript, took
long enough that later batches timed out after earlier ones had already
landed durably on the server. The bug that produced was checkpointing only
once at the end of a whole file, so a resume after partial failure resent
turns the server already had, and /import mints a fresh UUID per row with no
dedup, so that duplicated real data on a live deployment.
"""

import json

import httpx
import pytest

import scripts.import_claude_code_usage as importer
from scripts.import_claude_code_usage import _checkpoint_path, _sink_slug, load

REMOTE = "https://example.com"
SINK = _sink_slug(REMOTE)  # derived, not guessed: "remote-https-example-com"


def _transcript(tmp_path, n_turns, start_hour=10):
    """A transcript of n_turns assistant turns, each preceded by a user turn."""
    path = tmp_path / "session.jsonl"
    lines = []
    for i in range(n_turns):
        minute = f"{i:02d}"
        lines.append(json.dumps({
            "type": "user", "timestamp": f"2026-08-01T{start_hour}:{minute}:00.000Z",
            "message": {"content": [{"type": "text", "text": f"prompt {i}"}]},
        }))
        lines.append(json.dumps({
            "type": "assistant", "timestamp": f"2026-08-01T{start_hour}:{minute}:05.000Z",
            "message": {
                "id": f"msg_{i}", "model": "claude-opus-5",
                "usage": {"input_tokens": 10, "output_tokens": 10},
            },
        }))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class _CountingHandler:
    """Records every request's event count; optionally simulates the host
    becoming persistently unavailable from the Nth call onward, so all 3
    retries on that batch (and everything after it) also fail. A transient,
    self-recovering blip is a different test, see test_timeout_is_retried_*.
    """

    def __init__(self, fail_from_call: int | None = None):
        self.fail_from_call = fail_from_call
        self.calls = 0
        self.total_events_received = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.fail_from_call and self.calls >= self.fail_from_call:
            raise httpx.ReadTimeout("simulated persistently slow host", request=request)
        body = json.loads(request.content)
        n = len(body["events"])
        self.total_events_received += n
        return httpx.Response(200, json={"logged": n, "skipped_unpriced": 0, "total_cost_usd": 0.0})


@pytest.fixture
def small_batches(monkeypatch):
    """Force multiple small batches so a 6-turn transcript spans 3 requests,
    and skip the real backoff sleep between retries so the retry/exhaustion
    tests do not spend real wall-clock time waiting on a fake network."""
    monkeypatch.setattr(importer, "_REMOTE_BATCH_SIZE", 2)
    monkeypatch.setattr(importer, "_REMOTE_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(importer.time, "sleep", lambda _seconds: None)


@pytest.fixture
def mock_client(monkeypatch):
    """Wire scripts.import_claude_code_usage.httpx.Client to a MockTransport,
    so _push_remote's real batching/retry code runs against a fake server."""
    handler_holder = {}

    class _Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler_holder["fn"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(importer.httpx, "Client", _Client)

    def _install(handler):
        handler_holder["fn"] = handler

    return _install


def test_batches_the_whole_file(tmp_path, small_batches, mock_client):
    handler = _CountingHandler()
    mock_client(handler)

    path = _transcript(tmp_path, 6)
    stats = load(path, "proj", remote_url=REMOTE, import_key="k")

    assert stats["logged"] == 6
    assert handler.calls == 3  # 6 turns / batch size 2
    assert handler.total_events_received == 6


def test_checkpoint_persists_incrementally_not_only_at_the_end(tmp_path, small_batches, mock_client):
    """The bug this prevents: checkpointing once per file, not once per batch,
    discarded already-landed progress on a mid-file failure."""
    # 6 turns / batch size 2 = 3 batches. Failing from call 3 lets batches 1-2
    # (calls 1-2) succeed, then batch 3 exhausts all 3 of its retries (calls
    # 3-5) and the whole load() call raises.
    handler = _CountingHandler(fail_from_call=3)
    mock_client(handler)

    path = _transcript(tmp_path, 6)
    with pytest.raises(httpx.ReadTimeout):
        load(path, "proj", remote_url=REMOTE, import_key="k")

    checkpoint = _checkpoint_path(path, "proj", sink=SINK)
    assert checkpoint.exists()
    saved = json.loads(checkpoint.read_text())
    assert saved == ["msg_0", "msg_1", "msg_2", "msg_3"]  # batches 1-2, not batch 3


def test_resume_after_partial_failure_does_not_resend_landed_rows(tmp_path, small_batches, mock_client):
    """Full end-to-end: fail partway, then re-run with a working handler and
    confirm the retry only sends what did not already land."""
    failing = _CountingHandler(fail_from_call=3)
    mock_client(failing)
    path = _transcript(tmp_path, 6)
    with pytest.raises(httpx.ReadTimeout):
        load(path, "proj", remote_url=REMOTE, import_key="k")

    resuming = _CountingHandler()
    mock_client(resuming)
    stats = load(path, "proj", remote_url=REMOTE, import_key="k")

    assert stats["logged"] == 2  # only msg_4 and msg_5 were still pending
    assert resuming.total_events_received == 2
    assert stats["already_imported"] == 4


def test_timeout_is_retried_before_raising(tmp_path, small_batches, mock_client):
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("slow", request=request)
        body = json.loads(request.content)
        return httpx.Response(200, json={"logged": len(body["events"]), "skipped_unpriced": 0, "total_cost_usd": 0.0})

    mock_client(flaky)
    path = _transcript(tmp_path, 2)  # one batch, so all retries hit the same batch
    stats = load(path, "proj", remote_url=REMOTE, import_key="k")

    assert stats["logged"] == 2
    assert calls["n"] == 3  # two failures, then success


def test_exhausting_retries_raises_the_last_error(tmp_path, small_batches, mock_client):
    def always_slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    mock_client(always_slow)
    path = _transcript(tmp_path, 2)
    with pytest.raises(httpx.ReadTimeout):
        load(path, "proj", remote_url=REMOTE, import_key="k")


# --- checkpoint namespacing ------------------------------------------------


def test_local_and_remote_checkpoints_do_not_collide(tmp_path, small_batches, mock_client):
    """Local and remote are different destinations with independent history;
    a shared checkpoint would let one mark the other's turns as already done."""
    handler = _CountingHandler()
    mock_client(handler)

    path = _transcript(tmp_path, 2)
    local_stats = load(path, "proj", db_path=str(tmp_path / "local.db"))
    remote_stats = load(path, "proj", remote_url=REMOTE, import_key="k")

    assert local_stats["logged"] == 2
    assert remote_stats["logged"] == 2  # not treated as already-imported

    local_cp = _checkpoint_path(path, "proj", sink="local")
    remote_cp = _checkpoint_path(path, "proj", sink=SINK)
    assert local_cp != remote_cp
    assert local_cp.exists() and remote_cp.exists()


def test_local_checkpoint_filename_is_unchanged_by_the_sink_parameter(tmp_path):
    """Existing checkpoints on disk from before remote push existed must
    still resolve, or a local re-run would re-import everything."""
    p = _checkpoint_path(str(tmp_path / "x.jsonl"), "proj")
    assert p.name == ".x.proj.imported.json"
    assert p == _checkpoint_path(str(tmp_path / "x.jsonl"), "proj", sink="local")


def test_two_different_remote_urls_get_distinct_checkpoints(tmp_path, small_batches, mock_client):
    handler = _CountingHandler()
    mock_client(handler)
    path = _transcript(tmp_path, 2)

    load(path, "proj", remote_url="https://staging.example.com", import_key="k")
    stats = load(path, "proj", remote_url="https://prod.example.com", import_key="k")

    assert stats["logged"] == 2  # a different remote has not seen these turns
