"""Ranking cost levers by size, and refusing to conflate realised with possible."""

import pytest

from src.levers import analyse_levers


@pytest.fixture
def report(sample_db):
    return analyse_levers(period="all_time")


def test_levers_are_ranked_by_size_not_by_category(report):
    """The failure this corrects: a per-category waste report lets the loudest
    finding outrank the largest one."""
    savings = [x.saving_usd for x in report.levers]
    assert savings == sorted(savings, reverse=True)


def test_realised_and_hypothetical_are_distinguished(report):
    statuses = {x.status for x in report.levers}
    assert statuses <= {"realised", "hypothetical"}


def test_every_hypothetical_lever_states_its_assumption(report):
    """A saving that needs the cheap model to have been good enough is not a
    saving until something has checked."""
    for lever in report.levers:
        if lever.status == "hypothetical":
            assert lever.assumption, f"{lever.name} claims a saving with no stated assumption"


def test_cost_shape_is_reported(report):
    """Whether cost is input- or output-dominated decides which lever can matter."""
    d = report.as_dict()
    assert 0 <= d["cost_shape"]["input_percent"] <= 100
    assert 0 <= d["cost_shape"]["output_percent"] <= 100


def test_the_note_refuses_to_call_a_repricing_a_saving(report):
    assert "does not judge" in report.as_dict()["note"]


def test_empty_period_does_not_crash_or_invent_levers(temp_db):
    r = analyse_levers(period="today")
    assert r.calls == 0
    assert r.levers == []
    assert r.as_dict()["biggest_lever"] is None


def test_caching_is_reported_as_realised_when_it_happened(sample_db):
    r = analyse_levers(period="all_time")
    caching = [x for x in r.levers if x.name == "prompt caching"]
    if caching:
        assert caching[0].status == "realised"
