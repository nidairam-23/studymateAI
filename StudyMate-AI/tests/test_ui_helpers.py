"""Tests for the frontend display helpers (no Streamlit needed)."""

from datetime import date

from agent import generate_study_plan, prioritize_topics, replan_schedule
from ui_helpers import (
    DEFAULT_GOAL,
    STATUS_LABELS,
    build_topic,
    completed_topic_names,
    completion_fraction,
    goal_or_default,
    hours_label,
    plan_to_rows,
    sample_topics,
    tool_log_rows,
    topics_to_rows,
    unfinished_topic_names,
)


def make_plan():
    return generate_study_plan(sample_topics(), 3)


def test_build_topic_matches_backend_format():
    topic = build_topic("  Python ", 3.0, 5, 2, date(2026, 5, 1))
    assert topic == {
        "name": "Python",
        "difficulty": 3,
        "importance": 5,
        "estimated_hours": 2.0,
        "deadline": "2026-05-01",
    }
    assert build_topic("X", 1, 1, 1)["deadline"] is None


def test_sample_topics_are_valid_for_the_backend():
    ranked = prioritize_topics(sample_topics())
    assert len(ranked) == 4 and all("priority_score" in t for t in ranked)


def test_topics_to_rows():
    rows = topics_to_rows([build_topic("A", 2, 3, 1.5)])
    assert rows == [{"Topic": "A", "Difficulty": 2, "Importance": 3, "Estimated hours": 1.5, "Deadline": "—"}]


def test_plan_to_rows_shows_every_task_with_status_label():
    plan = make_plan()
    rows = plan_to_rows(plan)
    assert [r["Topic"] for r in rows] == [t["topic"] for t in plan["tasks"]]
    assert {r["Status"] for r in rows} <= set(STATUS_LABELS.values())
    assert rows[0]["Planned hours"] == plan["tasks"][0]["hours"]


def test_unfinished_and_completed_names_and_progress():
    plan = make_plan()
    assert completion_fraction(plan) == 0.0
    new = replan_schedule(plan, ["Python"], 2)
    assert completed_topic_names(new) == ["Python"]
    assert "Python" not in unfinished_topic_names(new)
    assert completion_fraction(new) == 0.25


def test_completion_fraction_of_empty_plan():
    assert completion_fraction(generate_study_plan([], 2)) == 0.0


def test_small_formatters():
    assert hours_label(2.0) == "2 h" and hours_label(1.5) == "1.5 h"
    assert goal_or_default("   ") == DEFAULT_GOAL
    assert goal_or_default(" Pass the exam ") == "Pass the exam"
    assert tool_log_rows([{"tool": "t", "ok": False, "detail": "d"}]) == [
        {"Tool": "t", "Result": "Problem", "Detail": "d"}
    ]
