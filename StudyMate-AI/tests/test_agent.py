"""Automated tests for the StudyMate AI backend (no API key or internet needed)."""

import copy
import json
from datetime import date
from types import SimpleNamespace

import pytest

import agent
from agent import (
    ValidationError,
    generate_study_plan,
    prioritize_topics,
    replan_schedule,
    run_agent,
    validate_schedule,
)

TODAY = date(2026, 1, 1)  # fixed "today" so deadline tests are deterministic


def make_topics():
    return [
        {"name": "Python", "difficulty": 3, "importance": 5, "estimated_hours": 2, "deadline": None},
        {"name": "Cloud Computing", "difficulty": 4, "importance": 4, "estimated_hours": 2, "deadline": None},
        {"name": "Databases", "difficulty": 2, "importance": 3, "estimated_hours": 1.5, "deadline": None},
    ]


def task_of(plan, name):
    return next(t for t in plan["tasks"] if t["topic"] == name)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Every test starts without an API key unless it sets one itself."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# ---- 1-7: prioritization and input validation --------------------------------


def test_prioritize_returns_sorted_topics_with_scores():  # 1
    result = prioritize_topics(make_topics(), today=TODAY)
    assert [t["name"] for t in result] == ["Python", "Cloud Computing", "Databases"]
    assert result[0]["priority_score"] == 13.0  # 2*5 + 1*3
    assert all("priority_score" in t for t in result)
    assert result[0]["estimated_hours"] == 2  # original details preserved


def test_input_topics_are_not_modified():
    topics = make_topics()
    snapshot = copy.deepcopy(topics)
    prioritize_topics(topics, today=TODAY)
    generate_study_plan(topics, 3, today=TODAY)
    assert topics == snapshot


def test_importance_and_difficulty_affect_priority():  # 2
    base = {"name": "A", "difficulty": 1, "importance": 1, "estimated_hours": 1, "deadline": None}
    more_important = {**base, "name": "B", "importance": 3}
    harder = {**base, "name": "C", "difficulty": 3}
    ranked = [t["name"] for t in prioritize_topics([base, harder, more_important], today=TODAY)]
    assert ranked == ["B", "C", "A"]


def test_deadline_raises_priority_and_is_optional():  # 3
    no_deadline = {"name": "Later", "difficulty": 3, "importance": 3, "estimated_hours": 1}
    soon = {"name": "Soon", "difficulty": 3, "importance": 3, "estimated_hours": 1, "deadline": "2026-01-03"}
    ranked = prioritize_topics([no_deadline, soon], today=TODAY)
    assert ranked[0]["name"] == "Soon"
    assert ranked[0]["priority_score"] > ranked[1]["priority_score"]


def test_past_deadline_is_maximally_urgent_and_warns():
    topics = [{"name": "Late", "difficulty": 1, "importance": 1, "estimated_hours": 1, "deadline": "2025-12-01"}]
    assert prioritize_topics(topics, today=TODAY)[0]["priority_score"] == 13.0  # 2 + 1 + 10
    plan = generate_study_plan(topics, 1, today=TODAY)
    assert any("has passed" in w for w in plan["warnings"])


def test_malformed_deadline_rejected():
    topics = [{"name": "X", "difficulty": 1, "importance": 1, "estimated_hours": 1, "deadline": "01/02/2026"}]
    with pytest.raises(ValidationError, match="ISO date"):
        prioritize_topics(topics, today=TODAY)


def test_empty_topic_list_prioritize_raises():  # 4
    with pytest.raises(ValidationError, match="must not be empty"):
        prioritize_topics([], today=TODAY)


def test_empty_topic_list_plan_is_graceful():  # 4
    plan = generate_study_plan([], 3, today=TODAY)
    assert plan["tasks"] == [] and plan["allocated_hours"] == 0
    assert plan["remaining_hours"] == 3 and plan["warnings"]
    assert validate_schedule(plan, 3)["valid"]


@pytest.mark.parametrize("bad", [0, 6, -1, 2.5, "3", None, True])
def test_invalid_difficulty(bad):  # 5
    topics = [{"name": "X", "difficulty": bad, "importance": 3, "estimated_hours": 1}]
    with pytest.raises(ValidationError, match="difficulty"):
        prioritize_topics(topics, today=TODAY)


@pytest.mark.parametrize("bad", [0, 6, -2, 1.5, "high"])
def test_invalid_importance(bad):  # 6
    topics = [{"name": "X", "difficulty": 3, "importance": bad, "estimated_hours": 1}]
    with pytest.raises(ValidationError, match="importance"):
        prioritize_topics(topics, today=TODAY)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), "2", None])
def test_invalid_estimated_hours(bad):  # 7
    topics = [{"name": "X", "difficulty": 3, "importance": 3, "estimated_hours": bad}]
    with pytest.raises(ValidationError, match="estimated_hours"):
        prioritize_topics(topics, today=TODAY)


def test_missing_fields_are_all_reported():
    with pytest.raises(ValidationError) as info:
        prioritize_topics([{"name": "X"}], today=TODAY)
    text = str(info.value)
    assert "difficulty" in text and "importance" in text and "estimated_hours" in text
    assert len(info.value.errors) == 3


def test_duplicate_topic_names_rejected():  # 18
    topics = make_topics() + [{"name": " python ", "difficulty": 1, "importance": 1, "estimated_hours": 1}]
    with pytest.raises(ValidationError, match="duplicate"):
        generate_study_plan(topics, 5, today=TODAY)


# ---- 8-11, 19: plan generation -------------------------------------------------


@pytest.mark.parametrize("bad", [-1, "3", None, float("nan"), float("inf"), True])
def test_invalid_available_hours(bad):  # 8
    with pytest.raises(ValidationError, match="available_hours"):
        generate_study_plan(make_topics(), bad, today=TODAY)


def test_plan_fits_in_time_limit_and_matches_schema():  # 9
    plan = generate_study_plan(make_topics(), 10, today=TODAY)
    assert set(plan) == {"tasks", "available_hours", "allocated_hours", "remaining_hours", "warnings"}
    assert plan["allocated_hours"] == 5.5 and plan["remaining_hours"] == 4.5
    assert plan["warnings"] == []
    assert all(t["status"] == "planned" and t["completed"] is False for t in plan["tasks"])
    assert validate_schedule(plan, 10) == {"valid": True, "errors": []}


def test_insufficient_time_never_exceeds_limit():  # 10
    plan = generate_study_plan(make_topics(), 3, today=TODAY)
    assert plan["allocated_hours"] <= 3 and plan["remaining_hours"] == 0
    assert all(t["hours"] >= 0 for t in plan["tasks"])
    assert any("Insufficient time" in w for w in plan["warnings"])


def test_partial_allocation_is_reported_honestly():  # 11
    plan = generate_study_plan(make_topics(), 3, today=TODAY)
    assert task_of(plan, "Python")["status"] == "planned"
    cloud = task_of(plan, "Cloud Computing")
    assert cloud["status"] == "partial" and cloud["hours"] == 1.0 and cloud["estimated_hours"] == 2.0
    last = task_of(plan, "Databases")
    assert last["status"] == "unscheduled" and last["hours"] == 0.0
    assert any("partially scheduled" in w for w in plan["warnings"])


def test_zero_available_hours_gives_unscheduled_tasks():
    plan = generate_study_plan(make_topics(), 0, today=TODAY)
    assert all(t["status"] == "unscheduled" for t in plan["tasks"])
    assert validate_schedule(plan, 0)["valid"]


def test_summary_totals_are_consistent():  # 19
    plan = generate_study_plan(make_topics(), 3.7, today=TODAY)
    total = sum(t["hours"] for t in plan["tasks"])
    assert plan["allocated_hours"] == pytest.approx(total)
    assert plan["remaining_hours"] == pytest.approx(3.7 - total)
    assert plan["available_hours"] == 3.7


def test_decimal_hours_are_handled():
    topics = [
        {"name": "A", "difficulty": 3, "importance": 5, "estimated_hours": 0.1},
        {"name": "B", "difficulty": 3, "importance": 4, "estimated_hours": 0.2},
        {"name": "C", "difficulty": 3, "importance": 3, "estimated_hours": 0.3},
    ]
    plan = generate_study_plan(topics, 0.6, today=TODAY)
    assert validate_schedule(plan, 0.6)["valid"]
    assert all(t["status"] == "planned" for t in plan["tasks"])


# ---- 12-13: validation ---------------------------------------------------------


def test_validate_accepts_correct_schedule():  # 12
    plan = generate_study_plan(make_topics(), 3, today=TODAY)
    assert validate_schedule(plan, 3) == {"valid": True, "errors": []}


def test_validate_detects_time_overrun():  # 13
    plan = generate_study_plan(make_topics(), 3, today=TODAY)
    plan["tasks"][0]["hours"] = 5.0
    result = validate_schedule(plan, 3)
    assert result["valid"] is False
    assert any("exceeds available" in e for e in result["errors"])


def test_validate_detects_negative_and_inconsistent_values():
    plan = generate_study_plan(make_topics(), 5, today=TODAY)
    plan["tasks"][0]["hours"] = -1.0
    plan["tasks"][1]["topic"] = "  "
    plan["allocated_hours"] = 99
    errors = validate_schedule(plan, 5)["errors"]
    assert any("negative" in e for e in errors)
    assert any("'topic' must be a non-empty string" in e for e in errors)
    assert any("allocated_hours" in e for e in errors)


def test_validate_never_crashes_on_garbage():
    assert validate_schedule(None, 3)["valid"] is False
    assert validate_schedule({"tasks": "nope"}, 3)["valid"] is False
    assert validate_schedule({"tasks": [1]}, 3)["valid"] is False
    assert validate_schedule(generate_study_plan(make_topics(), 3, today=TODAY), -2)["valid"] is False


# ---- 14-17: replanning ---------------------------------------------------------


def test_replan_after_one_topic_completed():  # 14
    plan = generate_study_plan(make_topics(), 3, today=TODAY)
    new = replan_schedule(plan, ["Python"], 3)
    python = task_of(new, "Python")
    assert python["completed"] is True and python["status"] == "completed" and python["hours"] == 0
    assert task_of(new, "Cloud Computing")["status"] == "planned"  # now gets its full 2 h
    assert task_of(new, "Databases")["hours"] == 1.0
    assert set(new) == set(plan)  # same schema as generate_study_plan
    assert validate_schedule(new, 3, previous_plan=plan)["valid"]


def test_partially_planned_task_still_needs_full_time_on_replan():
    plan = generate_study_plan(make_topics(), 3, today=TODAY)
    assert task_of(plan, "Cloud Computing")["hours"] == 1.0  # only 1 of 2 h planned before
    new = replan_schedule(plan, [], 10)
    cloud = task_of(new, "Cloud Computing")
    assert cloud["hours"] == 2.0 and cloud["status"] == "planned" and cloud["completed"] is False


def test_replan_preserves_completed_tasks_and_never_reschedules_them():  # 15
    plan = generate_study_plan(make_topics(), 10, today=TODAY)
    step1 = replan_schedule(plan, ["Python"], 10)
    step2 = replan_schedule(step1, ["Cloud Computing"], 10)  # Python not listed again
    for name in ("Python", "Cloud Computing"):
        task = task_of(step2, name)
        assert task["completed"] and task["hours"] == 0 and task["status"] == "completed"
    assert task_of(step2, "Databases")["hours"] == 1.5
    assert step2["allocated_hours"] == 1.5


def test_replan_with_reduced_time():  # 16
    plan = generate_study_plan(make_topics(), 10, today=TODAY)
    new = replan_schedule(plan, [], 2.5)
    assert new["available_hours"] == 2.5 and new["allocated_hours"] == 2.5
    assert task_of(new, "Python")["status"] == "planned"
    assert task_of(new, "Cloud Computing")["status"] == "partial"
    assert task_of(new, "Cloud Computing")["hours"] == 0.5
    assert task_of(new, "Databases")["status"] == "unscheduled"
    assert any("Insufficient time" in w for w in new["warnings"])


def test_replan_with_zero_remaining_hours():  # 17
    plan = generate_study_plan(make_topics(), 10, today=TODAY)
    new = replan_schedule(plan, ["Python"], 0)
    assert new["allocated_hours"] == 0 and new["remaining_hours"] == 0
    assert task_of(new, "Python")["status"] == "completed"
    assert task_of(new, "Cloud Computing")["status"] == "unscheduled"
    assert task_of(new, "Cloud Computing")["completed"] is False
    assert validate_schedule(new, 0)["valid"]


def test_replan_all_completed():
    plan = generate_study_plan(make_topics(), 10, today=TODAY)
    new = replan_schedule(plan, ["python", "CLOUD COMPUTING", "Databases"], 5)
    assert new["allocated_hours"] == 0 and new["remaining_hours"] == 5
    assert any("All topics are completed" in w for w in new["warnings"])


def test_replan_rejects_bad_arguments():
    plan = generate_study_plan(make_topics(), 3, today=TODAY)
    with pytest.raises(ValidationError, match="not in the plan"):
        replan_schedule(plan, ["Quantum Physics"], 3)
    with pytest.raises(ValidationError, match="not a single string"):
        replan_schedule(plan, "Python", 3)
    with pytest.raises(ValidationError, match="remaining_hours"):
        replan_schedule(plan, [], -1)
    with pytest.raises(ValidationError, match="plan must be"):
        replan_schedule({"tasks": []}, [], 1)
    broken = copy.deepcopy(plan)
    broken["tasks"][0]["hours"] = 50
    with pytest.raises(ValidationError, match="invalid"):
        replan_schedule(broken, [], 3)


def test_replan_does_not_modify_the_input_plan():
    plan = generate_study_plan(make_topics(), 3, today=TODAY)
    snapshot = copy.deepcopy(plan)
    replan_schedule(plan, ["Python"], 2)
    assert plan == snapshot


def test_validate_detects_reintroduced_completed_task():
    plan = generate_study_plan(make_topics(), 10, today=TODAY)
    done = replan_schedule(plan, ["Python"], 10)
    reintroduced = copy.deepcopy(done)
    python = task_of(reintroduced, "Python")
    python.update(completed=False, hours=2.0, status="planned")
    reintroduced["allocated_hours"] += 2.0
    reintroduced["remaining_hours"] -= 2.0
    result = validate_schedule(reintroduced, 10, previous_plan=done)
    assert result["valid"] is False
    assert any("reintroduced" in e for e in result["errors"])


# ---- 20: agent -----------------------------------------------------------------


def test_agent_falls_back_when_no_api_key():  # 20
    result = run_agent("Plan my week", topics=make_topics(), available_hours=3, today=TODAY)
    assert result["success"] and result["mode"] == "deterministic"
    assert result["action"] == "create_plan"
    assert "ANTHROPIC_API_KEY" in result["fallback_reason"]
    assert result["validation"] == {"valid": True, "errors": []}
    assert result["plan"]["allocated_hours"] <= 3
    assert "deterministic" in result["explanation"]


def test_agent_replans_when_a_plan_is_given():
    first = run_agent("Plan", topics=make_topics(), available_hours=3, today=TODAY, use_llm=False)
    second = run_agent(
        "I finished Python",
        current_plan=first["plan"],
        completed_topics=["Python"],
        remaining_hours=2,
        use_llm=False,
    )
    assert second["success"] and second["action"] == "replan"
    assert task_of(second["plan"], "Python")["completed"] is True
    assert second["plan"]["available_hours"] == 2


def test_agent_reports_bad_input_without_crashing():
    bad_topic = [{"name": "X", "difficulty": 9, "importance": 3, "estimated_hours": 1}]
    result = run_agent("Plan", topics=bad_topic, available_hours=3, today=TODAY)
    assert result["success"] is False and result["plan"] is None
    assert result["errors"] and "difficulty" in result["explanation"]
    empty = run_agent("Plan", topics=[], available_hours=3, today=TODAY)  # graceful: empty plan + warning
    assert empty["success"] and empty["plan"]["tasks"] == [] and empty["plan"]["warnings"]
    missing = run_agent("Plan", topics=make_topics(), today=TODAY)
    assert missing["success"] is False and "available_hours" in missing["errors"][0]
    assert run_agent(123)["success"] is False


def test_agent_retries_then_accepts_valid_plan(monkeypatch):
    real = agent.generate_study_plan
    calls = {"n": 0}

    def flaky(topics, hours, today=None):
        calls["n"] += 1
        plan = real(topics, hours, today=today)
        if calls["n"] == 1:
            plan["tasks"][0]["hours"] = 999.0  # deliberately broken first attempt
        return plan

    monkeypatch.setattr(agent, "generate_study_plan", flaky)
    result = run_agent("Plan", topics=make_topics(), available_hours=3, today=TODAY, use_llm=False)
    assert result["success"] and result["attempts"] == 2 and calls["n"] == 2


def test_agent_gives_up_after_bounded_attempts(monkeypatch):
    real = agent.generate_study_plan
    calls = {"n": 0}

    def always_broken(topics, hours, today=None):
        calls["n"] += 1
        plan = real(topics, hours, today=today)
        plan["tasks"][0]["hours"] = 999.0
        return plan

    monkeypatch.setattr(agent, "generate_study_plan", always_broken)
    result = run_agent("Plan", topics=make_topics(), available_hours=3, today=TODAY, use_llm=False)
    assert result["success"] is False and result["plan"] is None
    assert calls["n"] == 1 + agent.MAX_CORRECTION_ATTEMPTS == result["attempts"]
    assert result["validation"]["valid"] is False and result["errors"]


def test_agent_survives_unexpected_tool_exception(monkeypatch):
    def boom(topics, hours, today=None):
        raise RuntimeError("tool exploded")

    monkeypatch.setattr(agent, "generate_study_plan", boom)
    result = run_agent("Plan", topics=make_topics(), available_hours=3, use_llm=False)
    assert result["success"] is False and "tool exploded" in result["explanation"]


# ---- Claude mode with a fake client (no network) -----------------------------


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def tool_use(name):
    block = SimpleNamespace(type="tool_use", id="toolu_1", name=name, input={})
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def final_text(text):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def test_claude_mode_uses_tool_but_python_computes_numbers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    client = FakeClient([tool_use("create_study_plan"), final_text("Your plan is ready.")])
    monkeypatch.setattr(agent, "_make_client", lambda key: client)
    result = run_agent("Plan my exam week", topics=make_topics(), available_hours=3, today=TODAY)
    assert result["mode"] == "claude" and result["success"] and result["fallback_reason"] is None
    assert result["llm_comment"] == "Your plan is ready."
    assert result["plan"] == generate_study_plan(make_topics(), 3, today=TODAY)
    assert "sk-test-secret" not in json.dumps(client.messages.calls, default=str)  # key never sent as prompt text
    assert client.messages.calls[0]["tools"] == agent.TOOL_DEFINITIONS


def test_claude_mode_tool_result_is_sent_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = FakeClient([tool_use("create_study_plan"), final_text("ok")])
    monkeypatch.setattr(agent, "_make_client", lambda key: client)
    run_agent("Plan", topics=make_topics(), available_hours=3, today=TODAY)
    second_call_messages = client.messages.calls[1]["messages"]
    tool_result = second_call_messages[-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert json.loads(tool_result["content"])["ok"] is True


def test_claude_unknown_tool_is_rejected_then_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = FakeClient([tool_use("delete_everything"), final_text("sorry")])
    monkeypatch.setattr(agent, "_make_client", lambda key: client)
    result = run_agent("Plan", topics=make_topics(), available_hours=3, today=TODAY)
    assert result["mode"] == "deterministic" and result["success"]
    assert "did not produce a validated plan" in result["fallback_reason"]


def test_claude_api_error_falls_back_with_reason(monkeypatch):
    class AuthenticationError(Exception):  # same class name the real SDK uses
        pass

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-bad")
    client = FakeClient([AuthenticationError("401")])
    monkeypatch.setattr(agent, "_make_client", lambda key: client)
    result = run_agent("Plan", topics=make_topics(), available_hours=3, today=TODAY)
    assert result["mode"] == "deterministic" and result["success"]
    assert "authentication failed" in result["fallback_reason"]


def test_claude_tool_loop_is_bounded(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = FakeClient([tool_use("create_study_plan")] * 10)
    monkeypatch.setattr(agent, "_make_client", lambda key: client)
    result = run_agent("Plan", topics=make_topics(), available_hours=3, today=TODAY)
    assert len(client.messages.calls) == agent.MAX_TOOL_ITERATIONS
    assert result["success"]  # a validated plan existed, so it is still accepted


def test_missing_sdk_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def no_sdk(key):
        raise agent.LLMError("the 'anthropic' package is not installed")

    monkeypatch.setattr(agent, "_make_client", no_sdk)
    result = run_agent("Plan", topics=make_topics(), available_hours=3, today=TODAY)
    assert result["mode"] == "deterministic" and result["success"]
    assert "not installed" in result["fallback_reason"]
