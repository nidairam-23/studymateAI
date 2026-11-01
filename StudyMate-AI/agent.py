"""StudyMate AI - backend: planning tools and agent controller.

Layout of this file
-------------------
1. Deterministic scheduling tools (no network, no API key needed):
   ``prioritize_topics``, ``generate_study_plan``, ``validate_schedule``,
   ``replan_schedule``.
2. Agent controller: ``run_agent`` picks and runs the tools, validates every
   plan before accepting it (bounded retries) and returns a structured result.
   If ``ANTHROPIC_API_KEY`` is set, Claude *chooses which tool to call*
   (tool calling). Python still computes and validates every number.
   Without a key (or if the API fails) a rule-based controller is used.

Data conventions
----------------
Topic (input)::

    {"name": str, "difficulty": 1-5 int, "importance": 1-5 int,
     "estimated_hours": positive number, "deadline": "YYYY-MM-DD" or None}

Task (inside a plan)::

    {"topic": str, "hours": float, "estimated_hours": float,
     "priority_score": float, "completed": bool, "status": str,
     "difficulty": int, "importance": int, "deadline": str | None}

``status`` is one of:
    "planned"     - the task's full estimated_hours are scheduled
    "partial"     - only part of estimated_hours is scheduled (0 < hours < estimated)
    "unscheduled" - no time could be given to the task (hours == 0)
    "completed"   - the student finished it; it gets hours == 0 and is never rescheduled

Plan (output of generate_study_plan / replan_schedule)::

    {"tasks": [task, ...], "available_hours": float, "allocated_hours": float,
     "remaining_hours": float, "warnings": [str, ...]}

``available_hours`` is the time budget the plan was made for. ``allocated_hours``
is the sum of task hours, ``remaining_hours`` is the unused part of the budget.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime
from typing import Any, Callable, Iterable, Optional

try:  # Optional: lets a local ".env" file provide ANTHROPIC_API_KEY.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

Topic = dict[str, Any]
Task = dict[str, Any]
Plan = dict[str, Any]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TOLERANCE = 1e-6  # floating-point tolerance (hours) for all time comparisons

IMPORTANCE_WEIGHT = 2.0
DIFFICULTY_WEIGHT = 1.0
MAX_URGENCY = 10.0
URGENCY_SCALE_DAYS = 7.0

PRIORITY_FORMULA = """
priority_score = 2.0 * importance + 1.0 * difficulty + urgency

urgency = 0                                   if the topic has no deadline
urgency = 10 * 7 / (7 + days_left)            otherwise, days_left = max(deadline - today, 0)

Examples: due in 0 days -> 10, due in 7 days -> 5, due in 21 days -> 2.5.
A deadline that is already in the past is treated as "due today" (maximum
urgency) and a warning is added to the plan. Ties keep the input order.
Score range: 3 (importance 1, difficulty 1, no deadline) to 25.
"""

STATUS_PLANNED = "planned"
STATUS_PARTIAL = "partial"
STATUS_UNSCHEDULED = "unscheduled"
STATUS_COMPLETED = "completed"
VALID_STATUSES = (STATUS_PLANNED, STATUS_PARTIAL, STATUS_UNSCHEDULED, STATUS_COMPLETED)

TASK_KEYS = ("topic", "hours", "estimated_hours", "priority_score", "completed", "status")
PLAN_KEYS = ("tasks", "available_hours", "allocated_hours", "remaining_hours", "warnings")

MAX_CORRECTION_ATTEMPTS = 2  # extra tries after the first validation fails
MAX_TOOL_ITERATIONS = 4  # max Claude <-> tool round trips per request
MAX_API_RETRIES = 2  # SDK-level retries for network / rate-limit errors
API_TIMEOUT_SECONDS = 30.0
DEFAULT_MODEL = "claude-sonnet-5-5"  # override with the ANTHROPIC_MODEL env var

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ValidationError(ValueError):
    """Bad input data. ``errors`` holds one readable message per problem."""

    def __init__(self, errors: Iterable[str] | str):
        self.errors: list[str] = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


class ScheduleValidationError(RuntimeError):
    """A produced schedule failed validation. ``errors`` lists the problems."""

    def __init__(self, errors: Iterable[str]):
        self.errors: list[str] = list(errors)
        super().__init__("Schedule failed validation: " + "; ".join(self.errors))


class LLMError(RuntimeError):
    """Claude mode could not be used (missing SDK, API failure, bad response)."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _is_number(value: Any) -> bool:
    """True for finite int/float values (bool is rejected on purpose)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normalize_key(name: str) -> str:
    """Topic names are matched ignoring surrounding spaces and letter case."""
    return name.strip().casefold()


def _check_hours(value: Any, label: str) -> float:
    """Validate a time budget (>= 0 hours) and return it as a float."""
    if not _is_number(value):
        raise ValidationError(f"{label} must be a finite number of hours, got {value!r}.")
    if value < 0:
        raise ValidationError(f"{label} must not be negative, got {value}.")
    return float(value)


def _int_in_range(value: Any, low: int, high: int) -> Optional[int]:
    """Return value as int if it is an integer in [low, high], else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    else:
        return None
    return number if low <= number <= high else None


def _parse_deadline(value: Any) -> Optional[date]:
    """Convert None / date / 'YYYY-MM-DD' into a date. Raises ValueError otherwise."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO_DATE.match(value.strip()):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            pass  # e.g. 2026-02-31 -> fall through to the error below
    raise ValueError("'deadline' must be None or a valid ISO date string 'YYYY-MM-DD'")


def _status_for(hours: float, need: float) -> str:
    """Status of an unfinished task that gets `hours` of a required `need`."""
    if hours <= TOLERANCE:
        return STATUS_UNSCHEDULED
    if hours >= need - TOLERANCE:
        return STATUS_PLANNED
    return STATUS_PARTIAL


# --------------------------------------------------------------------------- #
# Function A - prioritize_topics
# --------------------------------------------------------------------------- #


def _validate_topics(topics: Any) -> list[Topic]:
    """Check the topic list; raise ValidationError listing every problem found."""
    if not isinstance(topics, (list, tuple)):
        raise ValidationError("topics must be a list of dictionaries.")
    if len(topics) == 0:
        raise ValidationError("topics must not be empty. Provide at least one topic.")

    errors: list[str] = []
    seen: dict[str, int] = {}
    for index, topic in enumerate(topics, start=1):
        label = f"Topic #{index}"
        if not isinstance(topic, dict):
            errors.append(f"{label}: must be a dictionary, got {type(topic).__name__}.")
            continue

        name = topic.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{label}: 'name' must be a non-empty string.")
        else:
            label = f"Topic #{index} ('{name.strip()}')"
            key = _normalize_key(name)
            if key in seen:
                errors.append(f"{label}: duplicate topic name (same as topic #{seen[key]}).")
            else:
                seen[key] = index

        for field in ("difficulty", "importance"):
            if field not in topic:
                errors.append(f"{label}: missing required field '{field}'.")
            elif _int_in_range(topic[field], 1, 5) is None:
                errors.append(f"{label}: '{field}' must be an integer from 1 to 5, got {topic[field]!r}.")

        if "estimated_hours" not in topic:
            errors.append(f"{label}: missing required field 'estimated_hours'.")
        elif not _is_number(topic["estimated_hours"]) or topic["estimated_hours"] <= 0:
            errors.append(f"{label}: 'estimated_hours' must be a positive number, got {topic['estimated_hours']!r}.")

        try:
            _parse_deadline(topic.get("deadline"))
        except ValueError as exc:
            errors.append(f"{label}: {exc}, got {topic.get('deadline')!r}.")

    if errors:
        raise ValidationError(errors)
    return list(topics)


def _urgency(deadline: Optional[date], today: date) -> float:
    """Urgency part of the score (see PRIORITY_FORMULA)."""
    if deadline is None:
        return 0.0
    days_left = max((deadline - today).days, 0)
    return MAX_URGENCY * URGENCY_SCALE_DAYS / (URGENCY_SCALE_DAYS + days_left)


def _priority_score(topic: Topic, today: date) -> float:
    importance = _int_in_range(topic["importance"], 1, 5)
    difficulty = _int_in_range(topic["difficulty"], 1, 5)
    urgency = _urgency(_parse_deadline(topic.get("deadline")), today)
    return round(IMPORTANCE_WEIGHT * importance + DIFFICULTY_WEIGHT * difficulty + urgency, 2)


def prioritize_topics(topics: list[Topic], today: Optional[date] = None) -> list[Topic]:
    """Return copies of the topics sorted by priority (highest first).

    Each returned dict contains all original keys plus ``priority_score``.
    The score formula is documented in ``PRIORITY_FORMULA``. ``today`` exists so
    tests (or a demo) can fix the date; it defaults to ``date.today()``.

    Raises:
        ValidationError: empty list, duplicate names, missing fields, difficulty
            or importance outside 1-5, non-positive hours, malformed deadline.
    """
    valid = _validate_topics(topics)
    today = today or date.today()
    scored: list[tuple[int, Topic]] = []
    for index, topic in enumerate(valid):
        item = dict(topic)  # never modify the caller's data
        item["priority_score"] = _priority_score(topic, today)
        scored.append((index, item))
    scored.sort(key=lambda pair: (-pair[1]["priority_score"], pair[0]))
    return [item for _, item in scored]


# --------------------------------------------------------------------------- #
# Shared planning helpers
# --------------------------------------------------------------------------- #


def _allocate_hours(needs: list[float], pool: float) -> list[float]:
    """Give each need (already in priority order) as much of the pool as possible."""
    allocations: list[float] = []
    for need in needs:
        hours = min(need, pool)
        if hours <= TOLERANCE:
            hours = 0.0
        pool = max(0.0, pool - hours)
        if pool <= TOLERANCE:
            pool = 0.0
        allocations.append(hours)
    return allocations


def _schedule_warnings(tasks: list[Task], available: float) -> list[str]:
    """Human-readable warnings derived from the actual task allocations."""
    warnings: list[str] = []
    open_tasks = [t for t in tasks if not t["completed"]]
    if not tasks:
        return warnings
    if not open_tasks:
        return ["All topics are completed. Nothing is left to schedule."]

    needed = sum(t["estimated_hours"] for t in open_tasks)
    if available <= TOLERANCE:
        warnings.append("No study time is available, so no unfinished topic could be scheduled.")
    elif needed > available + TOLERANCE:
        warnings.append(
            f"Insufficient time: unfinished topics need {needed:g} h but only {available:g} h are "
            "available. Some topics need further study."
        )
    for task in open_tasks:
        if task["status"] == STATUS_PARTIAL:
            warnings.append(
                f"'{task['topic']}' is only partially scheduled "
                f"({task['hours']:g} of {task['estimated_hours']:g} h); it needs further study."
            )
        elif task["status"] == STATUS_UNSCHEDULED and available > TOLERANCE:
            warnings.append(f"'{task['topic']}' is not scheduled because no time is left; it needs further study.")
    return warnings


def _assemble_plan(tasks: list[Task], available: float, warnings: list[str]) -> Plan:
    """Build the plan dict; every summary number is computed from the tasks."""
    allocated = sum(t["hours"] for t in tasks)
    return {
        "tasks": tasks,
        "available_hours": available,
        "allocated_hours": round(allocated, 6),
        "remaining_hours": round(max(0.0, available - allocated), 6),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Function B - generate_study_plan
# --------------------------------------------------------------------------- #


def _make_task(topic: Topic, hours: float) -> Task:
    estimated = float(topic["estimated_hours"])
    deadline = _parse_deadline(topic.get("deadline"))
    return {
        "topic": topic["name"].strip(),
        "hours": hours,
        "estimated_hours": estimated,
        "priority_score": topic["priority_score"],
        "completed": False,
        "status": _status_for(hours, estimated),
        "difficulty": _int_in_range(topic["difficulty"], 1, 5),
        "importance": _int_in_range(topic["importance"], 1, 5),
        "deadline": deadline.isoformat() if deadline else None,
    }


def generate_study_plan(topics: list[Topic], available_hours: float, today: Optional[date] = None) -> Plan:
    """Create a study plan: highest-priority topics get their time first.

    Allocation is greedy: topics are taken in priority order and each receives
    ``min(estimated_hours, time still unallocated)``. So total allocated time never
    exceeds ``available_hours``; the first topic that does not fit becomes
    "partial" and later ones are "unscheduled". Nothing is ever marked
    completed here.

    An empty ``topics`` list returns an empty plan with a warning.

    Raises:
        ValidationError: invalid ``available_hours`` or invalid topics.
    """
    available = _check_hours(available_hours, "available_hours")
    if isinstance(topics, (list, tuple)) and len(topics) == 0:
        return _assemble_plan([], available, ["No topics were provided, so there is nothing to schedule."])

    today = today or date.today()
    ranked = prioritize_topics(topics, today)
    allocations = _allocate_hours([float(t["estimated_hours"]) for t in ranked], available)
    tasks = [_make_task(topic, hours) for topic, hours in zip(ranked, allocations)]

    warnings: list[str] = []
    for topic in ranked:
        deadline = _parse_deadline(topic.get("deadline"))
        if deadline is not None and deadline < today:
            warnings.append(
                f"The deadline for '{topic['name'].strip()}' ({deadline.isoformat()}) has passed; "
                "it is treated as maximally urgent."
            )
    warnings.extend(_schedule_warnings(tasks, available))
    return _assemble_plan(tasks, available, warnings)


# --------------------------------------------------------------------------- #
# Function C - validate_schedule
# --------------------------------------------------------------------------- #


def _task_errors(task: Any, index: int) -> list[str]:
    """All problems found in a single task."""
    if not isinstance(task, dict):
        return [f"Task #{index}: must be a dictionary."]
    name = task.get("topic")
    label = f"Task #{index}" + (f" ('{name}')" if isinstance(name, str) and name.strip() else "")
    missing = [key for key in TASK_KEYS if key not in task]
    if missing:
        return [f"{label}: missing required field(s): {', '.join(missing)}."]

    errors: list[str] = []
    if not isinstance(name, str) or not name.strip():
        errors.append(f"{label}: 'topic' must be a non-empty string.")

    hours, estimated = task["hours"], task["estimated_hours"]
    hours_ok = _is_number(hours) and hours >= 0
    estimated_ok = _is_number(estimated) and estimated > 0
    if not _is_number(hours):
        errors.append(f"{label}: 'hours' must be a number, got {hours!r}.")
    elif hours < 0:
        errors.append(f"{label}: 'hours' must not be negative, got {hours}.")
    if not estimated_ok:
        errors.append(f"{label}: 'estimated_hours' must be a positive number, got {estimated!r}.")
    if not _is_number(task["priority_score"]):
        errors.append(f"{label}: 'priority_score' must be a number.")
    if not isinstance(task["completed"], bool):
        errors.append(f"{label}: 'completed' must be True or False.")
    if task["status"] not in VALID_STATUSES:
        errors.append(f"{label}: 'status' must be one of {VALID_STATUSES}, got {task['status']!r}.")

    if hours_ok and estimated_ok:
        if hours > estimated + TOLERANCE:
            errors.append(f"{label}: 'hours' ({hours:g}) exceed 'estimated_hours' ({estimated:g}).")
        if isinstance(task["completed"], bool):
            if task["completed"]:
                if hours > TOLERANCE:
                    errors.append(f"{label}: a completed task must have 0 hours, got {hours:g}.")
                if task["status"] != STATUS_COMPLETED:
                    errors.append(f"{label}: a completed task must have status 'completed'.")
            else:
                expected = _status_for(hours, estimated)
                if task["status"] != expected:
                    errors.append(
                        f"{label}: status {task['status']!r} does not match {hours:g} of "
                        f"{estimated:g} h (expected {expected!r})."
                    )
    return errors


def _replan_consistency_errors(plan: Plan, previous_plan: Plan) -> list[str]:
    """Compare a replanned schedule with the plan it came from."""
    errors: list[str] = []

    def by_key(source: Any) -> dict[str, Task]:
        tasks = source.get("tasks", []) if isinstance(source, dict) else []
        return {
            _normalize_key(t["topic"]): t
            for t in tasks
            if isinstance(t, dict) and isinstance(t.get("topic"), str)
        }

    old, new = by_key(previous_plan), by_key(plan)
    for key, old_task in old.items():
        new_task = new.get(key)
        if new_task is None:
            errors.append(f"'{old_task['topic']}' from the previous plan is missing after replanning.")
            continue
        if old_task.get("completed") is True and new_task.get("completed") is not True:
            errors.append(f"Completed task '{old_task['topic']}' was reintroduced as unfinished.")
        if new_task.get("estimated_hours") != old_task.get("estimated_hours"):
            errors.append(f"'{old_task['topic']}': estimated_hours changed during replanning.")
        if new_task.get("priority_score") != old_task.get("priority_score"):
            errors.append(f"'{old_task['topic']}': priority_score changed during replanning.")
    for key, new_task in new.items():
        if key not in old:
            errors.append(f"'{new_task['topic']}' appeared during replanning but was not in the previous plan.")
    return errors


def validate_schedule(plan: Plan, available_hours: float, previous_plan: Optional[Plan] = None) -> dict[str, Any]:
    """Check a plan. Returns ``{"valid": bool, "errors": [str, ...]}``.

    Never raises and never repairs anything: an invalid plan is reported as invalid.

    Checks: plan structure; task fields and types; non-empty, unique topic names;
    non-negative hours that do not exceed ``estimated_hours``; status matches the
    hours; completed tasks have 0 hours; total allocated time does not exceed
    ``available_hours``; summary values match the tasks (tolerance 1e-6 h).
    If ``previous_plan`` is given (replanning), also checks that no task was
    dropped or added, and that completed tasks were not reintroduced as unfinished.
    """
    errors: list[str] = []
    if not _is_number(available_hours) or available_hours < 0:
        return {"valid": False, "errors": [f"available_hours must be a non-negative number, got {available_hours!r}."]}
    if not isinstance(plan, dict):
        return {"valid": False, "errors": ["plan must be a dictionary."]}

    missing = [key for key in PLAN_KEYS if key not in plan]
    if missing:
        errors.append(f"plan is missing required key(s): {', '.join(missing)}.")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        errors.append("plan['tasks'] must be a list.")
        return {"valid": False, "errors": errors}

    total = 0.0
    seen: dict[str, int] = {}
    for index, task in enumerate(tasks, start=1):
        errors.extend(_task_errors(task, index))
        if not isinstance(task, dict):
            continue
        if _is_number(task.get("hours")):
            total += task["hours"]
        name = task.get("topic")
        if isinstance(name, str) and name.strip():
            key = _normalize_key(name)
            if key in seen:
                errors.append(f"Task #{index} ('{name}'): duplicate topic name (same as task #{seen[key]}).")
            else:
                seen[key] = index

    if total > available_hours + TOLERANCE:
        errors.append(f"Allocated time ({total:g} h) exceeds available time ({available_hours:g} h).")

    expected_summary = {
        "available_hours": float(available_hours),
        "allocated_hours": total,
        "remaining_hours": max(0.0, available_hours - total),
    }
    for key, expected in expected_summary.items():
        if key not in plan:
            continue
        value = plan[key]
        if not _is_number(value):
            errors.append(f"'{key}' must be a number, got {value!r}.")
        elif abs(value - expected) > TOLERANCE:
            errors.append(f"'{key}' is {value:g} but should be {expected:g} based on the tasks.")
    if "warnings" in plan and (
        not isinstance(plan["warnings"], list) or not all(isinstance(w, str) for w in plan["warnings"])
    ):
        errors.append("'warnings' must be a list of strings.")

    if previous_plan is not None:
        errors.extend(_replan_consistency_errors(plan, previous_plan))
    return {"valid": not errors, "errors": errors}


# --------------------------------------------------------------------------- #
# Function D - replan_schedule
# --------------------------------------------------------------------------- #


def _normalize_completed(completed_topics: Any) -> dict[str, str]:
    """Turn the completed-topics argument into {normalized key: original name}."""
    if completed_topics is None:
        return {}
    if not isinstance(completed_topics, (list, tuple, set, frozenset)):
        raise ValidationError("completed_topics must be a list, tuple or set of topic names (not a single string).")
    result: dict[str, str] = {}
    for name in completed_topics:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(f"completed_topics may only contain non-empty strings, got {name!r}.")
        result[_normalize_key(name)] = name
    return result


def replan_schedule(plan: Plan, completed_topics: Iterable[str], remaining_hours: float) -> Plan:
    """Replan the unfinished topics of ``plan`` within ``remaining_hours``.

    How ``completed_topics`` is interpreted: a list/tuple/set of topic names
    (case-insensitive, surrounding spaces ignored). Those topics are marked
    completed (``completed=True``, ``status="completed"``, ``hours=0``) and are never
    scheduled again. Tasks already completed in ``plan`` stay completed. A name that
    is not in the plan raises ``ValidationError`` (so typos are not silently ignored).

    How remaining study need is computed: a topic is either completed or not.
    Hours that were only *planned* earlier are not treated as studied. So every
    unfinished topic still needs its full ``estimated_hours``, even if the old plan
    only gave it part of that time. Unfinished topics are re-allocated in the
    order of their original ``priority_score`` (not recalculated).

    The result has exactly the same structure as ``generate_study_plan()`` and
    ``available_hours`` is set to ``remaining_hours``. The result is validated
    before it is returned.

    Raises:
        ValidationError: bad arguments, invalid input plan, unknown completed topic.
        ScheduleValidationError: the new schedule failed validation (should not happen).
    """
    hours = _check_hours(remaining_hours, "remaining_hours")
    if not isinstance(plan, dict) or not _is_number(plan.get("available_hours")):
        raise ValidationError("plan must be a dictionary returned by generate_study_plan() or replan_schedule().")
    previous_check = validate_schedule(plan, plan["available_hours"])
    if not previous_check["valid"]:
        raise ValidationError(["The existing plan is invalid and cannot be replanned:"] + previous_check["errors"])

    done = _normalize_completed(completed_topics)
    known = {_normalize_key(t["topic"]) for t in plan["tasks"]}
    unknown = [name for key, name in done.items() if key not in known]
    if unknown:
        raise ValidationError(f"completed_topics contains topic(s) not in the plan: {', '.join(map(repr, unknown))}.")

    tasks = [dict(task) for task in plan["tasks"]]
    open_indexes: list[int] = []
    for index, task in enumerate(tasks):
        if task["completed"] or _normalize_key(task["topic"]) in done:
            task.update(completed=True, hours=0.0, status=STATUS_COMPLETED)
        else:
            open_indexes.append(index)

    open_indexes.sort(key=lambda i: (-tasks[i]["priority_score"], i))
    allocations = _allocate_hours([tasks[i]["estimated_hours"] for i in open_indexes], hours)
    for index, allocated in zip(open_indexes, allocations):
        tasks[index].update(
            hours=allocated,
            completed=False,
            status=_status_for(allocated, tasks[index]["estimated_hours"]),
        )

    new_plan = _assemble_plan(tasks, hours, _schedule_warnings(tasks, hours))
    check = validate_schedule(new_plan, hours, previous_plan=plan)
    if not check["valid"]:
        raise ScheduleValidationError(check["errors"])
    return new_plan


# --------------------------------------------------------------------------- #
# Agent controller (shared parts)
# --------------------------------------------------------------------------- #


def _log(ctx: dict[str, Any], tool: str, ok: bool, detail: str) -> None:
    ctx["tool_calls"].append({"tool": tool, "ok": ok, "detail": detail})


def _produce_validated_plan(
    ctx: dict[str, Any],
    tool_name: str,
    produce: Callable[[], Plan],
    hours_budget: float,
    previous_plan: Optional[Plan] = None,
) -> tuple[Optional[Plan], dict[str, Any], int]:
    """Run a planning tool and accept its plan only if validate_schedule() passes.

    Bounded retry: 1 initial attempt + at most MAX_CORRECTION_ATTEMPTS more.
    Bad input (ValidationError) is not retried - running again cannot fix it.
    A plan that never validates is returned as ``None`` with the errors.
    """
    last_errors: list[str] = []
    attempts = 0
    for attempts in range(1, MAX_CORRECTION_ATTEMPTS + 2):
        try:
            plan = produce()
        except ValidationError as exc:
            _log(ctx, tool_name, False, f"invalid input: {exc}")
            return None, {"valid": False, "errors": exc.errors}, attempts
        except ScheduleValidationError as exc:
            last_errors = exc.errors
            _log(ctx, tool_name, False, f"attempt {attempts}: {exc}")
            continue
        except Exception as exc:  # unexpected tool failure: record it, never crash the app
            last_errors = [f"{type(exc).__name__}: {exc}"]
            _log(ctx, tool_name, False, f"attempt {attempts}: unexpected error {last_errors[0]}")
            continue

        validation = validate_schedule(plan, hours_budget, previous_plan=previous_plan)
        _log(ctx, "validate_schedule", validation["valid"], "; ".join(validation["errors"]) or "schedule is valid")
        if validation["valid"]:
            _log(ctx, tool_name, True, f"plan accepted on attempt {attempts}")
            return plan, validation, attempts
        last_errors = validation["errors"]
    return None, {"valid": False, "errors": last_errors}, attempts


def _do_create(ctx: dict[str, Any]) -> bool:
    """Tool: create a new plan from ctx topics/hours. Updates ctx; returns success."""
    if ctx["topics"] is None or ctx["available_hours"] is None:
        ctx["errors"] = ["Creating a plan requires both 'topics' and 'available_hours'."]
        return False
    plan, validation, attempts = _produce_validated_plan(
        ctx,
        "generate_study_plan",
        lambda: generate_study_plan(ctx["topics"], ctx["available_hours"], today=ctx["today"]),
        ctx["available_hours"],
    )
    ctx["attempts"] = attempts
    ctx["validation"] = validation
    if plan is None:
        ctx["errors"] = validation["errors"]
        return False
    ctx.update(plan=plan, action="create_plan", errors=[])
    return True


def _do_replan(ctx: dict[str, Any]) -> bool:
    """Tool: replan ctx['current_plan']. Updates ctx; returns success."""
    if ctx["current_plan"] is None or ctx["remaining_hours"] is None:
        ctx["errors"] = ["Replanning requires both 'current_plan' and 'remaining_hours'."]
        return False
    plan, validation, attempts = _produce_validated_plan(
        ctx,
        "replan_schedule",
        lambda: replan_schedule(ctx["current_plan"], ctx["completed_topics"], ctx["remaining_hours"]),
        ctx["remaining_hours"],
        previous_plan=ctx["current_plan"],
    )
    ctx["attempts"] = attempts
    ctx["validation"] = validation
    if plan is None:
        ctx["errors"] = validation["errors"]
        return False
    ctx.update(plan=plan, action="replan", errors=[])
    return True


def _explain(ctx: dict[str, Any]) -> str:
    """Short explanation built from the real plan numbers (not from the LLM)."""
    plan = ctx["plan"]
    tasks = plan["tasks"]
    count = lambda status: sum(1 for t in tasks if t["status"] == status)  # noqa: E731
    if ctx["action"] == "create_plan":
        text = (
            f"Created a validated study plan for {len(tasks)} topic(s): {plan['allocated_hours']:g} of "
            f"{plan['available_hours']:g} hours allocated ({count(STATUS_PLANNED)} fully planned, "
            f"{count(STATUS_PARTIAL)} partial, {count(STATUS_UNSCHEDULED)} unscheduled)."
        )
    else:
        text = (
            f"Replanned the schedule: {count(STATUS_COMPLETED)} completed topic(s) kept as completed; "
            f"{plan['allocated_hours']:g} of {plan['available_hours']:g} hours allocated to unfinished topics "
            f"({count(STATUS_PLANNED)} fully planned, {count(STATUS_PARTIAL)} partial, "
            f"{count(STATUS_UNSCHEDULED)} unscheduled)."
        )
    if plan["warnings"]:
        text += f" {len(plan['warnings'])} warning(s) - see 'warnings' in the plan."
    return text


def _new_context(
    goal: Any,
    topics: Any,
    available_hours: Any,
    current_plan: Any,
    completed_topics: Any,
    remaining_hours: Any,
    today: Optional[date],
) -> dict[str, Any]:
    return {
        "goal": goal,
        "topics": topics,
        "available_hours": available_hours,
        "current_plan": current_plan,
        "completed_topics": completed_topics,
        "remaining_hours": remaining_hours,
        "today": today,
        "plan": None,
        "action": "none",
        "validation": {"valid": False, "errors": []},
        "attempts": 0,
        "errors": [],
        "tool_calls": [],
        "llm_comment": None,
    }


def _run_deterministic(ctx: dict[str, Any]) -> None:
    """Rule-based controller: existing plan -> replan, otherwise create a plan."""
    if ctx["current_plan"] is not None:
        _do_replan(ctx)
    else:
        _do_create(ctx)


# --------------------------------------------------------------------------- #
# Optional Claude tool-calling mode
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are the planning agent of StudyMate AI, a study-planning assistant. You can only act by "
    "calling tools. Pick the tool that fits the student's goal and situation. Never invent hours, "
    "priorities or schedules yourself: the tools compute and validate them. After the tool result "
    "arrives, reply with at most three short sentences that only use numbers from the tool result."
)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "create_study_plan",
        "description": (
            "Create a NEW prioritized and validated study plan from the topics and available hours "
            "supplied by the app. Use when the student has no plan yet or wants to start over."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Short reason for choosing this tool."}},
        },
    },
    {
        "name": "replan_study_schedule",
        "description": (
            "Replan an EXISTING plan using the completed topics and the new remaining hours supplied "
            "by the app. Completed topics are kept and never rescheduled. Use when the student reports "
            "finished topics or changed available time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Short reason for choosing this tool."}},
        },
    },
]

_API_ERROR_HINTS = {
    "AuthenticationError": "authentication failed (check ANTHROPIC_API_KEY)",
    "PermissionDeniedError": "permission denied by the API",
    "RateLimitError": "rate limit reached",
    "APITimeoutError": "request timed out",
    "APIConnectionError": "network connection failed",
    "InternalServerError": "the API reported a server error",
    "BadRequestError": "the API rejected the request",
    "NotFoundError": "model or endpoint not found (check ANTHROPIC_MODEL)",
}


def _get_api_key() -> Optional[str]:
    """Read the optional API key from the environment (never from code or prompts)."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key or None


def _make_client(api_key: str) -> Any:
    """Create the Anthropic client (imported lazily so the SDK stays optional)."""
    try:
        import anthropic
    except ImportError as exc:
        raise LLMError("the 'anthropic' package is not installed (pip install anthropic)") from exc
    return anthropic.Anthropic(api_key=api_key, max_retries=MAX_API_RETRIES, timeout=API_TIMEOUT_SECONDS)


def _describe_api_error(exc: Exception) -> str:
    for cls in type(exc).__mro__:
        if cls.__name__ in _API_ERROR_HINTS:
            return _API_ERROR_HINTS[cls.__name__]
    return f"unexpected {type(exc).__name__}: {exc}"


def _build_user_prompt(ctx: dict[str, Any]) -> str:
    """Describe the situation to Claude. Contains no secrets."""
    topics, completed = ctx["topics"], ctx["completed_topics"]
    topic_count = len(topics) if isinstance(topics, (list, tuple)) else 0
    completed_count = len(completed) if isinstance(completed, (list, tuple, set, frozenset)) else 0
    return "\n".join(
        [
            f"Student goal: {ctx['goal']}",
            "Situation (facts supplied by the app):",
            f"- topics provided: {topic_count}",
            f"- available_hours for a new plan: {ctx['available_hours']}",
            f"- existing plan: {'yes' if ctx['current_plan'] is not None else 'no'}",
            f"- completed topics reported: {completed_count}",
            f"- remaining_hours for replanning: {ctx['remaining_hours']}",
        ]
    )


def _execute_tool(name: str, ctx: dict[str, Any]) -> tuple[str, bool]:
    """Run one supported tool requested by Claude. Returns (JSON text, is_error)."""
    if name == "create_study_plan":
        ok = _do_create(ctx)
    elif name == "replan_study_schedule":
        ok = _do_replan(ctx)
    else:
        return json.dumps({"ok": False, "errors": [f"Unknown tool {name!r}."]}), True
    if not ok:
        return json.dumps({"ok": False, "errors": ctx["errors"]}), True
    plan = ctx["plan"]
    summary = {
        "ok": True,
        "validation": ctx["validation"],
        "available_hours": plan["available_hours"],
        "allocated_hours": plan["allocated_hours"],
        "remaining_hours": plan["remaining_hours"],
        "tasks": [{"topic": t["topic"], "hours": t["hours"], "status": t["status"]} for t in plan["tasks"]],
        "warnings": plan["warnings"],
    }
    return json.dumps(summary), False


def _extract_text(blocks: list[Any]) -> str:
    texts = [getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"]
    return " ".join(t.strip() for t in texts if t and t.strip())


def _run_claude_agent(ctx: dict[str, Any], api_key: str) -> None:
    """Let Claude choose tools (bounded loop). Raises LLMError on any API problem."""
    client = _make_client(api_key)
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    messages: list[dict[str, Any]] = [{"role": "user", "content": _build_user_prompt(ctx)}]
    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            blocks = list(response.content)
            if response.stop_reason != "tool_use":
                ctx["llm_comment"] = _extract_text(blocks) or None
                break
            tool_blocks = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
            if not tool_blocks:
                raise LLMError("Claude signalled a tool call but sent none")
            messages.append({"role": "assistant", "content": blocks})
            results = []
            for block in tool_blocks:
                content, is_error = _execute_tool(block.name, ctx)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})
    except LLMError:
        raise
    except Exception as exc:  # SDK/network/response-shape problems -> reported, then fallback
        raise LLMError(_describe_api_error(exc)) from exc
    if ctx["plan"] is None:
        raise LLMError("Claude did not produce a validated plan")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def run_agent(
    goal: str,
    topics: Optional[list[Topic]] = None,
    available_hours: Optional[float] = None,
    current_plan: Optional[Plan] = None,
    completed_topics: Optional[Iterable[str]] = None,
    remaining_hours: Optional[float] = None,
    use_llm: bool = True,
    today: Optional[date] = None,
) -> dict[str, Any]:
    """Single entry point for the frontend. Never raises for bad user input.

    Which action runs:
        * ``current_plan`` is None -> create a new plan (needs ``topics`` and ``available_hours``).
        * ``current_plan`` given   -> replan (needs ``remaining_hours``; ``completed_topics`` optional).

    Mode: if ``use_llm`` is True and ``ANTHROPIC_API_KEY`` is set, Claude chooses the
    tool (falling back to rule-based mode if the API fails). Otherwise the rule-based
    controller runs. Python code always computes and validates the schedule.

    Returns a dict with keys:
        success (bool)          - True only if a validated plan was produced
        action (str)            - "create_plan", "replan" or "none"
        mode (str)              - "claude" or "deterministic"
        plan (dict | None)      - same structure as generate_study_plan(); None on failure
        validation (dict)       - {"valid": bool, "errors": [str]}
        explanation (str)       - short summary of what happened
        llm_comment (str|None)  - Claude's own wording (Claude mode only; may be None)
        fallback_reason (str|None) - why rule-based mode was used instead of Claude
        attempts (int)          - number of planning attempts used (max 3)
        tool_calls (list[dict]) - log of {"tool", "ok", "detail"}
        errors (list[str])      - problems that stopped the request (empty on success)
    """
    args = (goal, topics, available_hours, current_plan, completed_topics, remaining_hours, today)
    ctx = _new_context(*args)
    mode = "deterministic"
    fallback_reason: Optional[str] = None

    if not isinstance(goal, str):
        ctx["errors"] = ["goal must be a string."]
    else:
        api_key = _get_api_key()
        if use_llm and api_key:
            try:
                _run_claude_agent(ctx, api_key)
                mode = "claude"
            except LLMError as exc:
                fallback_reason = f"Claude mode failed ({exc}); the rule-based agent was used instead."
                ctx = _new_context(*args)  # discard anything the failed Claude run produced
        elif use_llm:
            fallback_reason = "ANTHROPIC_API_KEY is not set; the rule-based agent was used."
        else:
            fallback_reason = "LLM mode was disabled (use_llm=False); the rule-based agent was used."
        if mode == "deterministic":
            _run_deterministic(ctx)

    success = ctx["plan"] is not None
    if success:
        explanation = _explain(ctx) + f" Mode: {mode}."
    else:
        explanation = "Could not complete the request: " + ("; ".join(ctx["errors"]) or "unknown error") + "."
    if fallback_reason:
        explanation += f" Note: {fallback_reason}"

    return {
        "success": success,
        "action": ctx["action"] if success else "none",
        "mode": mode,
        "plan": ctx["plan"],
        "validation": ctx["validation"],
        "explanation": explanation,
        "llm_comment": ctx["llm_comment"],
        "fallback_reason": fallback_reason,
        "attempts": ctx["attempts"],
        "tool_calls": ctx["tool_calls"],
        "errors": [] if success else ctx["errors"],
    }


if __name__ == "__main__":  # quick console demo: python agent.py
    demo_topics = [
        {"name": "Python", "difficulty": 3, "importance": 5, "estimated_hours": 2, "deadline": None},
        {"name": "Cloud Computing", "difficulty": 4, "importance": 4, "estimated_hours": 2, "deadline": None},
        {"name": "Databases", "difficulty": 2, "importance": 3, "estimated_hours": 1.5, "deadline": None},
    ]
    first = run_agent("Plan my study session", topics=demo_topics, available_hours=3, use_llm=False)
    print(first["explanation"])
    for item in first["plan"]["tasks"]:
        print(f"  {item['topic']:<16} {item['hours']:>4g} h  {item['status']}")
    second = run_agent(
        "I finished Python, I now have 2 hours",
        current_plan=first["plan"],
        completed_topics=["Python"],
        remaining_hours=2,
        use_llm=False,
    )
    print(second["explanation"])
    for item in second["plan"]["tasks"]:
        print(f"  {item['topic']:<16} {item['hours']:>4g} h  {item['status']}")
