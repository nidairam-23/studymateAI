"""Display helpers for the Streamlit frontend (app.py).

This module only formats data for the screen. It contains NO scheduling logic:
all planning, validation and replanning is done by agent.py. It has no Streamlit
import, so it can be tested with plain pytest.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

STATUS_LABELS = {
    "planned": "🟢 Planned",
    "partial": "🟡 Partial",
    "unscheduled": "⚪ Not scheduled",
    "completed": "✔️ Completed",
}

DEFAULT_GOAL = "Create a study plan for my topics"


def hours_label(hours: float) -> str:
    """Format hours for display, e.g. 1.5 -> '1.5 h', 2.0 -> '2 h'."""
    return f"{hours:g} h"


def build_topic(
    name: str,
    difficulty: int,
    importance: int,
    estimated_hours: float,
    deadline: Optional[date] = None,
    subject: str = "",
) -> dict[str, Any]:
    """Create a topic dict in the format expected by agent.py.

    A module belongs to a subject; the subject is stored in the name
    ("Subject › Module") so agent.py needs no changes and the plan keeps it.
    """
    return {
        "name": make_topic_name(subject, name),
        "difficulty": int(difficulty),
        "importance": int(importance),
        "estimated_hours": float(estimated_hours),
        "deadline": deadline.isoformat() if deadline else None,
    }


SUBJECT_SEP = " › "
DEFAULT_SUBJECT = "General"


def make_topic_name(subject: str, module: str) -> str:
    """'Mathematics' + 'Calculus' -> 'Mathematics › Calculus' ('' if module is empty)."""
    subject = subject.replace("›", "-").strip()
    module = module.replace("›", "-").strip()
    if not module:
        return ""
    return f"{subject}{SUBJECT_SEP}{module}" if subject else module


def split_topic_name(name: str) -> tuple[str, str]:
    """'Mathematics › Calculus' -> ('Mathematics', 'Calculus'); no subject -> 'General'."""
    if SUBJECT_SEP in name:
        subject, module = name.split(SUBJECT_SEP, 1)
        return subject.strip() or DEFAULT_SUBJECT, module.strip()
    return DEFAULT_SUBJECT, name.strip()


def subject_names(topics: list[dict[str, Any]]) -> list[str]:
    """Distinct subjects in the order they were first added."""
    seen: list[str] = []
    for t in topics:
        subject = split_topic_name(t["name"])[0]
        if subject not in seen:
            seen.append(subject)
    return seen


def sample_modules() -> list[dict[str, Any]]:
    """Demo data grouped into subjects with modules."""
    soon = date.today() + timedelta(days=3)
    return [
        build_topic("Calculus Fundamentals", 4, 5, 1.5, soon, subject="Mathematics"),
        build_topic("Quadratic Equations", 2, 4, 1, subject="Mathematics"),
        build_topic("Thermodynamics", 4, 4, 1.5, subject="Physics"),
        build_topic("Newton's Laws", 3, 3, 1, subject="Physics"),
        build_topic("French Revolution", 2, 3, 1, subject="History"),
        build_topic("World War I", 3, 2, 1.5, subject="History"),
    ]


def sample_topics() -> list[dict[str, Any]]:
    """Demo data so the app can be shown quickly."""
    soon = date.today() + timedelta(days=3)
    return [
        build_topic("Python", 3, 5, 2),
        build_topic("Cloud Computing", 4, 4, 2),
        build_topic("Databases", 2, 3, 1.5, soon),
        build_topic("Operating Systems", 4, 2, 1),
    ]


def topics_to_rows(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for the 'your topics' table."""
    return [
        {
            "Topic": t["name"],
            "Difficulty": t["difficulty"],
            "Importance": t["importance"],
            "Estimated hours": t["estimated_hours"],
            "Deadline": t.get("deadline") or "—",
        }
        for t in topics
    ]


def plan_to_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for the study-plan table (one row per task, in plan order)."""
    return [
        {
            "Topic": t["topic"],
            "Status": STATUS_LABELS.get(t["status"], t["status"]),
            "Planned hours": t["hours"],
            "Estimated hours": t["estimated_hours"],
            "Priority": t["priority_score"],
            "Difficulty": t.get("difficulty"),
            "Importance": t.get("importance"),
            "Deadline": t.get("deadline") or "—",
        }
        for t in plan["tasks"]
    ]


def unfinished_topic_names(plan: dict[str, Any]) -> list[str]:
    """Names the student can still mark as completed."""
    return [t["topic"] for t in plan["tasks"] if not t["completed"]]


def completed_topic_names(plan: dict[str, Any]) -> list[str]:
    return [t["topic"] for t in plan["tasks"] if t["completed"]]


def completion_fraction(plan: dict[str, Any]) -> float:
    """Share of topics completed, from 0.0 to 1.0 (0.0 for an empty plan)."""
    tasks = plan["tasks"]
    return len(completed_topic_names(plan)) / len(tasks) if tasks else 0.0


def goal_or_default(goal: str) -> str:
    """Use the typed goal, or a default when the box is empty."""
    return goal.strip() or DEFAULT_GOAL


def tool_log_rows(tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Rows for the 'agent activity' table."""
    return [
        {"Tool": c["tool"], "Result": "OK" if c["ok"] else "Problem", "Detail": c["detail"]}
        for c in tool_calls
    ]


# --------------------------------------------------------------------------- #
# Dashboard helpers (display only)
# --------------------------------------------------------------------------- #


def minutes_label(hours: float) -> str:
    """Compact duration: 0.75 -> '45m', 2 -> '2h', 1.5 -> '1h 30m', 0 -> '0m'."""
    total = round(hours * 60)
    h, m = divmod(total, 60)
    if h and m:
        return f"{h}h {m}m"
    return f"{h}h" if h else f"{m}m"


def today_label(today: Optional[date] = None) -> str:
    """e.g. 'Tuesday, October 27'."""
    today = today or date.today()
    return f"{today:%A, %B} {today.day}"


def task_coverage(task: dict[str, Any]) -> float:
    """Share of a task's estimated time that is scheduled (1.0 when completed)."""
    if task["completed"]:
        return 1.0
    need = task["estimated_hours"]
    return min(task["hours"] / need, 1.0) if need > 0 else 0.0


def dashboard_stats(plan: Optional[dict[str, Any]], topics: list[dict[str, Any]]) -> dict[str, Any]:
    """Numbers for the four summary cards. Works with no plan yet."""
    if plan is None:
        return {"planned_hours": 0.0, "avg_coverage": 0.0, "done": 0, "total": 0, "topic_count": len(topics), "subject_count": len(subject_names(topics))}
    tasks = plan["tasks"]
    avg = sum(task_coverage(t) for t in tasks) / len(tasks) if tasks else 0.0
    return {
        "planned_hours": plan["allocated_hours"],
        "avg_coverage": avg,
        "done": len(completed_topic_names(plan)),
        "total": len(tasks),
        "topic_count": len(topics),
        "subject_count": len(subject_names(topics)),
    }


def attention_items(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Unfinished tasks that are only partly (or not) scheduled, least covered first."""
    items = [
        {
            "topic": t["topic"],
            "subject": split_topic_name(t["topic"])[0],
            "module": split_topic_name(t["topic"])[1],
            "coverage": task_coverage(t),
            "status": t["status"],
        }
        for t in plan["tasks"]
        if not t["completed"] and t["status"] in ("partial", "unscheduled")
    ]
    return sorted(items, key=lambda i: i["coverage"])


# --------------------------------------------------------------------------- #
# Subject / module analysis (display only: derived from topics and the plan)
# --------------------------------------------------------------------------- #


def module_rows(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for the modules table, with subject and module in separate columns."""
    rows = []
    for t in topics:
        subject, module = split_topic_name(t["name"])
        rows.append(
            {
                "Subject": subject,
                "Module": module,
                "Difficulty": t["difficulty"],
                "Importance": t["importance"],
                "Estimated hours": t["estimated_hours"],
                "Deadline": t.get("deadline") or "—",
            }
        )
    return sorted(rows, key=lambda r: r["Subject"].casefold())


def subject_overview(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-subject summary BEFORE planning: module count, hours, average ratings, next deadline."""
    rows = []
    for subject in subject_names(topics):
        mods = [t for t in topics if split_topic_name(t["name"])[0] == subject]
        deadlines = [t["deadline"] for t in mods if t.get("deadline")]
        rows.append(
            {
                "Subject": subject,
                "Modules": len(mods),
                "Total hours": sum(t["estimated_hours"] for t in mods),
                "Avg difficulty": round(sum(t["difficulty"] for t in mods) / len(mods), 1),
                "Avg importance": round(sum(t["importance"] for t in mods) / len(mods), 1),
                "Next deadline": min(deadlines) if deadlines else "—",
            }
        )
    return rows


def subject_analysis(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-subject analysis of a plan, weakest subject first.

    coverage = share of the subject's estimated hours that is finished or scheduled.
    status   = completed | needs attention (some open module is partial/unscheduled) | on track.
    """
    allocated = plan["allocated_hours"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for task in plan["tasks"]:
        groups.setdefault(split_topic_name(task["topic"])[0], []).append(task)

    result = []
    for subject, tasks in groups.items():
        total = sum(t["estimated_hours"] for t in tasks)
        covered = sum(t["estimated_hours"] if t["completed"] else t["hours"] for t in tasks)
        planned = sum(t["hours"] for t in tasks)
        open_tasks = [t for t in tasks if not t["completed"]]
        weakest = min(open_tasks, key=task_coverage) if open_tasks else None
        if not open_tasks:
            status = "completed"
        elif any(t["status"] in ("partial", "unscheduled") for t in open_tasks):
            status = "needs attention"
        else:
            status = "on track"
        result.append(
            {
                "subject": subject,
                "modules": len(tasks),
                "done": len(tasks) - len(open_tasks),
                "total_hours": total,
                "planned_hours": planned,
                "coverage": covered / total if total > 0 else 0.0,
                "share": planned / allocated if allocated > 0 else 0.0,
                "weakest": split_topic_name(weakest["topic"])[1] if weakest and weakest["status"] != "planned" else None,
                "status": status,
            }
        )
    return sorted(result, key=lambda r: r["coverage"])


SUBJECT_STATUS_LABELS = {"completed": "✔️ Completed", "needs attention": "🟡 Needs attention", "on track": "🟢 On track"}


def subject_analysis_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for the subject analysis table."""
    return [
        {
            "Subject": a["subject"],
            "Status": SUBJECT_STATUS_LABELS[a["status"]],
            "Modules done": f"{a['done']}/{a['modules']}",
            "Planned hours": a["planned_hours"],
            "Total hours": a["total_hours"],
            "Coverage %": round(a["coverage"] * 100),
            "Share of plan %": round(a["share"] * 100),
            "Weakest module": a["weakest"] or "—",
        }
        for a in subject_analysis(plan)
    ]