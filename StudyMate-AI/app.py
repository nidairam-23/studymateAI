"""StudyMate AI - VTU study planner (Streamlit).

Run:  pip install streamlit pandas altair   (optional: anthropic)
      streamlit run app.py
Needs Streamlit >= 1.39 (audio input and the live timer use st.audio_input and st.fragment).
If ANTHROPIC_API_KEY is set the plan order is suggested by Claude,
otherwise the built-in rule-based planner is used.
"""
import math
import os
import time
import uuid
from datetime import date, timedelta
from types import SimpleNamespace

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from provider_settings import load_streamlit_secrets, streamlit_secrets_or_empty

load_dotenv()
configured_secrets = streamlit_secrets_or_empty(st.secrets)
load_streamlit_secrets(configured_secrets, os.environ)

from agent import run_agent
from tests import chatassisstant, syllabuschecker

st.set_page_config(page_title="StudyMate AI", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------
# VTU curriculum (edit freely - names are per the 2022 scheme; colleges and
# electives vary, so verify against your own syllabus copy)
# ----------------------------------------------------------------------------
FIRST_YEAR = {
    1: ["Mathematics-I for CSE Stream", "Applied Physics for CSE Stream",
        "Principles of Programming using C", "Engineering Science Course (ESC)",
        "Communication Skills / English", "Indian Constitution",
        "Scientific Foundations of Health"],
    2: ["Mathematics-II for CSE Stream", "Applied Chemistry for CSE Stream",
        "Computer-Aided Engineering Drawing", "Programming Course (Python / C++)",
        "Professional Writing Skills in English", "Balake Kannada / Kannada",
        "Innovation and Design Thinking"],
}

BASE_BRANCH_CURRICULUM = {
    **FIRST_YEAR,
    3: ["Engineering Mathematics", "Digital Systems / Fundamentals", "Data Structures",
        "Operating Systems / Core Theory", "Professional Ethics / Social Connect",
        "Elective / Lab Course"],
    4: ["Analysis and Design of Algorithms", "Database / Systems Course",
        "Discrete Mathematics", "Microcontrollers / Embedded Systems",
        "Engineering Economics / Values", "Lab / Seminar"],
    5: ["Core Subject", "Professional Elective", "Research Methodology / IPR",
        "Technical Lab", "Open Elective", "Environmental Studies"],
    6: ["Core Subject", "Professional Elective", "Open Elective", "Mini Project",
        "Seminar / Lab"],
    7: ["Professional Elective", "Open Elective", "Project Phase-1", "Internship"],
    8: ["Technical Seminar", "Project Work Phase-2", "Internship"],
}

CURRICULUM = {
    "CSE": {**BASE_BRANCH_CURRICULUM},
    "ISE": {**BASE_BRANCH_CURRICULUM},
    "AIML": {**BASE_BRANCH_CURRICULUM},
    "AIDS": {**BASE_BRANCH_CURRICULUM},
    "ECE": {**BASE_BRANCH_CURRICULUM},
    "EEE": {**BASE_BRANCH_CURRICULUM},
    "Cyber Security": {**BASE_BRANCH_CURRICULUM},
    "Mechanical": {**BASE_BRANCH_CURRICULUM},
    "Other branch (add subjects manually)": {**FIRST_YEAR},
}
MODULES = [f"Module {i}" for i in range(1, 6)]  # VTU = 5 modules per subject

PAGE_DASH = "📊 Dashboard"
PAGE_SUBJ = "📚 Subjects & Modules"
PAGE_PLAN = "📅 My Study Plan"
PAGE_CHAT = "💬 Assistant"
SESSION_MIN = 90  # longest single study session (minutes)

UX_CSS = """
<style>
    :root {
        --ink: #24352c;
        --muted: #42544a;
        --line: rgba(76, 105, 88, 0.18);
        --surface: rgba(255, 255, 255, 0.78);
        --coral: #c97763;
        --lavender: #eadfff;
        --neon: #d8ff72;
        --mint: #dceee3;
        --sky: #deebf3;
    }

    .stApp {
        background: transparent !important;
        color: var(--ink);
        font-weight: 600;
    }
    .stApp :is(p, label, button, input, textarea, .stCaption, [data-testid="stCaptionContainer"]) {
        font-weight: 600;
        color: var(--ink) !important;
    }
    .stApp :is(h1, h2, h3, h4, .stTitle) {
        font-weight: 800;
        color: #281f3b !important;
        background: linear-gradient(90deg, rgba(234, 223, 255, 0.96), rgba(216, 255, 114, 0.42));
        border-left: 4px solid #a477df;
        border-radius: 4px;
        padding: 0.12rem 0.65rem 0.2rem;
        width: fit-content;
        max-width: 100%;
        text-shadow: 0 1px rgba(255, 255, 255, 0.65);
    }
    [data-testid="stAppViewContainer"] {
        background-image:
            linear-gradient(125deg, rgba(239, 229, 255, 0.46), rgba(224, 245, 232, 0.44), rgba(255, 236, 224, 0.48)),
            url('https://images.unsplash.com/photo-1455390582262-044cdead277a?auto=format&fit=crop&w=2400&q=85');
        background-color: #efe7f8;
        background-size: 260% 260%, cover;
        background-position: 0% 45%, center;
        background-repeat: no-repeat;
        background-attachment: scroll, fixed;
        animation: pastel-drift 32s ease-in-out infinite;
    }
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background: transparent !important;
    }

    @keyframes pastel-drift {
        0%, 100% { background-position: 0% 42%, center; }
        50% { background-position: 100% 58%, center; }
    }

    .stApp [data-testid="stHeader"] {
        background: rgba(247, 250, 245, 0.66);
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(76, 105, 88, 0.1);
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }
    .glass-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 10px 28px rgba(63, 88, 70, 0.07);
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    .hero-badge {
        display: inline-block;
        background: var(--lavender);
        color: #625180;
        border: 1px solid rgba(108, 83, 151, 0.18);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(116, 77, 62, 0.08);
    }
    .kpi-box {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.9), rgba(228, 241, 232, 0.84));
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1rem 0.75rem;
        height: 100%;
        box-shadow: 0 8px 22px rgba(63, 88, 70, 0.06);
        color: var(--ink);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }
    .kpi-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(63, 88, 70, 0.1);
    }
    .subject-summary {
        min-height: 92px;
        padding: 0.7rem 0.9rem;
        border: 1px solid rgba(83, 75, 99, 0.12);
        border-left-width: 3px;
        border-radius: 9px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .subject-summary--subjects {
        background: rgba(235, 226, 255, 0.78);
        border-left-color: #aa8be1;
    }
    .subject-summary--hours {
        background: rgba(220, 238, 227, 0.82);
        border-left-color: #78b892;
    }
    .subject-summary--branch {
        background: rgba(255, 236, 222, 0.82);
        border-left-color: #d29a70;
    }
    .subject-summary-label, .subject-summary-note {
        margin: 0;
        color: #536057;
        font-size: 0.72rem;
        font-weight: 500;
    }
    .stApp .subject-summary-label, .stApp .subject-summary-note {
        font-weight: 500;
    }
    .subject-summary-value {
        margin: 0.18rem 0 0;
        color: #2f4037;
        font-size: 1.15rem;
        font-weight: 650;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    .subject-summary-value--branch {
        font-size: 0.98rem;
    }
    .subject-summary-note {
        margin-top: 0.2rem;
    }
    .study-panel, .next-panel {
        min-height: 220px;
        padding: 1.15rem 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.76);
        border-radius: 12px;
        box-shadow: 0 14px 36px rgba(48, 57, 78, 0.11);
        backdrop-filter: blur(18px);
    }
    .study-panel {
        background: linear-gradient(155deg, rgba(237, 224, 255, 0.94), rgba(247, 241, 255, 0.9));
    }
    .next-panel {
        background: linear-gradient(155deg, rgba(222, 243, 230, 0.94), rgba(249, 253, 244, 0.92));
    }
    .desk-eyebrow {
        margin: 0 0 0.85rem;
        color: #675785;
        font-size: 0.68rem;
        font-weight: 800;
    }
    .desk-hours {
        margin: 0;
        color: var(--ink);
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.1;
    }
    .desk-hours span {
        margin-left: 0.35rem;
        color: var(--muted);
        font-size: 0.88rem;
        font-weight: 600;
    }
    .desk-stat-row {
        display: flex;
        justify-content: space-between;
        gap: 0.7rem;
        margin-top: 1.2rem;
        color: var(--muted);
        font-size: 0.8rem;
    }
    .desk-stat-row strong {
        color: var(--ink);
        font-size: 0.95rem;
    }
    .desk-progress {
        height: 7px;
        margin-top: 0.6rem;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(99, 82, 131, 0.13);
    }
    .desk-progress span {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #a989e8, var(--neon));
    }
    [class*="st-key-voice_input_panel"] {
        padding: 1rem 1.1rem 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.78);
        border-radius: 16px;
        background: linear-gradient(125deg, rgba(238, 228, 255, 0.96), rgba(222, 245, 232, 0.94));
        box-shadow: 0 12px 30px rgba(70, 61, 103, 0.13);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }
    [class*="st-key-voice_input_panel"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 34px rgba(70, 61, 103, 0.17);
    }
    .voice-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.7rem;
    }
    .voice-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #665384;
        font-size: 0.66rem;
        font-weight: 800;
    }
    .voice-pulse {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #b48be9;
        box-shadow: 0 0 0 4px rgba(180, 139, 233, 0.18);
        animation: mic-pulse 1.8s ease-out infinite;
    }
    @keyframes mic-pulse {
        0% { box-shadow: 0 0 0 2px rgba(180, 139, 233, 0.3); }
        75%, 100% { box-shadow: 0 0 0 8px rgba(180, 139, 233, 0); }
    }
    .voice-title {
        margin-top: 0.2rem;
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 800;
    }
    .voice-subtitle {
        margin-top: 0.12rem;
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 600;
    }
    .voice-wave {
        display: flex;
        align-items: center;
        gap: 3px;
        height: 32px;
        padding: 0 0.3rem;
    }
    .voice-wave span {
        width: 4px;
        height: 9px;
        border-radius: 999px;
        background: linear-gradient(#a989e8, #8fd7b0);
        transform-origin: center;
        animation: voice-wave 900ms ease-in-out infinite alternate;
    }
    .voice-wave span:nth-child(2n) { animation-delay: 180ms; }
    .voice-wave span:nth-child(3n) { animation-delay: 320ms; }
    @keyframes voice-wave {
        to { transform: scaleY(2.6); background: linear-gradient(#c0a3f2, var(--neon)); }
    }
    [class*="st-key-voice_input_panel"] [data-testid="stAudioInput"] {
        padding: 0.25rem 0.45rem;
        border: 1px solid rgba(103, 84, 132, 0.12);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.68);
        transition: border-color 160ms ease, box-shadow 160ms ease;
    }
    [class*="st-key-voice_input_panel"] [data-testid="stAudioInput"]:focus-within {
        border-color: #a989e8;
        box-shadow: 0 0 0 3px rgba(169, 137, 232, 0.18);
    }
    [class*="st-key-voice_input_panel"] [data-testid="stAudioInput"] button {
        border: 0;
        border-radius: 50%;
        background: #e8ddfb;
        color: #594477;
        transition: transform 160ms ease, background 160ms ease, box-shadow 160ms ease;
    }
    [class*="st-key-voice_input_panel"] [data-testid="stAudioInput"] button:hover {
        transform: scale(1.08);
        background: #d8c6f5;
        box-shadow: 0 0 0 5px rgba(180, 139, 233, 0.16);
    }
    .pill {
        display: inline-block;
        background: var(--mint);
        color: #3c6550;
        border: 1px solid rgba(76, 105, 88, 0.16);
        border-radius: 999px;
        padding: 0.3rem 0.7rem;
        font-weight: 700;
        font-size: 0.72rem;
        margin: 0.2rem 0.3rem 0 0;
    }
    .primary-btn button {
        background: var(--coral) !important;
        color: #fffaf7 !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        box-shadow: 0 5px 14px rgba(167, 91, 71, 0.18), 0 0 0 2px rgba(216, 255, 114, 0.34) !important;
    }
    .soft-btn button {
        border-radius: 999px !important;
    }
    .stButton > button {
        border-radius: 10px;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 16px rgba(63, 88, 70, 0.12);
    }
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div,
    .stTextInput > div,
    .stDateInput > div,
    .stDataFrame {
        background: rgba(255, 255, 255, 0.78) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        color: var(--ink) !important;
    }
    .stMarkdown, .stSubheader, .stTitle, .stHeader, .stMetric,
    .stDataFrame, .stSelectbox label, .stTextInput label, .stNumberInput label, .stCheckbox label {
        color: var(--ink) !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #42544a !important;
    }
    .stApp input::placeholder, .stApp textarea::placeholder {
        color: #52635a !important;
        opacity: 1;
    }
    .stApp [data-testid="stAudioInput"] [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stAudioInput"] p {
        color: #33443a !important;
    }
    .stTabs [role="tablist"] {
        background: rgba(255, 255, 255, 0.58);
        border-radius: 999px;
        border: 1px solid var(--line);
        padding: 0.22rem;
    }
    .stTabs [role="tab"] {
        color: var(--muted) !important;
        border-radius: 999px;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: var(--mint);
        color: var(--ink) !important;
    }
    .stSidebar {
        background: rgba(248, 250, 244, 0.86);
        border-right: 1px solid var(--line);
        backdrop-filter: blur(16px);
    }
    .stSidebar .stMarkdown, .stSidebar .stRadio, .stSidebar .stButton, .stSidebar .stCaption {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.38rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        background: rgba(255, 255, 255, 0.48);
        border: 1px solid transparent;
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] * {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
        background: #e9f2e9;
        border-color: var(--line);
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
        background: var(--lavender);
        border-color: rgba(108, 83, 151, 0.22);
        font-weight: 700;
    }
    .stSidebar .stButton > button {
        background: rgba(255, 255, 255, 0.66) !important;
        border: 1px solid var(--line) !important;
        color: var(--ink) !important;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--line);
        box-shadow: 0 6px 18px rgba(63, 88, 70, 0.05);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 12px 30px rgba(63, 74, 88, 0.08);
    }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #b995f7, var(--neon)) !important;
    }
    div[data-testid="stVegaLiteChart"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    @media (prefers-reduced-motion: reduce) {
        .stApp { animation: none; }
        .stButton > button, .kpi-box, [data-testid="stSidebar"] [data-testid="stRadioOption"],
        .voice-pulse, .voice-wave span {
            transition: none;
            animation: none;
        }
    }
</style>
"""


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "nav": PAGE_DASH, "tasks": [], "timer": None, "edits": {},
        "custom": {}, "saved": {}, "used_today": {}, "flash": None,
        "mode": "Rule-based agent", "pending_rows": [],
        "agent_plan": None, "agent_activity": None,
        "chat": [], "syllabus": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def keep(name, default):
    """Value remembered across page switches (widget keys are dropped when a
    page isn't rendered, so we mirror them in st.session_state.saved)."""
    return st.session_state.saved.get(name, default)


def remember(name, value):
    st.session_state.saved[name] = value
    return value


def get_task(task_id):
    return next((t for t in st.session_state.tasks if t["id"] == task_id), None)


# ----------------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------------
def priority(m):
    score = m["importance"] * 2 + m["difficulty"]
    if m["deadline"]:
        score += max(0, 10 - max((m["deadline"] - date.today()).days, 0))
    return score


def agent_topic_name(subject, module):
    return f"{subject} :: {module}"


def rows_to_agent_topics(rows):
    return [
        {
            "name": agent_topic_name(row["subject"], row["module"]),
            "difficulty": row["difficulty"],
            "importance": row["importance"],
            "estimated_hours": row["hours"],
            "deadline": row["deadline"].isoformat() if row["deadline"] else None,
        }
        for row in rows
    ]


def schedule(tasks, hours_per_day, used_today=0):
    """Assign dates to tasks in order, starting today."""
    cap = int(hours_per_day * 60)
    day = date.today()
    free = max(cap - used_today, 0)
    for t in tasks:
        while t["minutes"] > free:
            day += timedelta(days=1)
            free = cap
            if t["minutes"] > cap:
                break
        t["date"] = day.isoformat()
        free -= t["minutes"]


def gen_plan(rows):
    topics = rows_to_agent_topics(rows)
    result = run_agent(
        goal=keep("goal", "") or "Plan my study sessions",
        topics=topics,
        available_hours=sum(topic["estimated_hours"] for topic in topics),
        use_llm=True,
    )
    st.session_state.agent_activity = result
    if not result["success"]:
        st.session_state.flash = ("error", result["explanation"])
        return False

    agent_plan = result["plan"]
    st.session_state.agent_plan = agent_plan
    st.session_state.mode = "Claude tool-calling agent" if result["mode"] == "claude" else "Rule-based agent"
    rows_by_name = {agent_topic_name(row["subject"], row["module"]): row for row in rows}
    tasks = []
    for planned_topic in agent_plan["tasks"]:
        row = rows_by_name[planned_topic["topic"]]
        remaining = int(round(planned_topic["hours"] * 60))
        parts = max(1, math.ceil(remaining / SESSION_MIN))
        for p in range(1, parts + 1):
            chunk = min(SESSION_MIN, remaining)
            remaining -= chunk
            tasks.append({
                "id": uuid.uuid4().hex[:8], "subject": row["subject"],
                "module": row["module"], "topic": row["topic"], "minutes": chunk,
                "part": f"{p}/{parts}", "deadline": row["deadline"].isoformat() if row["deadline"] else None,
                "status": "pending", "actual": 0, "missed_count": 0,
                "priority": planned_topic["priority_score"], "date": None, "done_on": None,
            })
    schedule(tasks, keep("hpd", 3.0))
    st.session_state.tasks = tasks
    st.session_state.timer = None
    st.session_state.used_today = {}
    return True


def replan():
    tasks = st.session_state.tasks
    pending = [t for t in tasks if t["status"] == "pending"]
    agent_plan = st.session_state.agent_plan
    agent_rank = {}
    if agent_plan:
        module_states = {}
        for task in tasks:
            name = agent_topic_name(task["subject"], task["module"])
            module_states.setdefault(name, []).append(task["status"] == "done")
        completed_topics = [name for name, states in module_states.items() if all(states)]
        completed_set = set(completed_topics)
        remaining_hours = sum(
            task["estimated_hours"]
            for task in agent_plan["tasks"]
            if task["topic"] not in completed_set and not task["completed"]
        )
        result = run_agent(
            goal=keep("goal", "") or "Adjust my study plan",
            current_plan=agent_plan,
            completed_topics=completed_topics,
            remaining_hours=remaining_hours,
            use_llm=True,
        )
        st.session_state.agent_activity = result
        if result["success"]:
            st.session_state.agent_plan = result["plan"]
            agent_rank = {
                task["topic"]: index
                for index, task in enumerate(result["plan"]["tasks"])
            }
    pending.sort(key=lambda task: (
        -task["missed_count"],
        agent_rank.get(agent_topic_name(task["subject"], task["module"]), len(agent_rank)),
        -task["priority"],
    ))
    used = st.session_state.used_today.get(date.today().isoformat(), 0)
    schedule(pending, keep("hpd", 3.0), used_today=used)


def start_timer(task_id, minutes):
    now = time.time()
    st.session_state.timer = {"id": task_id, "start": now,
                              "end": now + minutes * 60, "minutes": minutes}


def finish_timer(task, tm, completed, expired=False):
    today = date.today().isoformat()
    elapsed = tm["minutes"] if expired else max(1, round((time.time() - tm["start"]) / 60))
    used = st.session_state.used_today
    used[today] = used.get(today, 0) + elapsed
    st.session_state.timer = None
    if completed:
        task["status"], task["done_on"], task["actual"] = "done", today, elapsed
        replan()
        st.session_state.flash = ("success", f"✅ Completed on time: {task['subject']} – {task['module']}")
    else:
        task["missed_count"] += 1
        task["priority"] += 3
        replan()
        why = "Time's up" if expired else "Task not finished"
        st.session_state.flash = ("warning", f"⏰ {why} – plan was re-adjusted and the task rescheduled.")


def submit_plan():
    rows = st.session_state.pending_rows
    if not rows:
        st.session_state.flash = ("error", "Select at least one subject and module first.")
        return
    if not gen_plan(rows):
        return
    st.session_state.nav = PAGE_PLAN
    st.session_state.flash = ("success", f"Plan generated ({st.session_state.mode}). Start a timer on today's tasks.")


def page_assistant():
    context = SimpleNamespace(
        CURRICULUM=CURRICULUM,
        MODULES=MODULES,
        keep=keep,
        remember=remember,
        get_task=get_task,
        gen_plan=gen_plan,
        start_timer=start_timer,
        finish_timer=finish_timer,
        replan=replan,
        timer_panel=timer_panel,
        syl=syllabuschecker,
    )
    chatassisstant.render_chat_page(context)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_subjects():
    st.markdown("<div class='hero-badge'>Study planning workspace</div>", unsafe_allow_html=True)
    st.title("📚 Subjects & Modules")
    st.caption("Pick your year and semester, choose your subjects, and turn them into a realistic study plan.")

    selected_count = len(st.session_state.pending_rows or [])
    daily_hours = float(keep("hpd", 3.0))
    branch_name = keep("branch", list(CURRICULUM)[0])
    year_number = keep("year", 1)
    semester_number = keep("sem", 1)
    top_cards = st.columns([1, 1, 1.35], gap="small")
    top_cards[0].markdown(
        "<div class='subject-summary subject-summary--subjects'>"
        "<p class='subject-summary-label'>Selected subjects</p>"
        f"<p class='subject-summary-value'>{selected_count}</p></div>",
        unsafe_allow_html=True,
    )
    top_cards[1].markdown(
        "<div class='subject-summary subject-summary--hours'>"
        "<p class='subject-summary-label'>Daily study hours</p>"
        f"<p class='subject-summary-value'>{daily_hours:.1f} h</p></div>",
        unsafe_allow_html=True,
    )
    top_cards[2].markdown(
        "<div class='subject-summary subject-summary--branch'>"
        "<p class='subject-summary-label'>Branch</p>"
        f"<p class='subject-summary-value subject-summary-value--branch'>{branch_name}</p>"
        f"<p class='subject-summary-note'>Year {year_number} · Semester {semester_number}</p></div>",
        unsafe_allow_html=True,
    )

    branches = list(CURRICULUM)
    c1, c2, c3 = st.columns(3)
    branch = remember("branch", c1.selectbox("Branch", branches, index=branches.index(keep("branch", branches[0]))))
    year = remember("year", c2.selectbox("Year", [1, 2, 3, 4], index=keep("year", 1) - 1,
                                         format_func=lambda y: f"Year {y}"))
    sems = [2 * year - 1, 2 * year]
    sem_default = keep("sem", sems[0])
    sem = remember("sem", c3.selectbox("Semester", sems, index=sems.index(sem_default) if sem_default in sems else 0,
                                       format_func=lambda s: f"Semester {s}"))
    ck = f"{branch}|{sem}"
    custom = st.session_state.custom.setdefault(ck, [])
    with st.container(border=True):
        with st.expander("➕ Add a subject that's not listed (electives, other branches)"):
            with st.form(f"add_{ck}", clear_on_submit=True):
                new = st.text_input("Subject name")
                if st.form_submit_button("Add subject") and new.strip():
                    custom.append(new.strip())
    options = CURRICULUM[branch].get(sem, []) + custom
    if not options:
        st.info("No subjects listed for this semester – add them above.")
        st.session_state.pending_rows = []
        return

    st.subheader(f"Year {year} · Semester {sem}")
    subs_key = f"subs|{ck}"
    chosen = remember(subs_key, st.multiselect(
        "Subjects", options, default=[s for s in keep(subs_key, []) if s in options]))
    syllabus_data = syllabuschecker.render_loader(ck, f"{branch} · Semester {sem}")
    if syllabus_data:
        for subject in chosen:
            with st.expander(f"Verify {subject} against the syllabus"):
                syllabuschecker.render_subject_check(
                    ck, subject, syllabus_data, st.session_state.edits
                )

    rows = []
    for s in chosen:
        mk = f"mods|{ck}|{s}"
        mods = remember(mk, st.multiselect(f"Modules – {s}", MODULES, default=keep(mk, MODULES)))
        for m in mods:
            e = st.session_state.edits.get((s, m), {})
            rows.append({
                "Subject": s, "Module": m, "Topic (optional)": e.get("Topic (optional)", ""),
                "Difficulty": e.get("Difficulty", 3), "Importance": e.get("Importance", 3),
                "Hours": e.get("Hours", 2.0), "Deadline": e.get("Deadline", pd.NaT),
            })

    st.session_state.pending_rows = []
    if rows:
        st.subheader("Fine-tune selected modules")
        df = pd.DataFrame(rows)
        df["Deadline"] = pd.to_datetime(df["Deadline"])
        sig = abs(hash(tuple((r["Subject"], r["Module"]) for r in rows)))
        edited = st.data_editor(
            df, key=f"editor_{sig}", hide_index=True, use_container_width=True,
            disabled=["Subject", "Module"],
            column_config={
                "Difficulty": st.column_config.NumberColumn("Difficulty (1-5)", min_value=1, max_value=5, step=1),
                "Importance": st.column_config.NumberColumn("Importance (1-5)", min_value=1, max_value=5, step=1),
                "Hours": st.column_config.NumberColumn("Est. hours", min_value=0.5, max_value=40.0, step=0.5),
                "Deadline": st.column_config.DateColumn("Deadline (optional)"),
            })
        out = []
        for _, r in edited.iterrows():
            st.session_state.edits[(r["Subject"], r["Module"])] = {
                k: r[k] for k in ["Topic (optional)", "Difficulty", "Importance", "Hours", "Deadline"]}
            dl = None if pd.isna(r["Deadline"]) else pd.to_datetime(r["Deadline"]).date()
            out.append({"subject": r["Subject"], "module": r["Module"],
                        "topic": r["Topic (optional)"] or "", "difficulty": int(r["Difficulty"]),
                        "importance": int(r["Importance"]), "hours": float(r["Hours"]), "deadline": dl})
        st.session_state.pending_rows = out

    st.divider()
    a, b = st.columns(2)
    with a:
        remember("hpd", st.number_input("Study hours available per day", 0.5, 16.0, float(keep("hpd", 3.0)), 0.5))
    with b:
        remember("goal", st.text_input("Your goal (optional)", keep("goal", ""), placeholder="e.g. Prepare for IA-2 / VTU exams"))
    action_col, help_col = st.columns([1, 2])
    with action_col:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        st.button("🚀 Submit & generate plan", type="primary", on_click=submit_plan)
        st.markdown('</div>', unsafe_allow_html=True)
    with help_col:
        st.caption("Tip: shortlist the important modules first, then fine-tune the difficulty, hours and deadlines below.")


@st.fragment(run_every=1)
def timer_panel():
    tm = st.session_state.timer
    if not tm:
        return
    task = get_task(tm["id"])
    left = int(tm["end"] - time.time())
    if left <= 0:
        finish_timer(task, tm, completed=False, expired=True)
        st.rerun()
    mins, secs = divmod(left, 60)
    with st.container(border=True):
        st.markdown(f"### ⏳ {mins:02d}:{secs:02d}")
        st.caption(f"{task['subject']} – {task['module']}")
        st.progress(min(1.0, max(0.0, 1 - left / (tm["minutes"] * 60))))
        a, b = st.columns(2)
        if a.button("✅ Completed", key="t_done", type="primary", use_container_width=True):
            finish_timer(task, tm, completed=True)
            st.rerun()
        if b.button("⏹ Couldn't finish", key="t_stop", use_container_width=True):
            finish_timer(task, tm, completed=False)
            st.rerun()


def task_label(t):
    s = f"**{t['subject']}** – {t['module']}"
    if t["topic"]:
        s += f" · {t['topic']}"
    return s + f"  _(part {t['part']})_"


def page_plan():
    st.title("Your Study Plan")
    tasks = st.session_state.tasks
    if not tasks:
        st.info("No plan yet. Choose subjects in **Subjects & Modules** and submit.")
        return
    today = date.today().isoformat()
    goal = keep("goal", "")
    head = st.columns([4, 1])
    head[0].caption(f"Daily study time: **{keep('hpd', 3.0):.1f} hours**" + (f" · Goal: {goal}" if goal else ""))
    if head[1].button("🔄 Adjust my plan", use_container_width=True):
        replan()
        st.session_state.flash = ("info", "Your remaining sessions have been rearranged.")
        st.rerun()

    activity = st.session_state.agent_activity
    if activity:
        with st.expander("How your plan was made", expanded=False):
            st.caption(f"Your goal: {goal or 'Plan my study sessions'}")
            if activity["success"]:
                plan = activity["plan"]
                if activity["action"] == "create_plan":
                    st.write(
                        f"StudyMate made a plan for {len(plan['tasks'])} modules, "
                        f"scheduling {plan['allocated_hours']:g} study hours."
                    )
                else:
                    completed_count = sum(task["completed"] for task in plan["tasks"])
                    st.write(
                        f"StudyMate updated your plan. {completed_count} completed module(s) "
                        f"were kept finished, and {plan['allocated_hours']:g} hours were assigned "
                        "to the remaining modules."
                    )
                st.success("The schedule was checked and is ready.")
            else:
                st.warning(activity["explanation"])

            if activity["fallback_reason"]:
                st.caption("StudyMate used its built-in planner. Your plan is still ready to use.")

            with st.expander("Technical details", expanded=False):
                mode_label = "Claude AI" if activity["mode"] == "claude" else "Built-in planner"
                st.write(f"Planner used: **{mode_label}**")
                if activity["tool_calls"]:
                    tool_names = {
                        "validate_schedule": "Check schedule",
                        "generate_study_plan": "Create study plan",
                        "replan_schedule": "Update study plan",
                    }
                    log_rows = [
                        {
                            "Step": tool_names.get(item["tool"], item["tool"]),
                            "Result": "Complete" if item["ok"] else "Needs attention",
                            "Details": item["detail"],
                        }
                        for item in activity["tool_calls"]
                    ]
                    st.dataframe(pd.DataFrame(log_rows), hide_index=True, use_container_width=True)
                if activity["fallback_reason"]:
                    st.caption(activity["fallback_reason"])

    if st.session_state.timer:
        timer_panel()

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Today")
    todays = [t for t in tasks if t["date"] == today]
    if not todays:
        st.success("Nothing scheduled for today 🎉")
    for t in todays:
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 1.4, 1.2])
            if t["status"] == "done":
                c1.markdown("✅ " + task_label(t))
                c2.caption(f"{t['actual']} min studied")
                continue
            note = f" · ⚠ rescheduled {t['missed_count']}×" if t["missed_count"] else ""
            if t["deadline"] and t["date"] > t["deadline"]:
                note += f" · ⛔ after deadline {t['deadline']}"
            c1.markdown(task_label(t) + f"\n\n⏱ planned {t['minutes']} min{note}")
            mins = c2.number_input("Timer (min)", 1, 240, int(t["minutes"]), key=f"min_{t['id']}",
                                   label_visibility="collapsed")
            if c3.button("▶ Start timer", key=f"s_{t['id']}", disabled=bool(st.session_state.timer),
                         use_container_width=True):
                start_timer(t["id"], mins)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Upcoming")
    upcoming = sorted({t["date"] for t in tasks if t["date"] > today and t["status"] == "pending"})
    if not upcoming:
        st.caption("No upcoming tasks.")
    for d in upcoming:
        day_tasks = [t for t in tasks if t["date"] == d and t["status"] == "pending"]
        total = sum(t["minutes"] for t in day_tasks) / 60
        with st.expander(f"{date.fromisoformat(d):%a, %d %b} · {total:.1f} h · {len(day_tasks)} task(s)"):
            for t in day_tasks:
                st.markdown("• " + task_label(t) + f" — {t['minutes']} min")
    st.markdown("</div>", unsafe_allow_html=True)


def page_dashboard():
    st.markdown("<div class='hero-badge'>Your study snapshot</div>", unsafe_allow_html=True)
    st.title("📊 Dashboard")
    tasks = st.session_state.tasks
    today = date.today().isoformat()
    today_tasks = [task for task in tasks if task["date"] == today]
    today_pending = [task for task in today_tasks if task["status"] == "pending"]
    today_done = [task for task in today_tasks if task["status"] == "done"]
    minutes_left = sum(task["minutes"] for task in today_pending)
    today_progress = round(len(today_done) / len(today_tasks) * 100) if today_tasks else 0

    summary_col, next_col = st.columns([1, 1.8], gap="large")
    with summary_col:
        st.markdown(
            "<div class='study-panel'>"
            f"<p class='desk-eyebrow'>STUDY DESK · {date.today():%a, %d %b}</p>"
            f"<p class='desk-hours'>{minutes_left / 60:.1f}<span>hours left today</span></p>"
            f"<div class='desk-stat-row'><span>Sessions complete</span><strong>{len(today_done)} / {len(today_tasks)}</strong></div>"
            f"<div class='desk-progress'><span style='width:{today_progress}%'></span></div>"
            f"<div class='desk-stat-row'><span>Daily study target</span><strong>{float(keep('hpd', 3.0)):.1f} h</strong></div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with next_col:
        if not tasks:
            st.markdown(
                "<div class='next-panel'><p class='desk-eyebrow'>YOUR NEXT STEP</p>"
                "<h3 style='margin:0 0 .45rem;color:#2f4037'>Start with one subject</h3>"
                "<p style='margin:0;color:#68776e'>Choose a subject and modules to build your first plan.</p></div>",
                unsafe_allow_html=True,
            )
            st.button("Choose subjects", key="dashboard_choose_subjects",
                      on_click=lambda: st.session_state.update(nav=PAGE_SUBJ))
        else:
            next_task = next((task for task in tasks if task["status"] == "pending"), None)
            title = f"{next_task['subject']} · {next_task['module']}" if next_task else "All sessions complete"
            detail = f"{next_task['minutes']} minutes planned" if next_task else "Your planned work is finished."
            st.markdown(
                "<div class='next-panel'><p class='desk-eyebrow'>UP NEXT</p>"
                f"<h3 style='margin:0 0 .45rem;color:#2f4037'>{title}</h3>"
                f"<p style='margin:0;color:#68776e'>{detail}</p></div>",
                unsafe_allow_html=True,
            )
            st.button("Open study plan", key="dashboard_open_plan",
                      on_click=lambda: st.session_state.update(nav=PAGE_PLAN))

    if not tasks:
        return
    df = pd.DataFrame(tasks)
    df["hours"] = df["minutes"] / 60
    df["Status"] = df.apply(lambda r: "Completed" if r["status"] == "done"
                            else ("Rescheduled" if r["missed_count"] > 0 else "Pending"), axis=1)
    done_h = df.loc[df["status"] == "done", "hours"].sum()
    total_h = df["hours"].sum()
    done_n = int((df["status"] == "done").sum())

    goal = keep("goal", "")
    if goal:
        st.caption(f"🎯 Goal: {goal}")

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi-box'><div style='color:#64748b;font-size:0.8rem;'>Tasks completed</div><div style='font-size:2rem;font-weight:800;line-height:1.1;'>" + f"{done_n}/{len(df)}" + "</div></div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi-box'><div style='color:#64748b;font-size:0.8rem;'>Hours done</div><div style='font-size:2rem;font-weight:800;line-height:1.1;'>" + f"{done_h:.1f}" + "</div></div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi-box'><div style='color:#64748b;font-size:0.8rem;'>Hours remaining</div><div style='font-size:2rem;font-weight:800;line-height:1.1;'>" + f"{total_h - done_h:.1f}" + "</div></div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi-box'><div style='color:#64748b;font-size:0.8rem;'>Rescheduled tasks</div><div style='font-size:2rem;font-weight:800;line-height:1.1;'>" + f"{int((df['missed_count'] > 0).sum())}" + "</div></div>", unsafe_allow_html=True)

    progress = (done_h / total_h) if total_h else 0.0
    st.progress(progress, text=f"Overall progress {progress * 100:.0f}%")
    if st.session_state.timer:
        st.warning("⏳ A timer is running – open **Plan & Replan** to finish it.")

    progress_tab, charts_tab = st.tabs(["Progress view", "Charts view"])
    with progress_tab:
        st.subheader("Task status")
        status_counts = df["Status"].value_counts()
        status_columns = st.columns(3)
        for column, status in zip(status_columns, ["Completed", "Pending", "Rescheduled"]):
            count = int(status_counts.get(status, 0))
            column.metric(status, f"{count} tasks")
            column.progress(count / len(df), text=f"{count / len(df):.0%} of plan")

        st.subheader("Subject progress")
        subject_totals = df.groupby("subject").agg(
            planned_hours=("hours", "sum"),
            completed_hours=("hours", lambda hours: hours[df.loc[hours.index, "status"] == "done"].sum()),
        ).sort_values("planned_hours", ascending=False)
        for subject, row in subject_totals.iterrows():
            planned = float(row["planned_hours"])
            completed = float(row["completed_hours"])
            ratio = completed / planned if planned else 0.0
            label, bar, amount = st.columns([1.5, 3, 1])
            label.markdown(f"**{subject}**")
            bar.progress(ratio, text=f"{ratio:.0%} complete")
            amount.caption(f"{completed:.1f} / {planned:.1f} h")

        st.subheader("Daily workload")
        daily = df.groupby("date").agg(
            hours=("hours", "sum"),
            tasks=("id", "count"),
            completed=("status", lambda statuses: int((statuses == "done").sum())),
        ).sort_index()
        max_daily_hours = float(daily["hours"].max()) if not daily.empty else 0.0
        for day, row in daily.iterrows():
            day_date = date.fromisoformat(day)
            label, bar, amount = st.columns([1.5, 3, 1])
            label.markdown(f"**{day_date:%a, %d %b}**")
            bar.progress(float(row["hours"]) / max_daily_hours if max_daily_hours else 0.0,
                         text=f"{int(row['completed'])} of {int(row['tasks'])} tasks complete")
            amount.caption(f"{float(row['hours']):.1f} h")

    with charts_tab:
        status_data = pd.DataFrame({
            "Status": ["Completed", "Pending", "Rescheduled"],
            "Tasks": [int(status_counts.get(s, 0)) for s in ["Completed", "Pending", "Rescheduled"]],
        })
        status_colors = alt.Scale(
            domain=["Completed", "Pending", "Rescheduled"],
            range=["#22d3ee", "#3b82f6", "#f59e0b"],
        )
        subject_chart_data = subject_totals.reset_index().rename(columns={
            "subject": "Subject", "planned_hours": "Planned", "completed_hours": "Completed",
        })
        daily_chart_data = df.groupby(["date", "Status"], as_index=False)["hours"].sum()
        daily_chart_data = daily_chart_data.rename(columns={"date": "Date", "hours": "Hours"})

        charts = st.columns(2)
        with charts[0]:
            st.subheader("Task status")
            status_chart = alt.Chart(status_data).mark_arc(innerRadius=62, outerRadius=100).encode(
                theta=alt.Theta("Tasks:Q"),
                color=alt.Color("Status:N", scale=status_colors, legend=alt.Legend(title=None)),
                tooltip=["Status:N", "Tasks:Q"],
            ).configure(background="transparent").configure_view(stroke=None)
            st.altair_chart(status_chart, use_container_width=True)
        with charts[1]:
            st.subheader("Hours by subject")
            subject_bars = alt.Chart(subject_chart_data).transform_fold(
                ["Planned", "Completed"], as_=["Work", "Hours"]
            ).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("Subject:N", sort="-x", title=None),
                x=alt.X("Hours:Q", title="Study hours"),
                color=alt.Color("Work:N", scale=alt.Scale(
                    domain=["Planned", "Completed"], range=["#2563eb", "#22d3ee"]
                ), legend=alt.Legend(title=None)),
                yOffset="Work:N",
                tooltip=["Subject:N", "Work:N", alt.Tooltip("Hours:Q", format=".1f")],
            ).configure(background="transparent").configure_view(stroke=None).configure_axis(
                domain=False, gridColor="rgba(125, 211, 252, 0.12)",
                labelColor="#bae6fd", titleColor="#bae6fd",
            )
            st.altair_chart(subject_bars, use_container_width=True)

        st.subheader("Daily workload")
        daily_chart = alt.Chart(daily_chart_data).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Date:O", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("hours:Q", title="Study hours"),
            color=alt.Color("Status:N", scale=status_colors, legend=alt.Legend(title=None)),
            tooltip=["Date:O", "Status:N", alt.Tooltip("hours:Q", format=".1f")],
        ).configure(background="transparent").configure_view(stroke=None).configure_axis(
            domain=False, gridColor="rgba(125, 211, 252, 0.12)",
            labelColor="#bae6fd", titleColor="#bae6fd",
        )
        st.altair_chart(daily_chart, use_container_width=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    init_state()
    st.markdown(UX_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## 🎓 StudyMate")
        st.markdown("<div class='pill'>Smart Study Planner</div>", unsafe_allow_html=True)
        st.radio("Navigate", [PAGE_DASH, PAGE_SUBJ, PAGE_PLAN, PAGE_CHAT], key="nav", label_visibility="collapsed")
        st.divider()
        if st.session_state.saved.get("branch"):
            st.caption(f"{keep('branch', '')}\n\nYear {keep('year', 1)} · Semester {keep('sem', 1)}")
        if st.button("Reset everything"):
            st.session_state.clear()
            st.rerun()

    # Anything still pending from an earlier day wasn't completed on time -> replan
    today = date.today().isoformat()
    if any(t["status"] == "pending" and t["date"] and t["date"] < today for t in st.session_state.tasks):
        replan()
        st.session_state.flash = ("info", "Some earlier tasks weren't completed, so your plan was moved forward.")

    if st.session_state.flash:
        kind, msg = st.session_state.flash
        getattr(st, kind)(msg)
        st.session_state.flash = None

    {
        PAGE_DASH: page_dashboard,
        PAGE_SUBJ: page_subjects,
        PAGE_PLAN: page_plan,
        PAGE_CHAT: page_assistant,
    }[st.session_state.nav]()


main()