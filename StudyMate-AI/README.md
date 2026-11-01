# StudyMate AI — Agentic Study Planner

**An Agentic AI-Based Productivity and Study Planning Assistant** (domain: Business Productivity, education use case)

> Team members: see `docs/TEAM_GUIDE.md` for file ownership (Person 1 backend / Person 2 frontend), the integration contract and the demo script.

## Abstract
StudyMate AI helps students turn a list of topics and a limited amount of study time into a realistic,
validated plan. It prioritizes topics, allocates time, checks the schedule, and **replans** when the student
finishes topics or the available time changes. The backend works fully offline with a rule-based
engine. If an Anthropic API key is configured, Claude acts as the agent's decision-maker through tool calling.

## Problem statement
Students rarely have enough time for everything. They guess what to study first, overrun their time budget,
and lose track when plans change. StudyMate AI makes these decisions transparent and keeps the schedule
consistent (never over the time limit, never rescheduling finished work).

## Main features
- Transparent priority score from importance, difficulty and optional deadline.
- Time allocation that never exceeds the available hours, with honest `planned` / `partial` / `unscheduled` status.
- Schedule validator that reports every problem (it never silently "fixes" a plan).
- Replanning that preserves completed topics and respects the new time limit.
- Agent controller with bounded validate-and-retry (max 2 correction attempts).
- Optional Claude tool calling, with automatic fallback to rule-based mode.

## Project structure
```text
StudyMate-AI/
├── app.py                # Streamlit frontend (Person 2)
├── agent.py              # Scheduling tools + agent controller (Person 1)
├── ui_helpers.py         # Display-only helpers used by app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── tests/
    ├── __init__.py
    ├── test_agent.py
    └── test_ui_helpers.py
└── docs/
    └── TEAM_GUIDE.md         # who owns what, integration contract, demo script
```

## Architecture and agent workflow
```text
 Streamlit app.py  ──►  run_agent(goal, ...)            (single entry point)
                              │
        ┌─────────────────────┴──────────────────────┐
   API key set?                                   no key / API error
        │ yes                                          │
 Claude picks a tool                         rule-based controller picks a tool
 (create_study_plan / replan_study_schedule)  (no plan yet → create, plan given → replan)
        └─────────────────────┬──────────────────────┘
                              ▼
   Python tools:  prioritize_topics → generate_study_plan  |  replan_schedule
                              ▼
                     validate_schedule  ── invalid? ──► retry (max 2 more attempts)
                              ▼
        structured result: plan + validation + explanation + tool log
```
**Scoring** (`PRIORITY_FORMULA` in `agent.py`):
`priority_score = 2*importance + 1*difficulty + urgency`, where `urgency = 0` without a deadline, otherwise
`10 * 7 / (7 + days_left)` (`days_left` is never below 0; a past deadline counts as due today and adds a warning).

**Allocation**: topics are taken in priority order; each gets `min(estimated_hours, time left)`.

**Claude never invents numbers.** Claude only chooses which Python tool to call. Hours, scores and validation are
computed by Python; your API key is read from the environment and is never put in a prompt.

## Setup
Requires Python 3.11+.

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```
Linux / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
The scheduling backend uses the Python standard library. Streamlit, pandas, and Altair power the frontend and dashboard. `anthropic` and `python-dotenv` are optional; `pytest` is for tests.

## Optional: AI providers
Copy `.env.example` to `.env` (it is git-ignored), then enter either `ANTHROPIC_API_KEY` for Claude or
`OPENAI_API_KEY` for ChatGPT. Never paste API keys into chat or commit `.env`. Restart Streamlit after saving
the file. The assistant uses `STUDYMATE_CLAUDE_MODEL` or `STUDYMATE_OPENAI_MODEL` if set; otherwise it uses
`claude-sonnet-5-5` or `gpt-4o-mini` respectively. The planner's Claude calls can be configured with `ANTHROPIC_MODEL`.

Without an Anthropic key, the planning backend runs in rule-based mode. `result["mode"]` tells you which mode ran, and
   `result["fallback_reason"]` explains a fallback (missing key, auth error, rate limit, network failure, bad response).

## Optional: offline voice transcription
Install `faster-whisper` from `requirements.txt` to record spoken prompts in the Assistant page and transcribe them
locally. The first transcription downloads the base speech model (about 150 MB); subsequent transcription runs locally.
This only transcribes audio. Generating assistant replies still requires a valid Anthropic or OpenAI API key. The study
planner itself continues to work offline in rule-based mode.

## Deploy on Streamlit Community Cloud
Push the project to a GitHub repository, create a Streamlit Community Cloud app for that repository, and set
`app.py` as the entry point. Add provider credentials in the app's **Settings → Secrets** using TOML, for example:

```toml
ANTHROPIC_API_KEY = "your-real-anthropic-key"
OPENAI_API_KEY = "your-real-openai-key"
```

Use only the provider you intend to enable; never commit real keys or key fragments. The app reads Streamlit secrets
as well as local `.env` values. Cloud audio is uploaded to the app's server for transcription, and the speech model is
downloaded there on the first transcription, so this is not private to the user's own device.

## Run the web app
```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501). In **Subjects & Modules**, choose a branch, year,
semester, subjects and modules; edit difficulty, importance, estimated hours or deadlines; enter a goal and daily study
capacity; then generate the plan. The app maps each subject/module to an agent topic and calls `run_agent` from
`agent.py`. The agent selects the create or replan tool, while Python performs prioritization, allocation and validation.
The frontend turns those allocations into sessions of up to 90 minutes and places them on days according to the daily
capacity. In **Plan & Replan**, start a timer and mark a session complete or unfinished. The app updates the plan and
calls the agent to replan. **Agent activity** shows the goal, action, mode, validation and tool-call log.

## Run the tests
```bash
python -m pytest -v
```
No API key or internet is needed. Quick console demo: `python agent.py`.

## Public API
```python
from agent import (
    prioritize_topics,
    generate_study_plan,
    validate_schedule,
    replan_schedule,
    run_agent,
)
```
| Function | Returns |
|---|---|
| `prioritize_topics(topics, today=None)` | topics sorted by priority, each with `priority_score` |
| `generate_study_plan(topics, available_hours, today=None)` | plan dict |
| `validate_schedule(plan, available_hours, previous_plan=None)` | `{"valid": bool, "errors": [str]}` |
| `replan_schedule(plan, completed_topics, remaining_hours)` | plan dict (same schema) |
| `run_agent(goal, topics=None, available_hours=None, current_plan=None, completed_topics=None, remaining_hours=None, use_llm=True, today=None)` | result dict |

Input topic:
```python
{"name": "Python", "difficulty": 3, "importance": 5, "estimated_hours": 2, "deadline": None}  # deadline: "YYYY-MM-DD" or None
```
Plan schema (identical for planning and replanning):
```python
{
  "tasks": [{
      "topic": "Python", "hours": 2.0, "estimated_hours": 2.0, "priority_score": 13.0,
      "completed": False, "status": "planned",      # planned | partial | unscheduled | completed
      "difficulty": 3, "importance": 5, "deadline": None,
  }],
  "available_hours": 3.0, "allocated_hours": 3.0, "remaining_hours": 0.0, "warnings": [],
}
```
`run_agent` result keys: `success`, `action` (`create_plan` / `replan` / `none`), `mode` (`claude` / `deterministic`),
`plan` (None on failure), `validation`, `explanation`, `llm_comment`, `fallback_reason`, `attempts`, `tool_calls`, `errors`.

Rules to know:
- `completed_topics` is a list/tuple/set of topic names (case-insensitive). Completed tasks get `hours = 0`, `status = "completed"`
  and are never rescheduled. A name that is not in the plan raises `ValidationError` (a `ValueError`).
- Replanning treats a topic as either completed or not. Time that was only *planned* before is not counted as studied, so
  an unfinished topic still needs its full `estimated_hours` (even if the old plan gave it only part of that time).
- `replan_schedule` sets `available_hours` in the new plan to the `remaining_hours` you pass.
- Bad input raises `ValidationError` (has `.errors`, a list of messages) in the four functions; `run_agent` returns `success=False` instead of raising.

## Frontend integration (Streamlit)
```python
import streamlit as st
from agent import generate_study_plan, replan_schedule, ValidationError

topics = [
    {"name": "Python", "difficulty": 3, "importance": 5, "estimated_hours": 2, "deadline": None},
    {"name": "Cloud Computing", "difficulty": 4, "importance": 4, "estimated_hours": 2, "deadline": None},
]

# 1-4. Generate a plan and display it
try:
    plan = generate_study_plan(topics, available_hours=3)
except ValidationError as exc:               # 7. handle invalid input
    st.error("\n".join(exc.errors))
else:
    st.session_state["plan"] = plan
    st.table(plan["tasks"])                  # list of dicts
    st.write(plan["allocated_hours"], plan["remaining_hours"])
    for warning in plan["warnings"]:
        st.warning(warning)

# 5-6. The student ticks "Python" as done and now has 2 hours left
try:
    new_plan = replan_schedule(st.session_state["plan"], ["Python"], remaining_hours=2)
    st.session_state["plan"] = new_plan
except ValidationError as exc:
    st.error("\n".join(exc.errors))
```
Using the agent instead (recommended for the demo, it also validates and explains):
```python
from agent import run_agent
result = run_agent("Plan my exam week", topics=topics, available_hours=3)
if result["success"]:
    st.write(result["explanation"]); st.table(result["plan"]["tasks"])
else:
    st.error(result["explanation"])
```
The frontend never needs to parse text or re-implement scheduling.

## How this demonstrates Agentic AI
- **Goal-directed**: the agent receives a goal ("plan my exam week", "I finished Python, I have 2 hours").
- **Tool use**: it selects and executes Python tools (planner, validator, replanner), with Claude choosing the tool when configured.
- **Feedback loop**: every plan is validated; failures trigger bounded correction attempts and are reported honestly.
- **Adaptation**: new information (completed topics, less time) leads to replanning that preserves state.
- **Safe fallback**: a rule-based controller keeps the app working without an API key. In that mode it is a deterministic
  tool-orchestrating workflow, not an LLM agent. The README and results state which mode ran.

## Known limitations
- A topic is either completed or not; partial progress (e.g. "I studied 1 of 2 hours") cannot be recorded yet.
- Greedy allocation: a high-priority long topic can use all the time. There are no clock times, breaks or daily limits.
- Past deadlines are only treated as maximally urgent, not forgiven or re-dated.
- The Claude integration was written against the Anthropic SDK's documented tool-use flow, but was only tested with a fake client
  (no live API call was made). Verify it once with your own key.

## Hackathon demo sequence
1. Select a branch, semester and several modules; give them different importance, difficulty, study hours and deadlines. Enter a concrete goal such as “Prepare for my upcoming exams” and generate the plan.
2. Expand **Agent activity** and point out the goal, selected create tool, deterministic/Claude mode, validation result and tool-call log.
3. Open **Plan & Replan**, start a short timer, mark the session complete, and show that the agent receives the updated completion state and replans unfinished work.
4. Mark a session unfinished or use **Replan now** and show the changed order and preserved completed work.
5. Run once without an API key to show deterministic fallback. With a configured `ANTHROPIC_API_KEY`, run again to demonstrate Claude tool selection. Do not claim a live Claude integration unless a real API call has been verified.
6. Run `python -m pytest -v` live.
