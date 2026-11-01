# StudyMate AI — Team Guide (Person 1 + Person 2)

## Who owns what
| Area | Files | Owner |
|---|---|---|
| Backend: prioritizing, planning, validation, replanning, agent controller, optional Claude mode | `agent.py`, `tests/test_agent.py` | **Person 1** (AI Agent & Backend) |
| Frontend: Streamlit screens, display helpers | `app.py`, `ui_helpers.py`, `tests/test_ui_helpers.py` | **Person 2** (Frontend) |
| Shared | `README.md`, `requirements.txt`, `.env.example`, `.gitignore`, `docs/` | Both |

Rule: `app.py` calls `run_agent(...)` for prioritization, plan creation, validation and replanning. The frontend maps
subject/module selections to agent topics, splits the validated allocations into timed sessions, and manages the UI
timer. Person 2 should not change the backend contract in `agent.py` without updating its tests and documentation.

## Integration contract (what Person 2 may rely on)
```python
from agent import (prioritize_topics, generate_study_plan, validate_schedule,
                   replan_schedule, run_agent, ValidationError)
```
- Topic: `{"name", "difficulty" 1-5, "importance" 1-5, "estimated_hours" > 0, "deadline" "YYYY-MM-DD" | None}`
- Plan (same shape for planning and replanning): `tasks`, `available_hours`, `allocated_hours`, `remaining_hours`, `warnings`
- Task `status`: `planned` | `partial` | `unscheduled` | `completed`
- `run_agent(...)` returns: `success`, `action`, `mode`, `plan`, `validation`, `explanation`, `llm_comment`,
  `fallback_reason`, `attempts`, `tool_calls`, `errors`. It does not raise on bad user input; check `success`.
- The four tool functions raise `ValidationError` (has `.errors`) on bad input.
- Replanning: pass completed topic names as a list/set; the new plan's `available_hours` equals the `remaining_hours` you pass.

## First-time checklist (do together)
1. `python -m venv .venv`, activate it, `python -m pip install -r requirements.txt`
2. `python -m pytest -v` (all tests should pass)
3. `python agent.py` (console demo of the backend)
4. `streamlit run app.py` (full app, rule-based agent mode when no key is configured)
5. Optional: copy `.env.example` to `.env`, add `ANTHROPIC_API_KEY`, restart, generate a plan, and check *Agent activity* shows `claude`

## Suggested git workflow
- One repo, branches `backend` (Person 1) and `frontend` (Person 2); merge into `main` before the demo.
- Never commit `.env`.
- If you change a field name or return value in `agent.py`, update `ui_helpers.py`, the tests and the README in the same change.

## Demo script (5 minutes)
1. Select a branch, year, semester and several modules; set different difficulty, importance, hours and deadlines. Enter a concrete study goal and generate the plan.
2. Expand *Agent activity*: show the selected action, deterministic/Claude mode, validation result and tool-call log.
3. In *Plan & Replan*, start a short session and mark it complete. Complete all sessions for one module to demonstrate that the agent preserves that completed topic during replanning.
4. Mark another session unfinished or click *Replan now* to show the schedule adapts.
5. Explain the fallback: no API key means deterministic mode; with a key Claude selects tools while Python computes and validates the schedule. Only describe Claude as live after verifying a real API call.
6. Run `python -m pytest -v` live.

## Known limitations to mention honestly
- Backend completion is topic-level; a module is marked complete after all its timed sessions are completed. Partial hours within a topic are not recorded by the backend.
- Greedy allocation, no clock times or breaks.
- The Claude tool-calling path was tested only with a fake client; verify once with a real key before presenting it as live.
