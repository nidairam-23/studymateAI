"""StudyMate AI assistant: a chat that can operate the app through tool calling.

Works with Claude (ANTHROPIC_API_KEY) and/or ChatGPT (OPENAI_API_KEY).
    pip install anthropic openai      (install only the one you use)
Voice (optional): speak with the microphone; replies can be read aloud.
    Speech-to-text uses OpenAI if OPENAI_API_KEY is set, otherwise:  pip install faster-whisper
    Needs Streamlit >= 1.39 (st.audio_input). Claude itself cannot transcribe audio.

app.py passes its own functions in through `ctx`, so this file never imports app.py.
The model never touches your data directly: it can only call the tools listed in
TOOL_SPECS, and every tool is plain Python that you can read and limit.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import date

import pandas as pd
import streamlit as st

import ollama as ollama_provider

CLAUDE_MODEL = os.getenv("STUDYMATE_CLAUDE_MODEL", "claude-sonnet-5-5")
OPENAI_MODEL = os.getenv("STUDYMATE_OPENAI_MODEL", "gpt-4o-mini")
MAX_STEPS = 6  # tool round-trips allowed per message
OFFLINE = "Offline (Ollama)"

SYSTEM_PROMPT = (
    "You are StudyMate, a friendly study assistant for VTU engineering students. "
    "You operate the app with tools: check progress, create or adjust the study plan, "
    "start timers and mark modules done. Call get_study_status before answering anything "
    "about the current plan, and list_subjects before using subject names you are unsure "
    "of. Never invent task ids or subjects. After changing something, say briefly what you "
    "did. Keep answers short, practical and encouraging."
)

TOOL_SPECS = [
    {"name": "get_study_status",
     "description": "Current plan: today's tasks (with ids), upcoming tasks, completed count, running timer.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_subjects",
     "description": "Subjects available for a semester of the student's branch.",
     "input_schema": {"type": "object", "properties": {
         "semester": {"type": "integer", "description": "1-8. Omit for the currently selected semester."}}}},
    {"name": "create_plan",
     "description": "Select subjects (all 5 modules each) and generate a fresh study plan. Replaces the existing plan.",
     "input_schema": {"type": "object", "properties": {
         "subjects": {"type": "array", "items": {"type": "string"}},
         "semester": {"type": "integer", "description": "1-8. Omit to keep the current semester."},
         "hours_per_module": {"type": "number", "description": "Default 2."},
         "difficulty": {"type": "integer", "description": "1 easy - 5 hard. Default 3."},
         "importance": {"type": "integer", "description": "1 low - 5 high. Default 3."},
         "daily_hours": {"type": "number", "description": "Study hours available per day."},
         "goal": {"type": "string"}},
         "required": ["subjects"]}},
    {"name": "start_timer",
     "description": "Start the study timer for a task. Only one timer can run at a time.",
     "input_schema": {"type": "object", "properties": {
         "task_id": {"type": "string"},
         "minutes": {"type": "integer", "description": "Defaults to the task's planned minutes."}},
         "required": ["task_id"]}},
    {"name": "complete_task",
     "description": "Mark a task as finished on time. The plan is re-adjusted.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "mark_not_finished",
     "description": "The student could not finish a task. It is rescheduled and the plan is re-adjusted.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "replan_now",
     "description": "Re-adjust all remaining sessions from today.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "set_daily_hours",
     "description": "Change study hours available per day and re-adjust the plan.",
     "input_schema": {"type": "object", "properties": {"hours": {"type": "number"}}, "required": ["hours"]}},
    {"name": "check_syllabus",
     "description": "Read what the student's loaded VTU syllabus PDF says about a subject (and optionally one module).",
     "input_schema": {"type": "object", "properties": {
         "subject": {"type": "string", "description": "Subject name or course code, e.g. BCS303."},
         "module": {"type": "integer", "description": "1-5. Omit for all modules."}},
         "required": ["subject"]}},
]


# ----------------------------------------------------------------------------
# Provider loops (same idea, different message formats)
# ----------------------------------------------------------------------------
def available_providers() -> list[str]:
    found = []
    if ollama_provider.available():
        found.append(OFFLINE)  # free and private, so it is listed first
    if os.getenv("ANTHROPIC_API_KEY"):
        found.append("Claude")
    if os.getenv("OPENAI_API_KEY"):
        found.append("ChatGPT")
    return found


def _call(tools: dict, name: str, args: dict) -> dict:
    fn = tools.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name}"}
    try:
        return fn(**(args or {}))
    except TypeError as exc:
        return {"error": f"Bad arguments: {exc}"}
    except Exception as exc:  # report to the model instead of crashing the app
        return {"error": str(exc)}


def _claude_turn(messages: list[dict], tools: dict) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msgs = list(messages)
    for _ in range(MAX_STEPS):
        resp = client.messages.create(model=CLAUDE_MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
                                      tools=TOOL_SPECS, messages=msgs)
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        msgs.append({"role": "assistant", "content": resp.content})
        results = [
            {"type": "tool_result", "tool_use_id": b.id,
             "content": json.dumps(_call(tools, b.name, b.input), default=str)}
            for b in resp.content if b.type == "tool_use"
        ]
        msgs.append({"role": "user", "content": results})
    return "That took too many steps. Could you try a simpler request?"


def _openai_turn(messages: list[dict], tools: dict) -> str:
    from openai import OpenAI

    client = OpenAI()
    oa_tools = [{"type": "function", "function": {
        "name": s["name"], "description": s["description"], "parameters": s["input_schema"]}} for s in TOOL_SPECS]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
    for _ in range(MAX_STEPS):
        msg = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs, tools=oa_tools).choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        msgs.append(msg)
        for call in msg.tool_calls:
            out = _call(tools, call.function.name, json.loads(call.function.arguments or "{}"))
            msgs.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(out, default=str)})
    return "That took too many steps. Could you try a simpler request?"


def run_chat_turn(history: list[dict], user_text: str, tools: dict, provider: str) -> str:
    messages = history + [{"role": "user", "content": user_text}]
    if provider == OFFLINE:
        return ollama_provider.chat_turn(messages, TOOL_SPECS, SYSTEM_PROMPT, lambda n, a: _call(tools, n, a), MAX_STEPS)
    return _claude_turn(messages, tools) if provider == "Claude" else _openai_turn(messages, tools)


# ----------------------------------------------------------------------------
# Tools: thin wrappers around the app's own functions (via ctx)
# ----------------------------------------------------------------------------
def build_tools(ctx) -> dict:
    ss = st.session_state

    def view(t):
        return {"id": t["id"], "subject": t["subject"], "module": t["module"], "part": t["part"],
                "date": t["date"], "minutes": t["minutes"], "status": t["status"],
                "rescheduled_times": t["missed_count"]}

    def find_open_task(task_id):
        t = ctx.get_task(task_id)
        if t is None:
            return None, {"error": f"No task with id {task_id}. Call get_study_status for valid ids."}
        if t["status"] == "done":
            return None, {"error": "That task is already completed."}
        return t, None

    def get_study_status():
        tasks = ss.tasks
        if not tasks:
            return {"has_plan": False, "hint": "No plan yet. Use create_plan."}
        today = date.today().isoformat()
        tm = ss.timer
        return {
            "has_plan": True, "today": today,
            "today_tasks": [view(t) for t in tasks if t["date"] == today],
            "upcoming_tasks": [view(t) for t in tasks if t["date"] > today and t["status"] == "pending"][:12],
            "completed": sum(t["status"] == "done" for t in tasks), "total": len(tasks),
            "daily_hours": ctx.keep("hpd", 3.0),
            "timer": {"task_id": tm["id"], "seconds_left": max(0, int(tm["end"] - time.time()))} if tm else None,
        }

    def subjects_for(sem):
        branch = ctx.keep("branch", list(ctx.CURRICULUM)[0])
        ck = f"{branch}|{sem}"
        return branch, ck, ctx.CURRICULUM[branch].get(sem, []) + ss.custom.get(ck, [])

    def list_subjects(semester=None):
        sem = int(semester or ctx.keep("sem", 1))
        branch, _, options = subjects_for(sem)
        return {"branch": branch, "semester": sem, "subjects": options}

    def create_plan(subjects, semester=None, hours_per_module=2.0, difficulty=3, importance=3,
                    daily_hours=None, goal=None):
        sem = int(semester or ctx.keep("sem", 1))
        branch, ck, options = subjects_for(sem)
        chosen, unknown = [], []
        for name in subjects:
            low = name.lower()
            hit = next((o for o in options if o.lower() == low), None) or next((o for o in options if low in o.lower()), None)
            if not hit:
                unknown.append(name)
            elif hit not in chosen:
                chosen.append(hit)
        if not chosen:
            return {"error": "No matching subjects.", "valid_subjects": options}
        d = min(max(int(difficulty), 1), 5)
        i = min(max(int(importance), 1), 5)
        h = min(max(float(hours_per_module), 0.5), 40.0)

        ctx.remember("year", (sem + 1) // 2)
        ctx.remember("sem", sem)
        ctx.remember(f"subs|{ck}", chosen)
        if daily_hours:
            ctx.remember("hpd", min(max(float(daily_hours), 0.5), 16.0))
        if goal:
            ctx.remember("goal", goal)

        rows = []
        for s in chosen:
            ctx.remember(f"mods|{ck}|{s}", ctx.MODULES)
            for m in ctx.MODULES:
                ss.edits[(s, m)] = {"Topic (optional)": "", "Difficulty": d, "Importance": i,
                                    "Hours": h, "Deadline": pd.NaT}
                rows.append({"subject": s, "module": m, "topic": "", "difficulty": d,
                             "importance": i, "hours": h, "deadline": None})
        ok = ctx.gen_plan(rows)
        if not ok:
            msg = ss.flash[1] if ss.flash else "The plan could not be created."
            ss.flash = None
            return {"error": msg}
        return {"created": True, "semester": sem, "subjects": chosen, "not_found": unknown,
                "tasks": len(ss.tasks), "planner": ss.mode}

    def start_timer(task_id, minutes=None):
        if ss.timer:
            return {"error": "A timer is already running. Finish it first."}
        t, err = find_open_task(task_id)
        if err:
            return err
        mins = max(1, min(int(minutes or t["minutes"]), 300))
        ctx.start_timer(task_id, mins)
        return {"started": True, "minutes": mins, "task": view(t)}

    def complete_task(task_id):
        t, err = find_open_task(task_id)
        if err:
            return err
        tm = ss.timer
        if tm and tm["id"] == task_id:
            ctx.finish_timer(t, tm, completed=True)
        else:
            today = date.today().isoformat()
            t["status"], t["done_on"], t["actual"] = "done", today, t["minutes"]
            ss.used_today[today] = ss.used_today.get(today, 0) + t["minutes"]
            ctx.replan()
        return {"completed": True, "task": view(t)}

    def mark_not_finished(task_id):
        t, err = find_open_task(task_id)
        if err:
            return err
        tm = ss.timer
        if tm and tm["id"] == task_id:
            ctx.finish_timer(t, tm, completed=False)
        else:
            t["missed_count"] += 1
            t["priority"] += 3
            ctx.replan()
        return {"rescheduled": True, "task": view(t)}

    def replan_now():
        ctx.replan()
        return {"replanned": True}

    def set_daily_hours(hours):
        ctx.remember("hpd", min(max(float(hours), 0.5), 16.0))
        ctx.replan()
        return {"daily_hours": ctx.keep("hpd", 3.0)}

    def check_syllabus(subject, module=None):
        syl = getattr(ctx, "syl", None)
        ck = f"{ctx.keep('branch', list(ctx.CURRICULUM)[0])}|{ctx.keep('sem', 1)}"
        data = ss.get("syllabus", {}).get(ck)
        if syl is None or not data:
            return {"error": "No syllabus PDF is loaded for this semester. Ask the student to upload it "
                             "on the Subjects & Modules page."}
        info = syl.analyse_subject(data["id"], subject, data["bytes"])
        if not info:
            return {"error": "That subject was not found in the syllabus PDF. Try the course code."}
        mods = {f"Module {n}": {"page": m["page"], "text": m["text"]}
                for n, m in info["modules"].items() if module is None or n == int(module)}
        return {"start_page": info["start_page"], "modules": mods,
                "note": "Text is auto-extracted from the PDF and may be incomplete."}

    return {fn.__name__: fn for fn in (get_study_status, list_subjects, create_plan, start_timer,
                                       complete_task, mark_not_finished, replan_now, set_daily_hours,
                                       check_syllabus)}


# ----------------------------------------------------------------------------
# Voice (speech in, speech out)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the offline speech model (first time only)...")
def _local_whisper():
    from faster_whisper import WhisperModel

    return WhisperModel("base", compute_type="int8")


def transcribe_audio(raw: bytes) -> str:
    """Speech to text: OpenAI Whisper when OPENAI_API_KEY is set, else offline faster-whisper."""
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        return OpenAI().audio.transcriptions.create(model="whisper-1", file=("speech.wav", raw)).text.strip()
    try:
        model = _local_whisper()
    except ImportError:
        raise RuntimeError("voice input needs OPENAI_API_KEY or:  pip install faster-whisper")
    segments, _ = model.transcribe(io.BytesIO(raw))
    return " ".join(seg.text.strip() for seg in segments).strip()


def speak(text: str) -> None:
    """Read text aloud with the browser's built-in voice (no API, no cost)."""
    import streamlit.components.v1 as components

    clean = re.sub(r"[*_`#>|]", "", text)[:700]
    script = json.dumps(clean).replace("</", "<\\/")
    components.html(
        f"<script>const u=new SpeechSynthesisUtterance({script});u.lang='en-IN';"
        "speechSynthesis.cancel();speechSynthesis.speak(u);</script>", height=0)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
QUICK_PROMPTS = [
    "What should I study today?",
    "Plan my current semester's first two subjects",
    "Start the timer for my first task",
]


def render_chat_page(ctx) -> None:
    st.markdown("<div class='hero-badge'>AI study assistant</div>", unsafe_allow_html=True)
    st.title("💬 StudyMate Assistant")
    st.caption("Ask in plain English. The assistant can create your plan, start timers, mark modules done and re-plan.")

    providers = available_providers()
    if not providers:
        st.info("No assistant is available yet. For a free offline one: install Ollama (ollama.com), run "
                "`ollama pull qwen2.5:3b`, then refresh this page. Or set ANTHROPIC_API_KEY / OPENAI_API_KEY.")
        return
    provider = providers[0]
    if len(providers) > 1:
        provider = st.radio("Assistant", providers, horizontal=True, key="chat_provider")

    st.session_state.setdefault("voice_n", 0)
    pending_speech = st.session_state.pop("speak_text", None)
    if pending_speech:
        speak(pending_speech)

    if st.session_state.timer:
        ctx.timer_panel()

    history = st.session_state.chat
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = None
    if not history:
        for col, text in zip(st.columns(len(QUICK_PROMPTS)), QUICK_PROMPTS):
            if col.button(text, use_container_width=True):
                prompt = text
    col_a, col_b = st.columns([1, 3], vertical_alignment="center")
    col_a.toggle("🔊 Read replies aloud", key="speak_on")
    if hasattr(st, "audio_input"):
        with col_b:
            with st.container(key="voice_input_panel"):
                st.markdown(
                    "<div class='voice-head'>"
                    "<div><div class='voice-chip'><span class='voice-pulse'></span>VOICE INPUT</div>"
                    "<div class='voice-title'>Speak to StudyMate</div>"
                    "<div class='voice-subtitle'>Ask your study question aloud</div></div>"
                    "<div class='voice-wave' aria-hidden='true'>"
                    "<span></span><span></span><span></span><span></span><span></span><span></span><span></span>"
                    "</div></div>",
                    unsafe_allow_html=True,
                )
                rec = st.audio_input(
                    "Speak to StudyMate", key=f"voice_{st.session_state.voice_n}",
                    label_visibility="collapsed",
                )
        if rec is not None:
            st.session_state.voice_n += 1  # a new widget key clears the recording on the next run
            try:
                with st.spinner("Listening..."):
                    heard = transcribe_audio(rec.getvalue())
            except Exception as exc:
                st.error(f"Could not understand the recording: {exc}")
            else:
                if heard:
                    prompt = heard
                else:
                    st.warning("I couldn't hear anything. Please try again.")
    else:
        col_b.caption("Update Streamlit (pip install -U streamlit) to enable the microphone.")
    prompt = st.chat_input("Ask StudyMate...") or prompt
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                reply = run_chat_turn(history, prompt, build_tools(ctx), provider)
        except Exception as exc:
            st.error(f"The assistant hit a problem: {exc}")
            return
        st.markdown(reply)
    history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply or "Done."}]
    if st.session_state.get("speak_on") and reply:
        st.session_state["speak_text"] = reply  # spoken on the next run, after the rerun below
    st.session_state.flash = None  # tools may have queued banners the chat already explained
    st.rerun()  # refresh sidebar/pages with whatever the tools changed