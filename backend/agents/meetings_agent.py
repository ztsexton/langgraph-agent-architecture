from __future__ import annotations

from ..tools.agent_config import get_agent_settings
from ..tools.langfuse_tracing import end_span, start_span
from ..tools.llm import ask_llm, get_llm
from ..tools.meetings import (
    create_meeting,
    edit_meeting_agenda,
    edit_meeting_notes,
    list_meetings,
)

from .types import AgentState
from .ui import a2ui_text


def meetings_agent(state: AgentState) -> AgentState:
    """Handle simple meeting operations based on the user input."""
    _span = start_span(name="agent:meetings_agent", input={"state": state}, metadata={"kind": "agent"})
    meetings_settings = get_agent_settings("meetings_agent")
    text = state.get("input", "").lower()

    if "list" in text and "meeting" in text:
        meetings = list_meetings()
        if not meetings:
            msg = "There are no meetings scheduled."
            out = {"output": msg, "a2ui": a2ui_text("Meetings", msg)}
            end_span(_span, output=out)
            return out

        lines = [
            f"{m.id}. {m.title} on {m.date} – agenda: {m.agenda}; notes: {m.notes or 'None'}"
            for m in meetings
        ]
        raw = "\n".join(lines)
        llm = get_llm(meetings_settings.model_name)
        if llm:
            try:
                out = {
                    "output": ask_llm(
                        f"Format these meeting entries for a user:\n\n{raw}",
                        model_name=meetings_settings.model_name,
                        system_prompt=meetings_settings.system_prompt,
                    ),
                    "a2ui": a2ui_text("Meetings", raw),
                }
                end_span(_span, output=out)
                return out
            except Exception:
                pass

        out = {"output": raw, "a2ui": a2ui_text("Meetings", raw)}
        end_span(_span, output=out)
        return out

    if "create" in text and "meeting" in text:
        title = "Untitled meeting"
        date = "2026-01-17"
        agenda = ""
        if "agenda" in text:
            parts = text.split("agenda", 1)
            agenda = parts[1].strip()
            text_before_agenda = parts[0]
        else:
            text_before_agenda = text
        tokens = text_before_agenda.split()
        if "on" in tokens:
            idx = tokens.index("on")
            if idx + 1 < len(tokens):
                date = tokens[idx + 1]
        try:
            meeting_idx = tokens.index("meeting")
            if "on" in tokens:
                on_idx = tokens.index("on")
                title_tokens = tokens[meeting_idx + 1 : on_idx]
            else:
                title_tokens = tokens[meeting_idx + 1 :]
            if title_tokens:
                title = " ".join(title_tokens).title()
        except ValueError:
            pass
        meeting = create_meeting(title=title, date=date, agenda=agenda)
        raw = f"Created meeting {meeting.id}: {meeting.title} on {meeting.date}."
        llm = get_llm(meetings_settings.model_name)
        if llm:
            try:
                out = {
                    "output": ask_llm(
                        f"User requested to create a meeting. Result: {raw}",
                        model_name=meetings_settings.model_name,
                        system_prompt=meetings_settings.system_prompt,
                    )
                }
                end_span(_span, output=out)
                return out
            except Exception:
                pass
        out = {"output": raw, "a2ui": a2ui_text("Meetings", raw)}
        end_span(_span, output=out)
        return out

    if "edit" in text and "agenda" in text:
        tokens = text.split()
        meeting_id = None
        for token in tokens:
            if token.isdigit():
                meeting_id = int(token)
                break
        new_agenda = text.split("agenda", 1)[1].strip()
        if meeting_id is None:
            out = {"output": "Please specify the meeting ID to edit."}
            end_span(_span, output=out)
            return out
        meeting = edit_meeting_agenda(meeting_id, new_agenda)
        if meeting is None:
            out = {"output": f"Meeting {meeting_id} not found."}
            end_span(_span, output=out)
            return out
        raw = f"Updated agenda for meeting {meeting.id}."
        llm = get_llm(meetings_settings.model_name)
        if llm:
            try:
                out = {
                    "output": ask_llm(
                        f"User requested to update a meeting agenda. Result: {raw}",
                        model_name=meetings_settings.model_name,
                        system_prompt=meetings_settings.system_prompt,
                    )
                }
                end_span(_span, output=out)
                return out
            except Exception:
                pass
        out = {"output": raw, "a2ui": a2ui_text("Meetings", raw)}
        end_span(_span, output=out)
        return out

    if "edit" in text and "notes" in text:
        tokens = text.split()
        meeting_id = None
        for token in tokens:
            if token.isdigit():
                meeting_id = int(token)
                break
        new_notes = text.split("notes", 1)[1].strip()
        if meeting_id is None:
            out = {"output": "Please specify the meeting ID to edit."}
            end_span(_span, output=out)
            return out
        meeting = edit_meeting_notes(meeting_id, new_notes)
        if meeting is None:
            out = {"output": f"Meeting {meeting_id} not found."}
            end_span(_span, output=out)
            return out
        raw = f"Updated notes for meeting {meeting.id}."
        llm = get_llm(meetings_settings.model_name)
        if llm:
            try:
                out = {
                    "output": ask_llm(
                        f"User requested to update meeting notes. Result: {raw}",
                        model_name=meetings_settings.model_name,
                        system_prompt=meetings_settings.system_prompt,
                    )
                }
                end_span(_span, output=out)
                return out
            except Exception:
                pass
        out = {"output": raw, "a2ui": a2ui_text("Meetings", raw)}
        end_span(_span, output=out)
        return out

    out = {
        "output": (
            "I can manage meetings. Try commands like 'list meetings', 'create meeting "
            "Team Sync on 2026-02-20 agenda Discuss progress', 'edit meeting 1 agenda New agenda' or "
            "'edit meeting 1 notes New notes'."
        ),
        "a2ui": a2ui_text(
            "Meetings",
            "I can manage meetings. Try: list meetings; create meeting Team Sync on 2026-02-20 agenda ...; edit meeting 1 agenda ...; edit meeting 1 notes ...",
        ),
    }
    end_span(_span, output=out)
    return out
