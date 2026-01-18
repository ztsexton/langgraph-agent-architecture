"""
Module defining a simple in‑memory meetings manager.

This module exposes a ``MeetingsManager`` class used by the multi‑agent
system.  It implements a handful of CRUD‑style operations that
specialised agents can call.  The manager is intentionally simplistic:
all meetings are stored in a Python dictionary and will be lost when
the server restarts.  Each meeting record contains an ``id``, a
``title``, a ``date`` (ISO format string), an ``agenda`` and any
associated ``notes``.

The functions defined here avoid external dependencies or API calls.
They can easily be swapped out for a persistent data store or real
calendar integration in a production system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Meeting:
    """Dataclass representing a single meeting."""

    id: int
    title: str
    date: str
    agenda: str
    notes: str = ""


class MeetingsManager:
    """Simple manager for storing and manipulating meeting objects."""

    def __init__(self) -> None:
        # Internal counter used to assign incremental IDs
        self._counter: int = 1
        # In‑memory store of meetings keyed by id
        self._meetings: Dict[int, Meeting] = {}

    def list_meetings(self) -> List[Meeting]:
        """Return a list of all meetings currently stored."""
        return list(self._meetings.values())

    def create_meeting(self, title: str, date: str, agenda: str) -> Meeting:
        """Create a new meeting and return it.

        Args:
            title: Human readable meeting title.
            date: ISO formatted date string (e.g. ``"2026-01-17"``).
            agenda: Description of agenda items.

        Returns:
            The newly created ``Meeting`` instance.
        """
        meeting = Meeting(id=self._counter, title=title, date=date, agenda=agenda)
        self._meetings[self._counter] = meeting
        self._counter += 1
        return meeting

    def edit_meeting_agenda(self, meeting_id: int, new_agenda: str) -> Optional[Meeting]:
        """Update the agenda of an existing meeting.

        Args:
            meeting_id: Identifier of the meeting to update.
            new_agenda: Replacement agenda text.

        Returns:
            The updated meeting or ``None`` if not found.
        """
        meeting = self._meetings.get(meeting_id)
        if meeting is None:
            return None
        meeting.agenda = new_agenda
        return meeting

    def edit_meeting_notes(self, meeting_id: int, new_notes: str) -> Optional[Meeting]:
        """Update the notes of an existing meeting.

        Args:
            meeting_id: Identifier of the meeting to update.
            new_notes: Replacement notes text.

        Returns:
            The updated meeting or ``None`` if not found.
        """
        meeting = self._meetings.get(meeting_id)
        if meeting is None:
            return None
        meeting.notes = new_notes
        return meeting

    def get_meeting(self, meeting_id: int) -> Optional[Meeting]:
        """Retrieve a single meeting by its identifier.

        Args:
            meeting_id: Identifier of the meeting to return.

        Returns:
            The meeting instance if found, otherwise ``None``.
        """
        return self._meetings.get(meeting_id)


__all__ = ["MeetingsManager", "Meeting"]
