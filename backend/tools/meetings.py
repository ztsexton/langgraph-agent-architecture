"""
Meeting management tools.

These functions provide a simple in‑memory calendar interface.  They wrap an
instance of :class:`MeetingsManager` defined in ``backend/meetings.py`` to
expose CRUD operations as standalone tools.  Because the manager is
instantiated at module import time, the meeting list persists across calls.
"""

from typing import List, Optional

from ..meetings import MeetingsManager, Meeting

# Shared manager instance for all meeting tools.  This preserves state across
# invocations while avoiding multiple copies of the meeting list.
_manager = MeetingsManager()


def list_meetings() -> List[Meeting]:
    """Return the current list of meetings."""
    return _manager.list_meetings()


def create_meeting(title: str, date: str, agenda: str) -> Meeting:
    """Create a new meeting.

    Args:
        title: Title of the meeting.
        date: Date in YYYY‑MM‑DD format.
        agenda: Agenda notes.

    Returns:
        The created Meeting object.
    """
    return _manager.create_meeting(title=title, date=date, agenda=agenda)


def edit_meeting_agenda(meeting_id: int, new_agenda: str) -> Optional[Meeting]:
    """Edit the agenda of an existing meeting.

    Args:
    from .langfuse_tracing import traced_tool
        meeting_id: ID of the meeting.
        new_agenda: New agenda string.

    Returns:
        The updated Meeting object if found, otherwise None.
    """
    return _manager.edit_meeting_agenda(meeting_id, new_agenda)


def edit_meeting_notes(meeting_id: int, new_notes: str) -> Optional[Meeting]:
    """Edit the notes of an existing meeting.

    Args:
        meeting_id: ID of the meeting.
        new_notes: New notes string.

    Returns:
        The updated Meeting object if found, otherwise None.
    """
    return _manager.edit_meeting_notes(meeting_id, new_notes)
