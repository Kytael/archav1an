"""The daemon's half of the control channel: write a request, read the acks.

The vocabulary is imported from the batch side rather than restated. Two
copies of the action list is two things to keep in step, and the batch is the
authority on what it can actually do. See the spec, 4.2.
"""
from tools.archive_batch.control import (
    ACTIONS, ControlError, new_id, read_acks, write_request)

# What the page shows. Enough to see the last few actions and their outcomes
# without turning the panel into a log viewer.
ACK_PREVIEW = 20


def submit(control_dir, action, now, **fields):
    """Write one request and return its id.

    Refuses an action the batch does not handle here rather than writing a file
    only to have it refused a second later: the operator gets the error in the
    HTTP response instead of in a panel they have to notice.
    """
    if action not in ACTIONS:
        raise ControlError(
            f"unknown action {action!r}; the batch handles "
            f"{', '.join(ACTIONS)}")
    request = dict(fields)
    request["id"] = new_id(now)
    request["action"] = action
    request["requested_at"] = now
    write_request(control_dir, request)
    return request["id"]


def acks(control_dir, limit=ACK_PREVIEW):
    """The tail of the ack log, oldest first."""
    return read_acks(control_dir, limit)
