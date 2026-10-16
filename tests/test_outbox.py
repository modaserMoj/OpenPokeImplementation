"""The outbox's rules: windows by kind, one delivery for everything pending, Fibonacci cooldown for email alerts,
nothing dropped, deliveries never overlap. Timing is checked on a fake clock (exact) and on short real windows."""

import asyncio
import time

import pytest
from helpers import run

from server.services.outbox import COMBINED_HEADER, EMAIL, REMINDER, REQUESTED, Outbox, combine, Item

REAL_COOLDOWNS_MIN = (2, 3, 5, 8, 13, 21)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _manual(clock, **kw):
    delivered = []

    async def capture(payload):
        delivered.append(payload)

    box = Outbox(capture, clock=clock, auto_schedule=False, **kw)
    box.delivered = delivered
    return box


# ------------------------------------------------------------------ exact timing on a fake clock (real settings)
def test_windows_by_kind_match_the_design():
    clock = Clock()
    for kind, seconds in ((REQUESTED, 1), (REMINDER, 10), (EMAIL, 120)):
        box = _manual(clock)
        box.add(kind, "x")
        assert box.due_at() == clock.now + seconds


def test_the_earliest_window_decides_and_everything_pending_goes_together():
    clock = Clock()
    box = _manual(clock)
    box.add(EMAIL, "email alert")        # would wait 2 minutes
    clock.now += 5
    box.add(REQUESTED, "your answer")    # due in 1 second
    assert box.due_at() == clock.now + 1
    run(box.flush_now())
    assert len(box.delivered) == 1 and "email alert" in box.delivered[0] and "your answer" in box.delivered[0]


def test_the_window_does_not_restart_when_more_items_arrive():
    clock = Clock()
    box = _manual(clock)
    box.add(EMAIL, "first")
    opened = clock.now
    for _ in range(10):
        clock.now += 10
        box.add(EMAIL, "another")
    assert box.due_at() == opened + 120


def test_email_cooldowns_follow_fibonacci_minutes_capped_at_21():
    clock = Clock()
    box = _manual(clock)
    gaps = []
    for _ in range(8):  # a long steady stream of email alerts
        box.add(EMAIL, "alert")
        due = box.due_at()
        clock.now = due
        run(box.flush_now())
        gaps.append(box._next_email_at - clock.now)
    assert [round(g / 60) for g in gaps] == [2, 3, 5, 8, 13, 21, 21, 21]


def test_an_alert_waits_for_the_later_of_its_window_and_the_cooldown():
    clock = Clock()
    box = _manual(clock)
    for _ in range(4):  # the cooldown after this 4th email message is 8 minutes
        box.add(EMAIL, "alert")
        clock.now = box.due_at()
        run(box.flush_now())
    cooldown_end = clock.now + 8 * 60
    clock.now += 10
    box.add(EMAIL, "during the cooldown")
    assert box.due_at() == cooldown_end  # later than its own 2-minute window
    clock.now = cooldown_end + 60 * 60  # long after: the window decides again
    box2 = _manual(clock)
    box2.add(EMAIL, "fresh")
    assert box2.due_at() == clock.now + 120


def test_the_cooldown_resets_after_an_hour_of_quiet():
    clock = Clock()
    box = _manual(clock)
    for _ in range(4):
        box.add(EMAIL, "alert")
        clock.now = box.due_at()
        run(box.flush_now())
    assert box.current_email_cooldown() == 13 * 60  # 4 messages used 2, 3, 5, 8; the next gap would be 13
    clock.now += 61 * 60
    box.add(EMAIL, "after a quiet hour")
    assert box.current_email_cooldown() == 2 * 60


def test_reminders_and_requested_results_are_never_held_by_the_email_cooldown():
    clock = Clock()
    box = _manual(clock)
    for _ in range(5):  # push the email cooldown to 13 minutes
        box.add(EMAIL, "alert")
        clock.now = box.due_at()
        run(box.flush_now())
    box.add(REMINDER, "take pills")
    assert box.due_at() == clock.now + 10


def test_alerts_riding_along_with_a_reminder_do_not_advance_the_cooldown():
    clock = Clock()
    box = _manual(clock)
    box.add(EMAIL, "alert")
    box.add(REMINDER, "stand-up")
    clock.now = box.due_at()
    run(box.flush_now())
    assert box.current_email_cooldown() == 2 * 60 and box._next_email_at == float("-inf")


# ------------------------------------------------------------------ the payload
def test_a_single_item_is_passed_through_unchanged():
    assert combine([Item(REQUESTED, "[SUCCESS] Email Checker: Alice sent it", 0)]) == "[SUCCESS] Email Checker: Alice sent it"


def test_combined_items_are_ordered_by_importance():
    payload = combine([Item(EMAIL, "email", 0), Item(REMINDER, "reminder", 1), Item(REQUESTED, "answer", 2)])
    lines = payload.splitlines()
    assert lines[0] == COMBINED_HEADER and lines[1:] == ["1. answer", "2. reminder", "3. email"]


# ------------------------------------------------------------------ real (short) timing
def test_nothing_is_dropped_under_a_burst_of_mixed_items():
    delivered = []

    async def capture(payload):
        delivered.append(payload)

    box = Outbox(capture, windows={REQUESTED: 0.02, REMINDER: 0.05, EMAIL: 0.08}, email_cooldowns=(0.1,))

    async def scenario():
        for i in range(30):
            box.add((REQUESTED, REMINDER, EMAIL)[i % 3], f"item-{i:02d}")
            await asyncio.sleep(0.004)
        await box.drain()

    run(scenario())
    everything = "\n".join(delivered)
    assert all(everything.count(f"item-{i:02d}") == 1 for i in range(30))
    assert len(delivered) < 30  # and they were combined


def test_deliveries_never_overlap_and_items_added_meanwhile_go_next():
    spans = []

    async def slow_capture(payload):
        start = time.monotonic()
        await asyncio.sleep(0.1)
        spans.append((start, time.monotonic(), payload))

    box = Outbox(slow_capture, windows={REQUESTED: 0.01, REMINDER: 0.01, EMAIL: 0.01})

    async def scenario():
        box.add(REQUESTED, "first")
        await asyncio.sleep(0.05)       # the first delivery is in progress
        box.add(REQUESTED, "second")
        await box.drain()

    run(scenario())
    assert [p for _, _, p in spans] == ["first", "second"]
    assert spans[1][0] >= spans[0][1]  # the second started after the first finished


def test_the_email_watcher_goes_through_the_outbox_as_an_email_alert(outbox):
    from server.services.gmail.importance_watcher import ImportantEmailWatcher

    outbox._auto_schedule = False
    run(ImportantEmailWatcher()._dispatch_summary("Invoice from ACME due Friday"))
    assert [(i.kind, i.text) for i in outbox.pending()] == [
        (EMAIL, "Important email watcher notification:\nInvoice from ACME due Friday")]


def test_the_main_agent_gets_one_turn_for_combined_updates(llm, outbox):
    """End to end: the default delivery reaches the real main agent loop, once."""
    from server.services import outbox as outbox_module

    real = Outbox(windows={REQUESTED: 0.02, REMINDER: 0.02, EMAIL: 0.02})  # default delivery: the main agent
    from helpers import completion, tool_call
    llm.script(completion("", [tool_call("send_message_to_user", {"message": "9:00 - pills and stand-up."})]))

    async def scenario():
        real.add(REMINDER, "[SUCCESS] Pills: take pills")
        real.add(REMINDER, "[SUCCESS] Standup: stand-up call")
        await real.drain()

    run(scenario())
    assert len(llm.calls) == 1
    turn_text = llm.calls[0]["messages"][0]["content"]
    assert "take pills" in turn_text and "stand-up call" in turn_text and COMBINED_HEADER in turn_text
