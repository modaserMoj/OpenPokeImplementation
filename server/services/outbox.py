"""Outbox: the one place background results wait before they reach the main agent (issues 4 and 5).

Before, every finished task, reminder and email alert called the main agent immediately and separately, so three
reminders and five emails at 9:00 meant eight model turns and eight messages. Now each result is added here, and
the outbox decides when to deliver:

  * A window opens when the first item arrives; everything that arrives before it closes goes out together as ONE
    turn. The window length depends on the kind of item:
        requested  - the result of something the user asked for:  1 second (they are waiting)
        reminder   - a reminder that fired:                      10 seconds (reminders due together arrive together)
        email      - an important-email alert:                  2 minutes
  * When the outbox delivers, it takes EVERYTHING pending, whatever the kind, so a pending email alert rides along
    with a reminder instead of causing its own message.
  * Email alerts also have a Fibonacci cooldown between email-only deliveries: 2, 3, 5, 8, 13, 21 minutes (capped at
    21), reset after an hour without email alerts. A long stream of alerts is grouped more and more, while the first
    alerts of a busy period still come through quickly.
  * Nothing is dropped and deliveries never overlap. Pending items are also saved to disk until they have been
    delivered, so a crash or restart does not lose them (worst case after a crash mid-delivery: delivered twice).

The window never restarts when new items arrive, so a steady stream cannot postpone delivery forever.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

from ..logging_config import logger

REQUESTED = "requested"
REMINDER = "reminder"
EMAIL = "email"

WINDOW_SECONDS: Dict[str, float] = {REQUESTED: 1.0, REMINDER: 10.0, EMAIL: 120.0}
EMAIL_COOLDOWN_SECONDS: Sequence[float] = tuple(m * 60.0 for m in (2, 3, 5, 8, 13, 21))
COOLDOWN_RESET_SECONDS = 60 * 60.0

COMBINED_HEADER = (
    "Several updates arrived together. Tell the user about them in ONE message, most important first "
    "(results of things they asked for, then reminders, then email alerts):"
)
_PRIORITY = {REQUESTED: 0, REMINDER: 1, EMAIL: 2}

Deliver = Callable[[str], Awaitable[object]]


@dataclass
class Item:
    kind: str
    text: str
    created_at: float


def combine(items: Sequence[Item]) -> str:
    """One payload for the main agent. A single item is passed through unchanged."""
    if len(items) == 1:
        return items[0].text
    ordered = sorted(items, key=lambda item: (_PRIORITY.get(item.kind, 9), item.created_at))
    return "\n".join([COMBINED_HEADER] + [f"{n}. {item.text}" for n, item in enumerate(ordered, start=1)])


async def _deliver_to_interaction_agent(payload: str) -> None:
    from ..agents.interaction_agent.runtime import InteractionAgentRuntime

    await InteractionAgentRuntime().handle_agent_message(payload)


class Outbox:
    def __init__(
        self,
        deliver: Optional[Deliver] = None,
        *,
        windows: Optional[Dict[str, float]] = None,
        email_cooldowns: Optional[Sequence[float]] = None,
        reset_after: float = COOLDOWN_RESET_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        auto_schedule: bool = True,
        state_path: Optional[Path] = None,
    ) -> None:
        self._deliver = deliver or _deliver_to_interaction_agent
        self._windows = dict(WINDOW_SECONDS, **(windows or {}))
        self._cooldowns = tuple(email_cooldowns) if email_cooldowns is not None else EMAIL_COOLDOWN_SECONDS
        self._reset_after = reset_after
        self._clock = clock
        self._auto_schedule = auto_schedule
        self._state_path = state_path
        self._in_flight: List[Item] = []

        self._pending: List[Item] = []
        self._timer: Optional[asyncio.Task] = None
        self._timer_due: Optional[float] = None
        self._delivery_lock: Optional[asyncio.Lock] = None
        self._delivering = False

        self._email_step = 0
        self._next_email_at = float("-inf")
        self._last_email_at = float("-inf")
        self.deliveries = 0

    # ------------------------------------------------------------------ adding
    def add(self, kind: str, text: str) -> None:
        """Queue a result. Never blocks; delivery happens later on the event loop."""
        if kind not in self._windows:
            kind = REQUESTED
        now = self._clock()
        opens_email_group = kind == EMAIL and not any(item.kind == EMAIL for item in self._pending)
        if opens_email_group and now - self._last_email_at >= self._reset_after:
            self._email_step = 0  # quiet for long enough: the cooldown starts over at 2 minutes
        self._pending.append(Item(kind, text, now))
        self._persist()
        if self._auto_schedule:
            self._schedule()

    def pending(self) -> List[Item]:
        return list(self._pending)

    # ------------------------------------------------------------------ timing
    def due_at(self) -> Optional[float]:
        """When the pending items will be delivered (on this outbox's clock)."""
        dues = []
        for item in self._pending:
            due = item.created_at + self._windows[item.kind]
            if item.kind == EMAIL:
                due = max(due, self._next_email_at)
            dues.append(due)
        return min(dues) if dues else None

    def current_email_cooldown(self) -> float:
        return self._cooldowns[min(self._email_step, len(self._cooldowns) - 1)] if self._cooldowns else 0.0

    def _schedule(self) -> None:
        due = self.due_at()
        if due is None:
            return
        if self._timer and not self._timer.done():
            if self._timer_due is not None and self._timer_due <= due:
                return  # the existing timer fires first anyway
            self._timer.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("Outbox has no running event loop; items wait for the next flush")
            return
        self._timer_due = due
        self._timer = loop.create_task(self._wait_and_flush(due), name="outbox-flush")

    async def _wait_and_flush(self, due: float) -> None:
        try:
            await asyncio.sleep(max(0.0, due - self._clock()))
        except asyncio.CancelledError:
            return
        self._timer, self._timer_due = None, None
        await self.flush_now()

    # ------------------------------------------------------------------ delivering
    async def flush_now(self) -> None:
        """Deliver everything pending as one payload (deliveries are serialised)."""
        if self._delivery_lock is None:
            self._delivery_lock = asyncio.Lock()
        async with self._delivery_lock:
            if not self._pending:
                return
            batch, self._pending = self._pending, []
            self._in_flight = batch
            self._persist()  # still on disk until delivered
            now = self._clock()
            if any(item.kind == EMAIL for item in batch):
                self._last_email_at = now
                if all(item.kind == EMAIL for item in batch):  # email alerts caused this interruption
                    self._next_email_at = now + self.current_email_cooldown()
                    self._email_step += 1
            self.deliveries += 1
            self._delivering = True
            try:
                await self._deliver(combine(batch))
            except Exception as exc:  # pragma: no cover - the runtime already catches its own errors
                logger.error("Outbox delivery failed: %s", exc)
            finally:
                self._delivering = False
                self._in_flight = []
                self._persist()
        if self._pending and self._auto_schedule:
            self._schedule()  # items that arrived during the delivery

    # ------------------------------------------------------------------ persistence
    def _persist(self) -> None:
        if not self._state_path:
            return
        try:
            items = [{"kind": i.kind, "text": i.text} for i in self._in_flight + self._pending]
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(items), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as exc:
            logger.warning("Could not save the outbox: %s", exc)

    def restore(self) -> int:
        """Queue again whatever was still waiting when the server stopped. Call once at startup."""
        if not self._state_path or not self._state_path.exists():
            return 0
        try:
            items = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        for item in items:
            self.add(item.get("kind", REQUESTED), item.get("text", ""))
        if items:
            logger.info("Outbox: %d undelivered items restored after a restart", len(items))
        return len(items)

    async def drain(self) -> None:
        """Wait until every pending item has been delivered (respecting the windows). For tests and shutdown."""
        while self._pending or self._delivering or (self._timer and not self._timer.done()):
            await asyncio.sleep(0.005)


_outbox: Optional[Outbox] = None


def get_outbox() -> Outbox:
    global _outbox
    if _outbox is None:
        _outbox = Outbox(state_path=Path(__file__).resolve().parent.parent / "data" / "outbox.json")
    return _outbox
