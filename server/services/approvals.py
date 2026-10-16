"""Send approvals: code (not a prompt) decides whether an email may leave the account.

An email can only be sent when the user approved that exact content. The flow:

  1. A Gmail tool that would send something (or create a draft that could later be sent) records a PROPOSAL:
     the recipients, the text and a fingerprint of them.
  2. When the main agent shows that text to the user (`send_draft`), the proposal is marked SHOWN - only if the
     text shown really contains the proposal's recipients and body.
  3. The main agent may then call `approve_send`. The store grants approval only if the turn started from the
     USER's own message (not a worker result, a reminder or an email alert), that message reads as a
     confirmation, and the proposal was shown and has not expired.
  4. The send tool consumes the approval. It works once, for that content only.

Proposals and approvals are saved to disk, so a draft shown before a restart can still be approved after it (the
30-minute expiry uses wall-clock time, so it still applies).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
import threading
import time
import uuid
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

APPROVAL_TTL_SECONDS = 30 * 60
SHOWN_TOGETHER_SECONDS = 2.0  # drafts shown within this of each other count as shown in the same message
MAX_CONFIRMATION_CHARS = 160

_YES = re.compile(
    r"\b(yes|yeah|yep|yup|ok|okay|sure|send( it| them| that| this)?|go ahead|go for it|confirm(ed)?|do it|ship it|"
    r"approve[d]?|looks good|sounds good|perfect|please do|that works)\b",
    re.IGNORECASE,
)
# A message that also says no, wait or "change it" is not a clean confirmation: refuse and let the agent ask again.
_NOT_CLEAN = re.compile(
    r"\b(no|nope|don'?t|do not|stop|cancel|wait|hold on|hold off|not yet|never|change|edit|instead|rather|actually|"
    r"but|fix|rewrite|add|remove|different|first let|before you)\b",
    re.IGNORECASE,
)


_SEND_YES = re.compile(
    r"\b(yes|yeah|yep|yup|send( it| them| that| this)?|go ahead|confirm(ed)?|ship it|approve[d]?|do it)\b", re.IGNORECASE
)


def is_send_confirmation(text: str) -> bool:
    """Stricter than is_confirmation: an explicit "yes"/"send" (a casual "ok thanks" or "looks good" is not)."""
    return is_confirmation(text) and bool(_SEND_YES.search(text or ""))


def is_confirmation(text: str) -> bool:
    """True if a user message reads as a plain "yes, send it" (short, affirmative, no hedging)."""
    text = (text or "").strip()
    if not text or len(text) > MAX_CONFIRMATION_CHARS:
        return False
    return bool(_YES.search(text)) and not _NOT_CLEAN.search(text)


_CHOICE = re.compile(
    r"^\s*(#?\d{1,2}|option \d{1,2}|(the )?(first|second|third|last|short(er)?|long(er)?)( one)?)\b", re.IGNORECASE
)


def is_choice(text: str) -> bool:
    """A plain pick from a "which one?" question ("2", "the second one", "the short one"), with no hedging."""
    text = (text or "").strip()
    return 0 < len(text) <= 40 and bool(_CHOICE.match(text)) and not _NOT_CLEAN.search(text)


def _plain(text: str) -> str:
    """Compare text ignoring case, spacing and HTML tags."""
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").lower().split())


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def recipients_of(*values) -> List[str]:
    """Every address mentioned in the given strings / lists (to, cc, bcc, extras), lower-cased, de-duplicated."""
    found: List[str] = []
    for value in values:
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            for address in _EMAIL.findall(str(item or "")):
                if address.lower() not in found:
                    found.append(address.lower())
    return found


def _now() -> float:
    return time.time()


@dataclass
class Proposal:
    id: str
    kind: str  # "draft" | "reply" | "forward"
    recipients: List[str]
    subject: str
    body: str
    fingerprint: str
    draft_id: Optional[str] = None
    shown: bool = False
    approved: bool = False
    used: bool = False
    created_at: float = field(default_factory=lambda: _now())
    shown_at: float = 0.0          # when the user last saw it: "yes" means the draft shown last
    superseded: bool = False       # a newer draft to the same people replaced it ("make it shorter")

    def expired(self) -> bool:
        return _now() - self.created_at > APPROVAL_TTL_SECONDS

    def label(self) -> dict:
        return {"proposal_id": self.id, "kind": self.kind, "to": ", ".join(self.recipients), "subject": self.subject}


def _content_key(proposal: "Proposal") -> tuple:
    """What the user actually judges: who it goes to and what it says (forwards: also which message)."""
    extra = proposal.fingerprint if proposal.kind == "forward" else ""
    return (tuple(sorted(proposal.recipients)), _plain(proposal.body), extra)


def fingerprint(kind: str, recipients: Sequence[str], body: str, extra: str = "") -> str:
    payload = json.dumps([kind, sorted(recipients), _plain(body), extra or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- turn context
@dataclass(frozen=True)
class Turn:
    origin: str  # "user" or "agent"
    text: str
    drafts: List[Tuple[str, str, str]] = field(default_factory=list)  # (to, subject, body) shown in THIS turn


_current_turn: contextvars.ContextVar[Optional[Turn]] = contextvars.ContextVar("approval_turn", default=None)


def begin_turn(origin: str, text: str) -> contextvars.Token:
    """Called by the main agent runtime: who started this turn, and what did they say?"""
    return _current_turn.set(Turn(origin, text))


def end_turn(token: contextvars.Token) -> None:
    _current_turn.reset(token)


def current_turn() -> Optional[Turn]:
    return _current_turn.get()


# --------------------------------------------------------------------------- store
class ApprovalStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self._proposals: Dict[str, Proposal] = {}
        self._path = path
        self._load()

    # -- persistence
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            for item in json.loads(self._path.read_text(encoding="utf-8")):
                proposal = Proposal(**item)
                if not proposal.used and not proposal.expired():
                    self._proposals[proposal.id] = proposal
        except Exception:  # an unreadable file only means nothing is pending; the user is asked again
            self._proposals = {}

    def _save(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            live = [asdict(p) for p in self._proposals.values() if not p.used and not p.expired()]
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(live), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass

    # -- revisions
    def _supersede_older(self, newer: "Proposal") -> None:
        """A new draft to the same people replaces any older, not-yet-approved draft to them (a revision)."""
        if newer.kind not in ("draft", "shown"):
            return
        for other in self._proposals.values():
            if (other is not newer and other.kind in ("draft", "shown") and not other.approved and not other.used
                    and sorted(other.recipients) == sorted(newer.recipients)
                    and _content_key(other) != _content_key(newer)):
                other.used = other.superseded = True

    def _retire_previously_shown(self, current: "Proposal") -> None:
        """The user is now looking at `current`: earlier drafts (any recipient) that are still unsent are dropped,
        unless they were shown in the same message. Old drafts must not come back as "which one?" options."""
        for other in self._proposals.values():
            if (other is not current and other.kind in ("draft", "shown") and not other.used
                    and _content_key(other) != _content_key(current)
                    and other.shown_at and current.shown_at - other.shown_at > SHOWN_TOGETHER_SECONDS):
                other.used = other.superseded = True

    # -- recording what could be sent
    def propose(
        self,
        kind: str,
        *,
        recipients: Sequence[str],
        subject: str = "",
        body: str = "",
        draft_id: Optional[str] = None,
        extra: str = "",
    ) -> Proposal:
        """Record something that may be sent. An identical, still-open proposal is reused."""
        key = fingerprint(kind, recipients, body, extra or (draft_id or ""))
        with self._lock:
            for existing in self._proposals.values():
                if existing.fingerprint == key and not existing.used and not existing.expired():
                    return existing
            proposal = Proposal(
                id="p_" + uuid.uuid4().hex[:6], kind=kind, recipients=[r.lower() for r in recipients],
                subject=subject, body=body, fingerprint=key, draft_id=draft_id,
            )
            # The user may already have approved this exact text when the main agent showed a draft it wrote
            # itself: the approval carries over to the identical draft, once. Different text gets nothing.
            for approved in self._proposals.values():
                if (approved.kind == "shown" and approved.approved and not approved.used and not approved.expired()
                        and _content_key(approved) == _content_key(proposal)):
                    proposal.shown = proposal.approved = True
                    approved.used = True
                    break
            self._supersede_older(proposal)
            self._proposals[proposal.id] = proposal
            self._save()
            return proposal

    def record_shown(self, to_text: str, subject: str, body: str) -> Optional[Proposal]:
        """The user was shown this exact email (send_draft). Even before any agent has created it, the user can
        approve this content; an agent may then send exactly this text and nothing else."""
        recipients = recipients_of(to_text)
        if not recipients:
            return None
        key = fingerprint("shown", recipients, body)
        with self._lock:
            for existing in self._proposals.values():
                if existing.fingerprint == key and not existing.used and not existing.expired():
                    existing.shown_at = _now()
                    self._retire_previously_shown(existing)
                    self._save()
                    return existing
            shown = Proposal(id="p_" + uuid.uuid4().hex[:6], kind="shown", recipients=recipients, subject=subject,
                             body=body, fingerprint=key, shown=True, shown_at=_now())
            self._supersede_older(shown)
            self._retire_previously_shown(shown)
            self._proposals[shown.id] = shown
            self._save()
            return shown

    def mark_shown(self, to_text: str, body_text: str) -> List[Proposal]:
        """The user was just shown `to_text` / `body_text`. Mark every proposal that text honestly covers."""
        shown_to = (to_text or "").lower()
        shown_body = _plain(body_text)
        marked: List[Proposal] = []
        with self._lock:
            for proposal in self._proposals.values():
                if proposal.used or proposal.expired():
                    continue
                all_recipients_visible = all(r in shown_to for r in proposal.recipients)
                # Equal, not "contained in": a revision often extends the old text ("Now at 5 AM." vs "Work meeting is
                # now at 5 AM."), and containment marked the OLD draft as shown instead of recording the new one.
                body_visible = proposal.kind == "forward" or _plain(proposal.body) == shown_body
                if all_recipients_visible and body_visible:
                    proposal.shown = True
                    proposal.shown_at = _now()
                    marked.append(proposal)
            for proposal in marked:
                self._retire_previously_shown(proposal)
            if marked:
                self._save()
        return marked

    # -- approving
    def candidates(self) -> List[Proposal]:
        with self._lock:
            return [p for p in self._proposals.values() if p.shown and not p.approved and not p.used and not p.expired()]

    def approve(self, proposal_id: Optional[str] = None) -> Tuple[Optional[Proposal], str]:
        """Approve a shown proposal for the CURRENT turn. Returns (proposal, "") or (None, reason)."""
        turn = current_turn()
        if turn is None or turn.origin != "user":
            return None, "Approval can only come from the user's own message, not from a background event or a worker's result."
        # After a "which one?" question, the user's pick ("2") is their confirmation for that one draft.
        if not (is_confirmation(turn.text) or (proposal_id and is_choice(turn.text))):
            return None, "The user's latest message is not a clear confirmation. Ask them to confirm (for example 'yes, send it')."
        # Group by content: the text the main agent showed and the identical draft an agent created are ONE thing
        # to approve, not two (that ambiguity is what made the agent ask again and again).
        groups: Dict[tuple, List[Proposal]] = {}
        for candidate in self.candidates():
            groups.setdefault(_content_key(candidate), []).append(candidate)
        if proposal_id:
            groups = {k: v for k, v in groups.items() if any(p.id == proposal_id for p in v)}
        if not groups:
            # Already approved but not sent yet (e.g. the agent created the draft but has not sent it): the user
            # saying yes again should not be refused - point the agent at it instead.
            pending_send = [p for p in self._proposals.values()
                            if p.approved and not p.used and not p.expired() and p.kind != "shown"
                            and (not proposal_id or p.id == proposal_id)]
            if pending_send:
                return max(pending_send, key=lambda p: p.shown_at or p.created_at), ""
            return None, "There is no shown, unexpired draft waiting for approval. Show the exact draft to the user first."
        if len(groups) > 1 and not proposal_id:
            # "yes" answers the draft the user saw LAST. Only drafts shown together (one message) are ambiguous.
            latest = {key: max(p.shown_at for p in v) for key, v in groups.items()}
            newest = max(latest.values())
            groups = {k: v for k, v in groups.items() if newest - latest[k] <= SHOWN_TOGETHER_SECONDS}
        if len(groups) > 1:
            listing = "; ".join(f"{v[0].id}: to {', '.join(v[0].recipients)} - {v[0].subject or v[0].kind}" for v in groups.values())
            return None, f"More than one draft is waiting. Ask the user which one, then pass its proposal_id. Waiting: {listing}"
        group = next(iter(groups.values()))
        with self._lock:
            for proposal in group:
                proposal.approved = True
            self._save()
        return next((p for p in group if p.kind != "shown"), group[0]), ""

    # -- sending
    def consume(self, kind: str, *, recipients: Sequence[str], body: str = "", extra: str = "") -> bool:
        """Use up an approval for exactly this content. False if there is none."""
        wanted = fingerprint(kind, recipients, body, extra)
        with self._lock:
            for proposal in self._proposals.values():
                if proposal.fingerprint == wanted and proposal.approved and not proposal.used and not proposal.expired():
                    proposal.used = True
                    self._save()
                    return True
        return False

    def knows_draft(self, draft_id: str) -> bool:
        with self._lock:
            return any(p.draft_id == draft_id for p in self._proposals.values())

    def consume_draft(self, draft_id: str) -> bool:
        with self._lock:
            for proposal in self._proposals.values():
                if proposal.draft_id == draft_id and proposal.approved and not proposal.used and not proposal.expired():
                    proposal.used = True
                    self._save()
                    return True
        return False

    def clear(self) -> None:
        with self._lock:
            self._proposals = {}
            self._save()


_store = ApprovalStore(Path(__file__).resolve().parent.parent / "data" / "approvals.json")


def get_approval_store() -> ApprovalStore:
    return _store
