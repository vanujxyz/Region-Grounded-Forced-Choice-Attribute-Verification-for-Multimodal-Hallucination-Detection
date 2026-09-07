"""TRD §4 — contrastive pair construction.

AMBER's state and action questions occur in contrastive pairs over the same
(image, object): one true attribute and one false one.  That means the
forced-choice option set is supplied by the dataset; no antonym lexicon is
needed on the evaluation path.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from pydantic import BaseModel

from src.data.loader import load_questions

log = logging.getLogger(__name__)

STATE_RE = re.compile(r"^Is the (?P<obj>.+?) (?P<attr>.+?) in this image\?$")
ACTION_RE = re.compile(r"^Does the (?P<obj>.+?) (?P<attr>.+?) in this image\?$")
NUMBER_RE = re.compile(r"^(?:Are there|Is there) (?P<num>\w+) (?P<obj>.+?) in this image\?$")

STATE_QTYPE = "discriminative-attribute-state"
ACTION_QTYPE = "discriminative-attribute-action"
NUMBER_QTYPE = "discriminative-attribute-number"

# TRD §4 expected outcomes.
EXPECTED_PARSE = {STATE_QTYPE: 4764, ACTION_QTYPE: 792, NUMBER_QTYPE: 2072}
EXPECTED_PAIRS = {STATE_QTYPE: 2378, ACTION_QTYPE: 396}
EXPECTED_GROUPS = {STATE_QTYPE: 2380, ACTION_QTYPE: 396}


class AttrPair(BaseModel):
    image: str
    obj: str  # "sky"
    positive_attr: str  # "sunny"   (gold answer yes)
    negative_attr: str  # "gloomy"  (gold answer no)
    positive_id: int
    negative_id: int

    @property
    def options(self) -> list[str]:
        """The forced-choice option set, in a fixed content-independent order.

        Sorted alphabetically so option position cannot leak which attribute is
        the gold answer (TRD §4, TRD §16).
        """
        return sorted([self.positive_attr, self.negative_attr])


class NumberQuestion(BaseModel):
    """A number question.  Routed to counting.py, never to the attribute module."""

    id: int
    image: str
    num_word: str
    obj: str
    truth: str


def _parse(qtype: str, regex: re.Pattern[str]) -> list[tuple]:
    """Parse every question of ``qtype``; assert a 100% parse rate."""
    questions = load_questions(qtype)
    parsed, failures = [], []
    for q in questions:
        m = regex.match(q.query)
        if m is None:
            failures.append(q)
        else:
            parsed.append((q, m))
    if failures:
        sample = [f.query for f in failures[:5]]
        raise AssertionError(
            f"{regex.pattern!r} failed to parse {len(failures)}/{len(questions)} "
            f"{qtype} questions. Examples: {sample}"
        )
    expected = EXPECTED_PARSE[qtype]
    if len(parsed) != expected:
        raise AssertionError(
            f"{qtype}: expected {expected} parsed questions, got {len(parsed)}"
        )
    return parsed


def _build_pairs(qtype: str, regex: re.Pattern[str]) -> list[AttrPair]:
    parsed = _parse(qtype, regex)

    groups: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
    for q, m in parsed:
        groups[(q.image, m.group("obj"))].append((m.group("attr"), q.truth, q.id))

    pairs: list[AttrPair] = []
    discarded = 0
    for (image, obj), members in groups.items():
        truths = sorted(t for _, t, _ in members)
        if len(members) == 2 and truths == ["no", "yes"]:
            pos = next(m for m in members if m[1] == "yes")
            neg = next(m for m in members if m[1] == "no")
            pairs.append(
                AttrPair(
                    image=image,
                    obj=obj,
                    positive_attr=pos[0],
                    negative_attr=neg[0],
                    positive_id=pos[2],
                    negative_id=neg[2],
                )
            )
        else:
            discarded += 1

    log.info(
        "%s: %d groups -> %d clean pairs, %d discarded",
        qtype, len(groups), len(pairs), discarded,
    )

    expected_groups = EXPECTED_GROUPS[qtype]
    expected_pairs = EXPECTED_PAIRS[qtype]
    if len(groups) != expected_groups:
        raise AssertionError(
            f"{qtype}: expected {expected_groups} (image, obj) groups, got {len(groups)}"
        )
    if len(pairs) != expected_pairs:
        raise AssertionError(
            f"{qtype}: expected {expected_pairs} clean pairs, got {len(pairs)} "
            f"({discarded} discarded)"
        )

    # Deterministic order, independent of dict insertion order.
    pairs.sort(key=lambda p: p.positive_id)
    return pairs


def load_state_pairs() -> list[AttrPair]:
    """The 2,378 clean state pairs."""
    return _build_pairs(STATE_QTYPE, STATE_RE)


def load_action_pairs() -> list[AttrPair]:
    """The 396 clean action pairs."""
    return _build_pairs(ACTION_QTYPE, ACTION_RE)


def load_attr_pairs() -> list[AttrPair]:
    """All 2,774 attribute pairs (state + action).  Number is excluded by design."""
    return load_state_pairs() + load_action_pairs()


def load_number_questions() -> list[NumberQuestion]:
    """Number questions, unpaired, for counting.py (TRD §4, §9)."""
    parsed = _parse(NUMBER_QTYPE, NUMBER_RE)
    return [
        NumberQuestion(
            id=q.id, image=q.image, num_word=m.group("num"), obj=m.group("obj"), truth=q.truth
        )
        for q, m in parsed
    ]
