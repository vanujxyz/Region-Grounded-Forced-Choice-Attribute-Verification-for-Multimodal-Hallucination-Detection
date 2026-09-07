"""TRD §14 — test_pairs.py.

Asserts: regexes parse 4764/4764 state and 792/792 action; 2378 state pairs;
396 action pairs; options sorted alphabetically.
"""

import pytest

from src.data.loader import load_questions
from src.data.pairs import (
    ACTION_QTYPE,
    ACTION_RE,
    NUMBER_RE,
    STATE_QTYPE,
    STATE_RE,
    load_action_pairs,
    load_attr_pairs,
    load_number_questions,
    load_state_pairs,
)


def test_state_regex_parses_all_4764():
    qs = load_questions(STATE_QTYPE)
    assert len(qs) == 4764
    assert sum(1 for q in qs if STATE_RE.match(q.query)) == 4764


def test_action_regex_parses_all_792():
    qs = load_questions(ACTION_QTYPE)
    assert len(qs) == 792
    assert sum(1 for q in qs if ACTION_RE.match(q.query)) == 792


def test_number_regex_parses_all_2072():
    qs = load_questions("discriminative-attribute-number")
    assert len(qs) == 2072
    assert sum(1 for q in qs if NUMBER_RE.match(q.query)) == 2072


def test_state_pair_count_is_2378():
    assert len(load_state_pairs()) == 2378


def test_action_pair_count_is_396():
    assert len(load_action_pairs()) == 396


def test_pairs_are_one_yes_one_no():
    for p in load_attr_pairs():
        assert p.positive_id != p.negative_id
        assert p.positive_attr != p.negative_attr


def test_pair_ids_match_gold_answers():
    """The positive id's gold answer is yes and the negative id's is no."""
    truth = {q.id: q.truth for q in load_questions()}
    for p in load_attr_pairs():
        assert truth[p.positive_id] == "yes"
        assert truth[p.negative_id] == "no"


def test_options_sorted_alphabetically():
    """TRD §4/§16: option order must not leak which attribute is correct."""
    for p in load_attr_pairs():
        assert p.options == sorted(p.options)
        assert set(p.options) == {p.positive_attr, p.negative_attr}


def test_option_order_is_content_independent():
    """The positive attribute must land in position 0 only by alphabet, not design."""
    pairs = load_attr_pairs()
    first_is_positive = sum(1 for p in pairs if p.options[0] == p.positive_attr)
    # If order leaked the answer this would be 0 or len(pairs).
    assert 0 < first_is_positive < len(pairs)


def test_every_paired_id_is_unique():
    ids = [i for p in load_attr_pairs() for i in (p.positive_id, p.negative_id)]
    assert len(ids) == len(set(ids)) == 2 * (2378 + 396)


def test_number_questions_are_unpaired_and_routed_out():
    ns = load_number_questions()
    assert len(ns) == 2072
    assert {n.truth for n in ns} <= {"yes", "no"}
    # Number ids never appear among the attribute pairs.
    attr_ids = {i for p in load_attr_pairs() for i in (p.positive_id, p.negative_id)}
    assert not (attr_ids & {n.id for n in ns})


def test_number_words_are_the_nine_specified():
    """TRD §9: the complete measured set of number words."""
    expected = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
    assert {n.num_word for n in load_number_questions()} == expected


@pytest.mark.parametrize(
    "query,obj,attr",
    [
        ("Is the sky sunny in this image?", "sky", "sunny"),
        ("Is the mountain short in this image?", "mountain", "short"),
    ],
)
def test_state_regex_groups(query, obj, attr):
    m = STATE_RE.match(query)
    assert m.group("obj") == obj and m.group("attr") == attr


def test_action_regex_groups():
    m = ACTION_RE.match("Does the man sit in this image?")
    assert m.group("obj") == "man" and m.group("attr") == "sit"
