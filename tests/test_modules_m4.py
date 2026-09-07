"""M4 — existence (§8), counting (§9), relation (§10) parsing and logic.

These tests cover what is settled regardless of how the D-042 degenerate-label
question is resolved: regexes, counts, number-word mapping, and the pure
counting/existence decision logic.
"""

import collections

import pytest

from src.data.loader import load_questions
from src.modules.counting import (
    EXPECTED_COUNT,
    NUMBER_WORDS,
    CountConfusion,
    load_number_questions,
    word_to_int,
)
from src.modules.counting import predict as count_predict
from src.modules.existence import EXIST_RE, best_score, parse_existence_questions
from src.modules.existence import predict as exist_predict
from src.modules.relation import REL_RE, TRD_REL_RE, parse_relation_questions


# ----------------------------- §10 relation --------------------------------
def test_trd_relation_regex_is_the_known_defect():
    """D-001: the regex as written in TRD §10 parses nothing. Kept testable."""
    qs = load_questions("discriminative-relation") + load_questions("relation")
    assert len(qs) == 1664
    assert sum(1 for q in qs if TRD_REL_RE.match(q.query)) == 0


def test_corrected_relation_regex_parses_all_1664():
    qs = load_questions("discriminative-relation") + load_questions("relation")
    assert sum(1 for q in qs if REL_RE.match(q.query)) == 1664


def test_parse_relation_questions():
    rs = parse_relation_questions()
    assert len(rs) == 1664
    r = next(x for x in rs if x.query == "Is there direct contact between the person and grass?")
    assert r.a == "person" and r.b == "grass"


# ----------------------------- §8 existence --------------------------------
def test_exist_regex_parses_all_4924():
    qs = load_questions("discriminative-hallucination")
    assert len(qs) == 4924
    assert sum(1 for q in qs if EXIST_RE.match(q.query)) == 4924


def test_parse_existence_questions():
    es = parse_existence_questions()
    assert len(es) == 4924
    assert es[0].obj and " " not in es[0].obj.strip()[:0] + ""


def test_existence_gold_is_constant_no():
    """D-042: every existence question's gold answer is 'no'.

    Documented as a test so the property cannot silently change, and so anyone
    reading the suite sees why existence accuracy is not interpretable alone.
    """
    golds = collections.Counter(e.truth for e in parse_existence_questions())
    assert golds["no"] == 4924
    assert golds["yes"] == 0


def test_best_score_on_empty_detections():
    assert best_score([]) == float("-inf")


def test_existence_predicts_yes_only_above_threshold():
    qs = parse_existence_questions()[:2]
    dets = {(qs[0].image, qs[0].obj): [[0, 0, 1, 1, 0.9]],
            (qs[1].image, qs[1].obj): [[0, 0, 1, 1, 0.05]]}
    recs = exist_predict(qs, lambda i, o: dets.get((i, o), []), threshold=0.5)
    assert recs[0]["pred"] == "yes" and recs[1]["pred"] == "no"


def test_existence_no_detection_sets_fell_back():
    qs = parse_existence_questions()[:1]
    recs = exist_predict(qs, lambda i, o: [], threshold=0.5)
    assert recs[0]["pred"] == "no"
    assert recs[0]["fell_back"] is True
    assert recs[0]["detector_score"] is None


# ----------------------------- §9 counting ---------------------------------
def test_number_words_are_the_nine_specified():
    assert set(NUMBER_WORDS) == {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    }


def test_word_to_int():
    assert word_to_int("three") == 3
    assert word_to_int("  Nine ") == 9


def test_unknown_number_word_raises():
    with pytest.raises(ValueError, match="unknown number word"):
        word_to_int("eleven")
    with pytest.raises(ValueError):
        word_to_int("0")


def test_number_questions_load_and_are_balanced():
    ns = load_number_questions()
    assert len(ns) == EXPECTED_COUNT == 2072
    golds = collections.Counter(n.truth for n in ns)
    assert golds["yes"] == golds["no"] == 1036


def test_counting_exact_match_is_yes():
    ns = load_number_questions()
    q = next(n for n in ns if word_to_int(n.num_word) == 2)
    recs = count_predict([q], lambda i, o: [[0, 0, 1, 1, 0.5]] * 2)
    assert recs[0]["pred"] == "yes"
    assert recs[0]["predicted_count"] == 2 and recs[0]["claimed_count"] == 2


def test_counting_mismatch_is_no():
    ns = load_number_questions()
    q = next(n for n in ns if word_to_int(n.num_word) == 2)
    recs = count_predict([q], lambda i, o: [[0, 0, 1, 1, 0.5]])
    assert recs[0]["pred"] == "no"
    assert recs[0]["predicted_count"] == 1


def test_counting_zero_detections_sets_fell_back():
    ns = load_number_questions()[:1]
    recs = count_predict(ns, lambda i, o: [])
    assert recs[0]["predicted_count"] == 0
    assert recs[0]["fell_back"] is True


def test_count_confusion_logging():
    c = CountConfusion()
    c.record(2, 2)
    c.record(2, 3)
    c.record(1, 0)
    d = c.as_dict()
    assert d["n"] == 3
    assert d["exact_match_rate"] == pytest.approx(1 / 3)
    assert d["mean_abs_error"] == pytest.approx(2 / 3)
    assert d["predicted_zero_rate"] == pytest.approx(1 / 3)
    assert d["confusion"]["claimed2_pred2"] == 1


def test_count_confusion_empty_is_not_computed():
    assert CountConfusion().as_dict()["mean_abs_error"] == "NOT_COMPUTED"
