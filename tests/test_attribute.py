"""TRD §14 — test_attribute.py.

Asserts: each cell returns exactly 2 predictions per pair; forced-choice cells
always emit exactly one `yes`.

The cell decision logic is exercised through ``decide``, a pure function, so
these run without loading SigLIP.
"""

import pytest

from src.config import load_config
from src.data.pairs import load_attr_pairs
from src.modules.attribute import (
    CELL_DECISION,
    CELL_REGION,
    CELLS,
    TieLog,
    decide,
    fit_tau,
)

K = 10.0


def _answers(options, scores, cell, tau=None, tie_log=None):
    return decide(options, scores, cell, tau, K, tie_log)[0]


# --------------------------------------------------------------------------
# Cell table matches the TRD §5 / §7 2x2
# --------------------------------------------------------------------------
def test_cell_table():
    assert CELLS == ("A", "B", "C", "D")
    assert CELL_REGION == {"A": "full", "B": "full", "C": "crop", "D": "crop"}
    assert CELL_DECISION == {
        "A": "threshold",
        "B": "forced_choice",
        "C": "threshold",
        "D": "forced_choice",
    }


# --------------------------------------------------------------------------
# Exactly two predictions per pair, one per question id
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", CELLS)
def test_each_cell_yields_two_answers_per_pair(cell):
    tau = 0.0 if CELL_DECISION[cell] == "threshold" else None
    answers = _answers(["gloomy", "sunny"], [-14.8, -9.0], cell, tau)
    assert len(answers) == 2
    assert set(answers) == {"gloomy", "sunny"}
    assert all(v in ("yes", "no") for v in answers.values())


# --------------------------------------------------------------------------
# Forced choice emits exactly one yes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", ["B", "D"])
@pytest.mark.parametrize(
    "scores",
    [[-14.8, -9.0], [9.0, -14.8], [0.0, 0.0], [-1.0, -1.000001], [100.0, -100.0]],
)
def test_forced_choice_emits_exactly_one_yes(cell, scores):
    answers = _answers(["gloomy", "sunny"], scores, cell)
    assert sorted(answers.values()) == ["no", "yes"]


@pytest.mark.parametrize("cell", ["B", "D"])
def test_forced_choice_picks_the_argmax(cell):
    answers = _answers(["gloomy", "sunny"], [-14.8, -9.0], cell)
    assert answers["sunny"] == "yes" and answers["gloomy"] == "no"
    answers = _answers(["gloomy", "sunny"], [-2.0, -9.0], cell)
    assert answers["gloomy"] == "yes" and answers["sunny"] == "no"


# --------------------------------------------------------------------------
# D-006: order invariance (regression guard) and tie accounting
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", ["B", "D"])
def test_forced_choice_is_invariant_to_option_order(cell):
    """Reversing the option list must not change any answer.

    Independent per-option scoring makes this true by construction; the test
    exists so it can never silently stop being true.
    """
    opts, scores = ["gloomy", "sunny"], [-14.8, -9.0]
    a = _answers(opts, scores, cell)
    b = _answers(list(reversed(opts)), list(reversed(scores)), cell)
    assert a == b


@pytest.mark.parametrize("cell", ["A", "C"])
def test_threshold_cells_are_invariant_to_option_order(cell):
    opts, scores = ["gloomy", "sunny"], [-14.8, -9.0]
    a = _answers(opts, scores, cell, tau=-12.0)
    b = _answers(list(reversed(opts)), list(reversed(scores)), cell, tau=-12.0)
    assert a == b


def test_confidences_are_invariant_to_option_order():
    _, ca = decide(["gloomy", "sunny"], [-14.8, -9.0], "D", None, K)
    _, cb = decide(["sunny", "gloomy"], [-9.0, -14.8], "D", None, K)
    assert ca == cb


def test_ties_are_counted_not_hidden():
    log = TieLog()
    _answers(["gloomy", "sunny"], [-5.0, -5.0], "D", tie_log=log)
    _answers(["gloomy", "sunny"], [-5.0, -6.0], "D", tie_log=log)
    d = log.as_dict()
    assert d["tie_count"] == 1
    assert d["comparisons"] == 2
    assert d["tie_rate"] == 0.5


def test_tie_rate_not_computed_when_empty():
    assert TieLog().as_dict()["tie_rate"] == "NOT_COMPUTED"


# --------------------------------------------------------------------------
# Threshold cells
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", ["A", "C"])
def test_threshold_cell_can_emit_two_yes_or_two_no(cell):
    """Unlike forced choice, independent thresholding may answer yes twice."""
    assert list(_answers(["a", "b"], [5.0, 6.0], cell, tau=0.0).values()) == ["yes", "yes"]
    assert list(_answers(["a", "b"], [-5.0, -6.0], cell, tau=0.0).values()) == ["no", "no"]


@pytest.mark.parametrize("cell", ["A", "C"])
def test_threshold_boundary_is_inclusive(cell):
    """TRD §7: yes iff score >= tau."""
    assert _answers(["a", "b"], [1.0, 0.999], cell, tau=1.0) == {"a": "yes", "b": "no"}


@pytest.mark.parametrize("cell", ["A", "C"])
def test_threshold_cell_without_tau_raises(cell):
    with pytest.raises(ValueError, match="tau"):
        _answers(["a", "b"], [1.0, 2.0], cell, tau=None)


def test_unknown_cell_raises():
    with pytest.raises(ValueError, match="unknown cell"):
        _answers(["a", "b"], [1.0, 2.0], "Z")


# --------------------------------------------------------------------------
# D-020: base-rate-matched tau for the A'/C' control cells
# --------------------------------------------------------------------------
def test_base_rate_tau_hits_the_target_count():
    from src.modules.attribute import fit_tau_base_rate

    data = [(float(i), "yes" if i >= 5 else "no") for i in range(10)]
    tau, info = fit_tau_base_rate(data)
    assert info["target_yes"] == 5
    assert info["realised_yes"] == 5
    assert sum(1 for s, _ in data if s >= tau) == 5
    assert info["selection"] == "base_rate_matched"


def test_base_rate_tau_reports_ties_rather_than_hiding_them():
    from src.modules.attribute import fit_tau_base_rate

    data = [(1.0, "yes"), (1.0, "yes"), (1.0, "no"), (0.0, "no")]
    tau, info = fit_tau_base_rate(data)
    assert info["target_yes"] == 2
    assert info["realised_yes"] == 3  # three scores tie at tau
    assert info["ties_at_tau"] == 1


def test_base_rate_tau_may_be_worse_than_accuracy_maximising():
    """A' trades accuracy for a correct base rate. That is the point."""
    from src.modules.attribute import fit_tau, fit_tau_base_rate

    data = [(3.0, "yes"), (2.0, "no"), (1.0, "no"), (0.0, "no")]
    _, best = fit_tau(data)
    _, info = fit_tau_base_rate(data)
    assert info["fit_accuracy"] <= best


def test_base_rate_tau_explicit_target():
    from src.modules.attribute import fit_tau_base_rate

    data = [(float(i), "no") for i in range(10)]
    _, info = fit_tau_base_rate(data, target_yes=3)
    assert info["realised_yes"] == 3


def test_base_rate_tau_rejects_bad_target():
    from src.modules.attribute import fit_tau_base_rate

    with pytest.raises(ValueError):
        fit_tau_base_rate([(1.0, "yes")], target_yes=0)
    with pytest.raises(ValueError):
        fit_tau_base_rate([(1.0, "yes")], target_yes=5)


def test_diag_cells_are_not_part_of_the_2x2():
    from src.modules.attribute import ALL_CELLS, DIAG_CELLS

    assert DIAG_CELLS == ("Ap", "Cp")
    assert set(DIAG_CELLS).isdisjoint(CELLS)
    assert ALL_CELLS == ("A", "B", "C", "D", "Ap", "Cp")


def test_diag_cells_route_like_their_parents():
    from src.modules.attribute import decision_of, region_of

    assert region_of("Ap") == region_of("A") == "full"
    assert region_of("Cp") == region_of("C") == "crop"
    assert decision_of("Ap") == decision_of("Cp") == "threshold"


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------
def test_forced_choice_confidence_is_softmax_of_the_winner():
    _, conf = decide(["gloomy", "sunny"], [-14.8, -9.0], "D", None, K)
    assert conf["sunny"] == conf["gloomy"]  # both carry the winner's probability
    assert 0.5 < conf["sunny"] <= 1.0


def test_threshold_confidence_is_sigmoid_around_tau():
    _, conf = decide(["a"], [0.0], "A", 0.0, K)
    assert conf["a"] == pytest.approx(0.5)


def test_all_confidences_in_unit_interval():
    for cell, tau in [("A", 0.0), ("C", 0.0), ("B", None), ("D", None)]:
        _, conf = decide(["a", "b"], [3.0, -2.0], cell, tau, K)
        assert all(0.0 <= v <= 1.0 for v in conf.values())


# --------------------------------------------------------------------------
# tau fitting
# --------------------------------------------------------------------------
def test_fit_tau_finds_a_perfect_separator():
    data = [(-10.0, "no"), (-9.0, "no"), (1.0, "yes"), (2.0, "yes")]
    tau, acc = fit_tau(data)
    assert acc == 1.0
    assert -9.0 < tau <= 1.0


def test_fit_tau_on_unseparable_data_reports_real_accuracy():
    data = [(0.0, "yes"), (0.0, "no")]
    _, acc = fit_tau(data)
    assert acc == 0.5


def test_fit_tau_empty_raises():
    with pytest.raises(ValueError):
        fit_tau([])


# --------------------------------------------------------------------------
# Config contract
# --------------------------------------------------------------------------
def test_prompt_template_is_shared_and_from_config():
    """TRD §16: the template must never vary between cells."""
    cfg = load_config()
    assert cfg["attribute"]["prompt_template"] == "a photo of a {attr} {obj}"
    assert cfg["attribute"]["confidence_k"] == 10.0


def test_attribute_revision_is_pinned():
    rev = load_config()["attribute"]["revision"]
    assert rev and len(rev) == 40


def test_real_pairs_feed_the_cells_cleanly():
    """Every real pair produces two options the cells can consume."""
    for p in load_attr_pairs()[:200]:
        assert len(p.options) == 2
        answers = _answers(p.options, [1.0, 2.0], "D")
        assert sorted(answers.values()) == ["no", "yes"]
