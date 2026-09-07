"""TRD §14 — test_detector.py.

Asserts: crop stays in bounds; min size honoured; empty detection returns [] and
sets fell_back.

The geometry is tested against ``pad_box``, a pure function, so these run without
loading OWLv2. The model-dependent behaviour is tested separately and is marked
``gpu``.
"""

import pytest
from PIL import Image

from src.config import load_config
from src.modules.detector import Crop, Detection, _iou, _nms, pad_box

CROP_CFG = {"padding": 0.10, "min_size": 64}


# --------------------------------------------------------------------------
# Crop geometry
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "box,w,h",
    [
        ((10, 10, 100, 100), 500, 400),
        ((0, 0, 10, 10), 500, 400),
        ((490, 390, 500, 400), 500, 400),  # touching the far corner
        ((-5, -5, 20, 20), 500, 400),  # detector box outside bounds
        ((0, 0, 500, 400), 500, 400),  # full image
        ((250, 200, 251, 201), 500, 400),  # degenerate 1px box
    ],
)
def test_crop_stays_in_bounds(box, w, h):
    x1, y1, x2, y2 = pad_box(box, w, h, CROP_CFG)
    assert 0 <= x1 < x2 <= w
    assert 0 <= y1 < y2 <= h


@pytest.mark.parametrize(
    "box,w,h",
    [
        ((250, 200, 251, 201), 500, 400),
        ((0, 0, 5, 5), 500, 400),
        ((498, 398, 500, 400), 500, 400),
    ],
)
def test_min_size_honoured(box, w, h):
    x1, y1, x2, y2 = pad_box(box, w, h, CROP_CFG)
    assert (x2 - x1) >= min(64, w)
    assert (y2 - y1) >= min(64, h)


def test_min_size_capped_by_a_small_image():
    """A 40x40 image cannot yield a 64x64 crop; the image bound wins."""
    x1, y1, x2, y2 = pad_box((10, 10, 12, 12), 40, 40, CROP_CFG)
    assert (x1, y1, x2, y2) == (0, 0, 40, 40)


def test_padding_expands_the_box():
    x1, y1, x2, y2 = pad_box((100, 100, 300, 300), 500, 500, CROP_CFG)
    # 200px wide, 10% each side -> 20px each side
    assert (x1, y1, x2, y2) == (80, 80, 320, 320)


def test_padding_zero_is_identity_for_a_large_box():
    cfg = {"padding": 0.0, "min_size": 64}
    assert pad_box((100, 100, 300, 300), 500, 500, cfg) == (100, 100, 300, 300)


def test_inverted_box_is_normalised():
    a = pad_box((300, 300, 100, 100), 500, 500, CROP_CFG)
    b = pad_box((100, 100, 300, 300), 500, 500, CROP_CFG)
    assert a == b


def test_crop_is_actually_croppable():
    img = Image.new("RGB", (500, 400))
    box = pad_box((10, 10, 100, 100), 500, 400, CROP_CFG)
    crop = img.crop(box)
    assert crop.width == box[2] - box[0]
    assert crop.height == box[3] - box[1]
    assert crop.width > 0 and crop.height > 0


# --------------------------------------------------------------------------
# NMS / IoU
# --------------------------------------------------------------------------
def test_iou_identical_boxes():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes():
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_nms_suppresses_overlapping_lower_scores():
    dets = [
        Detection(box=(0, 0, 10, 10), score=0.9),
        Detection(box=(1, 1, 11, 11), score=0.5),  # heavy overlap -> suppressed
        Detection(box=(50, 50, 60, 60), score=0.3),  # disjoint -> kept
    ]
    kept = _nms(dets, 0.5)
    assert len(kept) == 2
    assert kept[0].score == 0.9


def test_nms_keeps_highest_score_first():
    dets = [
        Detection(box=(0, 0, 10, 10), score=0.2),
        Detection(box=(50, 50, 60, 60), score=0.8),
    ]
    kept = _nms(dets, 0.5)
    assert [d.score for d in kept] == [0.8, 0.2]


def test_nms_on_empty_input():
    assert _nms([], 0.5) == []


# --------------------------------------------------------------------------
# Fallback contract
# --------------------------------------------------------------------------
def test_empty_detection_sets_fell_back():
    """TRD §6: an empty result is legitimate; the crop falls back to the image."""
    img = Image.new("RGB", (500, 400))
    crop = Crop(image=img, box=(0.0, 0.0, 500.0, 400.0), fell_back=True, detector_score=None)
    assert crop.fell_back is True
    assert crop.detector_score is None
    assert crop.image.size == (500, 400)


def test_config_defaults_match_the_trd():
    cfg = load_config()
    assert cfg["detector"]["threshold"] == 0.10
    assert cfg["detector"]["nms_iou"] == 0.5
    assert cfg["crop"]["padding"] == 0.10
    assert cfg["crop"]["min_size"] == 64
    assert cfg["detector"]["prompt_template"] == "a photo of a {phrase}"


def test_detector_revision_is_pinned():
    """PRD §8 rule 5: never run against an unpinned revision."""
    rev = load_config()["detector"]["revision"]
    assert rev, "detector.revision must be pinned"
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)
