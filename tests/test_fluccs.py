"""FLUCCS reference: ring orientation (outer clockwise, holes counter-clockwise) and class mapping."""
import numpy as np

from fpcube.sources.fluccs import CLASSES, _rings_to_geom, classify


def test_rings_hole_and_two_parts():
    outer = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]          # clockwise
    hole = [(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]               # counter-clockwise
    other = [(20, 0), (20, 5), (25, 5), (25, 0), (20, 0)]
    g = _rings_to_geom([outer, hole, other])
    assert g.geom_type == "MultiPolygon" and abs(g.area - (100 - 4 + 25)) < 1e-9


def test_classify():
    names = list(CLASSES)
    c = classify(np.array([2100, 2110, 2130, 2140, 3100, 4110, 6170, 1400, 1650, 1100, 0]))
    assert [names[i] if i >= 0 else None for i in c] == ["pasture", "pasture", "pasture", "crops", "rangeland",
                                                          "forest", "wetland", "urban", "mining_reclaimed",
                                                          "rural_residential", None]


def test_every_land_code_has_one_class():
    codes = np.arange(1000, 9000)
    assert (classify(codes) >= 0).all()
    assert sum(len(v) for v in CLASSES.values()) == len(codes)
    names = list(CLASSES)
    c = classify(np.array([2210, 2240, 2420, 2510, 2520, 2610, 3210, 9990]))
    assert [names[i] if i >= 0 else None for i in c] == ["crops", "crops", "crops", "specialty_farms",
                                                          "specialty_farms", "rural_open", "rangeland", None]
