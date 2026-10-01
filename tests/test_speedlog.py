import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import Dashboard, make_chart       # noqa: E402
from speedlog import SpeedRecord, SpeedStore, fit_model, load_records   # noqa: E402


def fake(n=400, street_effect=8.0, colour_effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        street = rng.choice(["Main St", "Kingsway"])
        colour = rng.choice(["white", "black", "red"], p=[.5, .35, .15])
        hour = rng.uniform(7, 19)
        sp = 40 + (street_effect if street == "Kingsway" else 0) \
            + (colour_effect if colour == "red" else 0) + rng.normal(0, 5)
        out.append(SpeedRecord("2026-01-01T00:00:00", hour, street, f"V{i}", colour, "car",
                               round(sp, 1), 50, int(sp > 55), "calibrated", 5.0))
    return out


def test_regression_finds_real_effect_and_ignores_null_colour():
    m = fit_model(fake())
    assert m["ready"]
    t = {x["name"]: x for x in m["terms"]}
    assert abs(t["street: Main St"]["beta"] + 8.0) < 3 * t["street: Main St"]["se"] + 0.5   # baseline is Kingsway or Main
    # colours have no true effect: estimates should be small relative to the street effect
    assert abs(t["colour: red"]["beta"]) < 3.5
    assert m["r2"] > 0.3


def test_regression_recovers_colour_effect_when_present():
    t = {x["name"]: x for x in fit_model(fake(1500, colour_effect=6.0, seed=3))["terms"]}
    assert t["colour: red"]["sig"] and abs(t["colour: red"]["beta"] - 6.0) < 1.5


def test_not_ready_with_little_data():
    assert fit_model(fake(10))["ready"] is False


def test_history_roundtrip(tmp_path):
    st = SpeedStore(tmp_path / "h.csv")
    for r in fake(5):
        st.add(r)
    assert len(SpeedStore(tmp_path / "h.csv").records) == 5
    assert load_records(tmp_path / "nope.csv") == []


def test_report_and_live_server(tmp_path):
    recs = fake(60)
    out = make_chart(recs, 50, tmp_path / "r.html", "Main St")
    html = out.read_text()
    assert "CarWatch speed dashboard" in html and '"limit": 50' in html
    assert "http://" not in html.replace("http://www.w3.org", "")           # no external resources
    d = Dashboard(port=8799)
    try:
        d.update(recs, 50, "Main St")
        data = json.loads(urllib.request.urlopen("http://127.0.0.1:8799/data.json").read())
        assert data["totals"]["total"] == 60 and data["model"]["ready"]
        assert b"canvas" in urllib.request.urlopen("http://127.0.0.1:8799/").read()
    finally:
        d.close()


def test_flags_feed_and_evidence_route_is_contained(tmp_path):
    ev = tmp_path / "evidence"
    (ev / "pkt").mkdir(parents=True)
    (ev / "pkt" / "summary.html").write_text("<p>ok</p>")
    (tmp_path / "secret.txt").write_text("nope")
    d = Dashboard(port=8798, evidence_dir=ev)
    try:
        flags = {"counts": {"WEAVING": 2}, "recent": [{"time": "2026-01-01 10:00:00", "vehicle": "V0001",
                                                        "type": "car", "flag": "WEAVING", "detail": "x", "evidence": "pkt"}]}
        d.update(fake(30), 50, "Main St", flags=flags, meta={"behaviour": "learned model"})
        data = json.loads(urllib.request.urlopen("http://127.0.0.1:8798/data.json").read())
        assert data["flags"]["counts"]["WEAVING"] == 2 and data["meta"]["behaviour"] == "learned model"
        assert urllib.request.urlopen("http://127.0.0.1:8798/evidence/pkt/summary.html").read() == b"<p>ok</p>"
        for bad in ("/evidence/../secret.txt", "/evidence/%2e%2e/secret.txt"):
            try:
                urllib.request.urlopen("http://127.0.0.1:8798" + bad)
                raise AssertionError("traversal served a file outside the evidence folder")
            except urllib.error.HTTPError as e:
                assert e.code == 404
    finally:
        d.close()
