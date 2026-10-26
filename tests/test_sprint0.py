"""Pruebas del Sprint 0: verificaciones no tautológicas, /api/source e infraestructura WS."""

import json

from fastapi.testclient import TestClient

from domain.metamaterials import (
    MetamaterialUnitCell,
    compute_spectral_response,
    verify_bragg_bandgap,
    verify_energy_conservation_spectrum,
)
from main import app

client = TestClient(app)


def test_lossless_medium_conserves_rt_without_forcing():
    res = verify_energy_conservation_spectrum(MetamaterialUnitCell(a=10.0, d=2.0, w=1.0, periods=5))
    assert res["max_energy_deviation"] < 1e-9
    assert res["min_absorption_lossy"] > -1e-9


def test_absorption_is_not_clamped():
    """A = 1 - T - R sin recortes: con pérdidas debe ser estrictamente positiva en el espectro."""
    cell = MetamaterialUnitCell(a=10.0, d=2.0, w=1.0, periods=5, loss_factor=0.01)
    assert min(p.absorption for p in compute_spectral_response(cell)) > 0.0


def test_bragg_measured_against_real_stack():
    res = verify_bragg_bandgap(10.0, 2.2)
    assert res["relative_error_percent"] < 3.0
    assert res["min_transmission"] < 0.05


def test_source_endpoint_returns_executed_code():
    r = client.get("/api/source/fdtd2d.FDTD2DSimulator.step")
    assert r.status_code == 200
    body = r.json()
    assert "def step" in body["source"]
    assert body["endLine"] > body["startLine"]


def test_source_endpoint_rejects_non_whitelisted():
    assert client.get("/api/source/os.system").status_code == 404
    assert client.get("/api/source/domain.units").status_code == 404


def test_ch9_verify_ws_verdict_uses_numerical_result():
    with client.websocket_connect("/ws/physics") as ws:
        ws.send_text(json.dumps({"type": "ch9_fdtd_verify", "particles": []}))
        data = json.loads(ws.receive_text())
    res = data["fdtdVerifyResult"]
    assert res["is_all_verified"] == (
        res["cfl_stable_test"]["numerically_stable"]
        and res["cfl_stable_test"]["theorem_verified"]
        and res["cfl_unstable_test"]["theorem_verified"]
        and res["pulse_speed_test"]["is_verified"]
    )
