"""Pruebas de integración para los endpoints REST del Laboratorio Paramétrico (Capítulo 11)."""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_api_sweep_1d():
    """Verifica POST /api/sweep/1d con corrida rápida sobre 2 puntos."""
    payload = {
        "parameter_name": "w_mm",
        "param_values": [0.75, 1.25],
        "base_a_mm": 10.0,
        "base_d_mm": 5.0,
        "base_w_mm": 1.0,
        "family": "square_ring",
        "periods": 1,
        "is_pec": True,
        "dx_mm": 1.0,
        "f0_ghz": 10.0,
    }
    res = client.post("/api/sweep/1d", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["parameter_name"] == "w_mm"
    assert len(data["table_rows"]) == 2
    assert "best_design" in data
    assert data["best_design"]["rank"] == 1


def test_api_sweep_export_csv():
    """Verifica descarga directa de CSV vía POST /api/sweep/export/csv."""
    payload = {
        "parameter_name": "w_mm",
        "param_values": [0.5, 1.0],
        "base_a_mm": 10.0,
        "base_d_mm": 5.0,
        "family": "square_ring",
        "periods": 1,
        "dx_mm": 1.0,
    }
    res = client.post("/api/sweep/export/csv", json=payload)
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "Ranking,Design_ID,Familia" in res.text


def test_api_sweep_robustness():
    """Verifica GET /api/sweep/robustness."""
    res = client.get("/api/sweep/robustness")
    assert res.status_code == 200
    data = res.json()
    assert data["spearman_rank_correlation"] >= 0.90
    assert data["is_robust"] is True
