"""Pruebas de integración para los endpoints REST y WebSocket de FDTD v2."""

import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_fdtd_validate_all_endpoint():
    """Verifica que GET /api/fdtd/validate retorne el diccionario completo de casos V1-V11."""
    res = client.get("/api/fdtd/validate")
    assert res.status_code == 200
    data = res.json()
    assert "V1" in data
    assert "V11" in data
    assert data["V1"]["is_verified"] is True
    assert data["V2"]["is_verified"] is True


def test_fdtd_validate_single_case():
    """Verifica GET /api/fdtd/validate/{case_id}."""
    res = client.get("/api/fdtd/validate/v3")
    assert res.status_code == 200
    data = res.json()
    assert data["case"] == "V3_tfsf_leakage"
    assert data["is_verified"] is True

    # Caso inexistente
    res404 = client.get("/api/fdtd/validate/v999")
    assert res404.status_code == 404


def test_fdtd_benchmarks():
    """Verifica endpoints analíticos de Airy y plasma."""
    r_airy = client.get("/api/fdtd/benchmarks/airy?eps_r=4.0&thickness_mm=10.0")
    assert r_airy.status_code == 200
    d_airy = r_airy.json()
    assert len(d_airy["frequencies_ghz"]) == 100
    assert "formula" in d_airy

    r_plasma = client.get("/api/fdtd/benchmarks/plasma?a_mm=10.0&wire_width_mm=1.0")
    assert r_plasma.status_code == 200
    d_plasma = r_plasma.json()
    assert d_plasma["plasma_frequency_ghz"] > 0.0


def test_fdtd_spectrum_post():
    """Verifica cálculo síncrono/aislado de espectro vía POST /api/fdtd/spectrum."""
    payload = {
        "dx_mm": 1.0,
        "courant_factor": 0.70,
        "pml_thickness": 12,
        "geometry": {
            "a_mm": 10.0,
            "d_mm": 3.0,
            "w_mm": 1.0,
            "periods": 2,
            "family": "rod_array",
            "relative_permittivity": 4.0,
            "conductivity": 0.0,
            "is_pec": False,
        },
        "max_steps": 400,
        "f_min_ghz": 6.0,
        "f_max_ghz": 14.0,
    }
    res = client.post("/api/fdtd/spectrum", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["frequencies_ghz"]) > 0
    assert len(data["transmission"]) == len(data["frequencies_ghz"])
    assert data["simulation_steps"] > 0


def test_fdtd_websocket_stream():
    """Verifica que el WebSocket /ws/fdtd acepte comandos y emita frames binarios y métricas."""
    with client.websocket_connect("/ws/fdtd") as ws:
        # Mensaje de bienvenida
        init_msg = json.loads(ws.receive_text())
        assert init_msg["type"] == "connection_ready"

        # Enviar start para simulación corta
        cfg = {
            "type": "start",
            "steps_per_frame": 4,
            "config": {
                "dx_mm": 1.0,
                "max_steps": 120,
                "pml_thickness": 10,
                "geometry": {
                    "a_mm": 10.0,
                    "d_mm": 2.0,
                    "w_mm": 1.0,
                    "periods": 1,
                    "family": "rod_array",
                },
            },
        }
        ws.send_text(json.dumps(cfg))

        # Recibir confirmación de inicio
        started = json.loads(ws.receive_text())
        assert started["type"] == "simulation_started"
        assert "grid" in started

        # Recibir al menos un frame de métricas + bytes binarios
        frame_msg = json.loads(ws.receive_text())
        assert frame_msg["type"] == "stream_frame"
        bin_frame = ws.receive_bytes()
        assert len(bin_frame) > 12 # Header Struct <IfHH (12 bytes) + data

        # Cancelar simulación
        ws.send_text(json.dumps({"type": "cancel"}))
