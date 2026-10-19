"""Pruebas de integración para endpoints REST y WebSocket de la API de física."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_and_constants(client):
    """Verifica endpoints informativos y constantes universales."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}

    const_resp = client.get("/api/physics/constants")
    assert const_resp.status_code == 200
    data = const_resp.json()
    assert "coulomb_constant" in data
    assert "speed_of_light" in data
    assert data["speed_of_light"]["value"] == 299792458.0


def test_rest_coulomb(client):
    """Verifica cálculo REST de fuerzas de Coulomb."""
    payload = [
        {"id": "p1", "position": {"x": 0.0, "y": 0.0}, "charge": 1e-6, "label": "q1"},
        {"id": "p2", "position": {"x": 0.05, "y": 0.0}, "charge": -2e-6, "label": "q2"},
    ]
    resp = client.post("/api/physics/coulomb", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "coulomb_result"
    assert len(data["forces"]) == 2  # F12 y F21
    f12 = data["forces"][0]
    assert f12["distance"] == 0.05
    assert f12["isAttractive"] is True


def test_rest_trajectory_simulation(client):
    """Verifica cálculo REST de simulación de trayectoria (Capítulo 5)."""
    payload = {
        "query": {
            "dynamicParticle": {
                "particle": {
                    "id": "e1",
                    "position": {"x": 0.0, "y": 0.0},
                    "charge": -1.6e-19,
                    "label": "e-",
                },
                "mass": 9.1e-31,
                "velocity": {"x": 1e5, "y": 0.0},
            },
            "uniformFields": [
                {
                    "fieldVector": {"x": 0.0, "y": 500.0},
                    "xBounds": [-0.1, 0.1],
                    "yBounds": [-0.1, 0.1],
                }
            ],
            "totalTime": 0.0001,
            "dt": 0.00001,
        },
        "sources": [
            {"id": "src1", "position": {"x": 0.05, "y": 0.05}, "charge": 1e-9, "label": "+Q"}
        ],
    }

    resp = client.post("/api/physics/trajectory", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "trajectory_result"
    assert data["trajectory"] is not None
    assert len(data["trajectory"]) > 5
    # La partícula debe haberse desplazado en x y deflactado
    first_pt = data["trajectory"][0]
    last_pt = data["trajectory"][-1]
    assert last_pt["time"] > first_pt["time"]
    assert last_pt["position"]["x"] > first_pt["position"]["x"]


def test_websocket_realtime_trajectory(client):
    """Verifica el flujo interactivo por WebSocket para trayectoria dinámica (Capítulo 5)."""
    with client.websocket_connect("/ws/physics") as ws:
        request_msg = {
            "type": "trajectory_sim",
            "particles": [],
            "dimensions": 2,
            "trajectoryQuery": {
                "dynamicParticle": {
                    "particle": {
                        "id": "p_movil",
                        "position": {"x": 0.0, "y": 0.0},
                        "charge": 1e-6,
                        "label": "q+",
                    },
                    "mass": 1e-6,
                    "velocity": {"x": 2.0, "y": 1.0},
                },
                "uniformFields": [
                    {
                        "fieldVector": {"x": 0.0, "y": -50.0},
                    }
                ],
                "totalTime": 0.01,
                "dt": 0.001,
            },
        }
        ws.send_json(request_msg)
        resp_data = ws.receive_json()

        assert resp_data["type"] == "trajectory_result"
        assert "trajectory" in resp_data
        assert len(resp_data["trajectory"]) == 11  # 0.01 / 0.001 + 1 paso inicial
        assert resp_data["computeTimeMs"] >= 0
