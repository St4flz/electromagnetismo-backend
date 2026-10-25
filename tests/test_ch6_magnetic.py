"""Pruebas unitarias para el Capítulo 6: Magnetismo, Biot-Savart y Fuerza de Lorentz."""

import math
import pytest
from domain.vectors import Vector2D
from domain.magnetic import (
    biot_savart_point_charge_2d,
    biot_savart_wire_field_2d,
    lorentz_force_2d,
    cyclotron_parameters,
    verify_cyclotron_physics,
    verify_wien_filter,
)


def test_biot_savart_point_charge():
    # Carga positiva en el origen moviéndose hacia +x
    # En (0, 0.1), r apunta hacia +y => (v x r)_z = vx*ry > 0 => Bz > 0
    q = 1e-6
    v = Vector2D(100.0, 0.0)
    pos = Vector2D(0.0, 0.0)
    eval_p = Vector2D(0.0, 0.1)

    bz = biot_savart_point_charge_2d(q, v, pos, eval_p)
    assert bz > 0.0

    # Si se invierte la velocidad, Bz debe invertir su signo
    bz_inv = biot_savart_point_charge_2d(q, Vector2D(-100.0, 0.0), pos, eval_p)
    assert math.isclose(bz, -bz_inv, rel_tol=1e-5)


def test_biot_savart_wire_field_direction():
    current = 10.0  # [A]
    wire_pos = Vector2D(0.0, 0.0)
    eval_p = Vector2D(0.1, 0.0)

    # En (x > 0, y = 0), para un alambre saliente +z, B apunta en +y
    b_vec = biot_savart_wire_field_2d(current, wire_pos, eval_p)
    assert math.isclose(b_vec.x, 0.0, abs_tol=1e-9)
    assert b_vec.y > 0.0


def test_cyclotron_physics_verification():
    # Comprobar la simulación numérica de Boris contra los valores analíticos
    mass = 1e-6
    charge = 1e-6
    speed = 50.0
    Bz = 0.5

    result = verify_cyclotron_physics(mass, charge, speed, Bz, dt=0.0001)

    assert result["is_radius_verified"], f"Fallo en radio de giro: {result['radius_error_percent']}%"
    assert result["is_zero_work_verified"], f"Fallo en trabajo nulo: {result['max_ke_drift_percent']}%"
    assert result["radius_error_percent"] < 0.5


def test_wien_filter_condition():
    # Ey = 500 V/m, Bz = 0.05 T => v_wien = 10,000 m/s
    res_matched = verify_wien_filter(
        mass=1e-6,
        charge=1e-6,
        speed=10000.0,
        Ey=500.0,
        Bz=0.05,
    )
    assert res_matched["is_speed_matched"]
    assert res_matched["is_undeflected"]
