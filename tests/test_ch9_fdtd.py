"""Pruebas unitarias para el Capítulo 9: Introducción a FDTD."""

import pytest
from domain.fdtd2d import (
    FDTD2DSimulator,
    verify_cfl_stability_condition,
    verify_fdtd_pulse_velocity,
)


def test_cfl_stability_theorem():
    # Estable: Courant = 0.5 <= 0.7071
    res_stable = verify_cfl_stability_condition(0.5)
    assert res_stable["theoretically_stable"]
    assert res_stable["numerically_stable"]
    assert res_stable["theorem_verified"]

    # Inestable: Courant = 1.05 > 0.7071 (explosión numérica)
    res_unstable = verify_cfl_stability_condition(1.05)
    assert not res_unstable["theoretically_stable"]
    assert not res_unstable["numerically_stable"]
    assert res_unstable["theorem_verified"]


def test_fdtd_pulse_speed_in_vacuum():
    res = verify_fdtd_pulse_velocity(Nx=60, Ny=30, dx=0.003)
    assert res["is_verified"], f"Error de dispersión: {res['dispersion_error_percent']}%"
    assert res["dispersion_error_percent"] < 5.5


def test_fdtd_dielectric_slab_insertion():
    sim = FDTD2DSimulator(Nx=50, Ny=30, dx=0.002)
    sim.add_dielectric_slab(start_x=20, end_x=28, relative_permittivity=4.0)

    # Verificar que el medio fue aplicado
    assert sim.eps_r[25, 15] == 4.0
    assert sim.eps_r[5, 15] == 1.0

    # Ejecutar 20 pasos para verificar que no hay NaN
    sim.step(20)
    assert len(sim.time_history_trans) == 20
