"""Pruebas unitarias para el Capítulo 7: Onda Electromagnética en el Vacío 2D."""

import math
import pytest
from domain.vectors import Vector2D
from domain.units import SPEED_OF_LIGHT
from domain.em_wave import (
    PlaneWave2D,
    VACUUM_IMPEDANCE,
    verify_vacuum_wave_speed,
    verify_orthogonality_and_equipartition,
    verify_poynting_impedance,
)


def test_vacuum_speed_of_light_relation():
    res = verify_vacuum_wave_speed(1.5e9)
    assert res["is_verified"]
    assert res["relative_error_percent"] < 1e-4


def test_orthogonality_and_energy_equipartition():
    res = verify_orthogonality_and_equipartition(
        amplitude_E0=80.0,
        frequency=2.4e9,
        angle_rad=math.pi / 4.0,
    )
    assert res["is_orthogonal"]
    assert res["is_ratio_verified"], f"Error en razón E/B: {res['eb_ratio_error_percent']}%"
    assert res["is_equipartition_verified"], f"Error en equipartición: {res['equipartition_error_percent']}%"


def test_poynting_impedance_vacuum():
    res = verify_poynting_impedance(amplitude_E0=120.0, frequency=10e9)
    assert res["is_verified"]
    assert math.isclose(res["derived_vacuum_impedance_ohms"], 376.73, rel_tol=1e-3)
