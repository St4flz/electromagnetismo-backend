"""Pruebas unitarias para el campo eléctrico y principio de superposición."""

import math
from domain.vectors import Vector2D
from domain.particles import Particle
from domain.electric_field import electric_field_at, electric_field_superposition
from domain.units import COULOMB_CONSTANT


def test_electric_field_single_charge():
    # Carga positiva en el origen
    # Campo a r = 1m sobre +x debe apuntar hacia +x con magnitud k*q/1^2
    q = 1e-6
    source = Particle(position=Vector2D(0.0, 0.0), charge=q, label="q1")
    point = Vector2D(1.0, 0.0)

    fp = electric_field_at(point, source)
    expected_E = COULOMB_CONSTANT * q / (1.0 ** 2)

    assert math.isclose(fp.magnitude, expected_E, rel_tol=1e-5)
    assert fp.E.x > 0
    assert math.isclose(fp.E.y, 0.0, abs_tol=1e-12)


def test_dipole_field_symmetry():
    # Dipolo simétrico: +q en (0, 0.05) y -q en (0, -0.05)
    # En el origen (0, 0), el campo de ambas cargas debe sumar hacia -y
    q = 1e-6
    p_pos = Particle(position=Vector2D(0.0, 0.05), charge=q, label="+q")
    p_neg = Particle(position=Vector2D(0.0, -0.05), charge=-q, label="-q")

    center = Vector2D(0.0, 0.0)
    fp = electric_field_superposition(center, [p_pos, p_neg])

    # El campo debe apuntar hacia abajo (hacia la carga negativa)
    assert math.isclose(fp.E.x, 0.0, abs_tol=1e-9)
    assert fp.E.y < 0
