"""Pruebas unitarias para la Ley de Coulomb: conservación, 3ra ley de Newton y valores canónicos."""

import math
import pytest
from domain.vectors import Vector2D
from domain.particles import Particle
from domain.coulomb import coulomb_force, all_forces
from domain.units import COULOMB_CONSTANT


def test_coulomb_two_positive_charges_repel():
    # Dos cargas de +1 uC separadas por 1 metro a lo largo del eje x
    # q1 en (0, 0), q2 en (1, 0)
    # F esperada sobre q1 debida a q2: hacia -x (repulsión)
    # Magnitud: k * 1e-6 * 1e-6 / 1^2 = k * 1e-12 N ~ 0.0089875 N
    p1 = Particle(position=Vector2D(0.0, 0.0), charge=1e-6, label="q1", id="p1")
    p2 = Particle(position=Vector2D(1.0, 0.0), charge=1e-6, label="q2", id="p2")

    res = coulomb_force(p1, p2)
    expected_magnitude = COULOMB_CONSTANT * 1e-6 * 1e-6 / (1.0 ** 2)

    assert math.isclose(res.force_magnitude, expected_magnitude, rel_tol=1e-5)
    assert not res.is_attractive
    # La fuerza sobre p1 apunta en -x (alejándose de p2)
    assert res.force.x < 0
    assert math.isclose(res.force.y, 0.0, abs_tol=1e-12)


def test_coulomb_opposite_charges_attract():
    # q1 = +1 uC en (0,0), q2 = -1 uC en (1, 0)
    # Fuerza sobre q1 debe apuntar hacia +x (hacia q2, atracción)
    p1 = Particle(position=Vector2D(0.0, 0.0), charge=1e-6, label="q1", id="p1")
    p2 = Particle(position=Vector2D(1.0, 0.0), charge=-1e-6, label="q2", id="p2")

    res = coulomb_force(p1, p2)
    assert res.is_attractive
    assert res.force.x > 0
    assert math.isclose(res.force.y, 0.0, abs_tol=1e-12)


def test_third_law_of_newton():
    # F12 = -F21
    p1 = Particle(position=Vector2D(0.02, 0.03), charge=2.5e-6, label="q1", id="p1")
    p2 = Particle(position=Vector2D(0.08, -0.01), charge=-1.8e-6, label="q2", id="p2")

    f_on_p1 = coulomb_force(p1, p2).force
    f_on_p2 = coulomb_force(p2, p1).force

    # Suma de fuerzas internas = 0 (3ra Ley de Newton)
    f_net = f_on_p1 + f_on_p2
    assert math.isclose(f_net.magnitude, 0.0, abs_tol=1e-9)
