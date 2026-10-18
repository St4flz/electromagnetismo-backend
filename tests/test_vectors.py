"""Pruebas unitarias para las operaciones vectoriales Vector2D y Vector3D."""

import math
import pytest
from domain.vectors import Vector2D, Vector3D


def test_vector2d_algebra():
    v1 = Vector2D(3.0, 4.0)
    v2 = Vector2D(1.0, 2.0)

    # Suma y resta
    assert v1 + v2 == Vector2D(4.0, 6.0)
    assert v1 - v2 == Vector2D(2.0, 2.0)

    # Multiplicación escalar
    assert v1 * 2.0 == Vector2D(6.0, 8.0)
    assert 2.0 * v1 == Vector2D(6.0, 8.0)

    # Magnitud y vector unitario
    assert math.isclose(v1.magnitude, 5.0)
    unit = v1.unit
    assert math.isclose(unit.magnitude, 1.0)
    assert math.isclose(unit.x, 0.6)
    assert math.isclose(unit.y, 0.8)

    # Producto punto
    assert math.isclose(v1.dot(v2), 3.0 * 1.0 + 4.0 * 2.0)


def test_vector3d_algebra():
    v1 = Vector3D(1.0, 2.0, 2.0)
    assert math.isclose(v1.magnitude, 3.0)

    unit = v1.unit
    assert math.isclose(unit.magnitude, 1.0)

    # Producto cruz: i x j = k
    i = Vector3D(1.0, 0.0, 0.0)
    j = Vector3D(0.0, 1.0, 0.0)
    k = i.cross(j)
    assert math.isclose(k.x, 0.0)
    assert math.isclose(k.y, 0.0)
    assert math.isclose(k.z, 1.0)
