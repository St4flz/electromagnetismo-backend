"""Campo eléctrico — la fuerza por unidad de carga.

El campo eléctrico E en un punto del espacio responde la pregunta:

    "¿Qué fuerza sentiría una carga de prueba q₀ si la pongo aquí?"

        E = F / q₀

Para una carga puntual q, el campo que crea en un punto P es:

             k · q
    E(P) = ―――――― · r̂
              r²

Donde:
    r  = distancia desde la carga q hasta el punto P  [m]
    r̂  = vector unitario que apunta DE la carga HACIA el punto P

Principio de superposición:
    E_total(P) = Σ E_i(P)

    El campo total es la suma vectorial de los campos individuales.
    Cada carga contribuye independientemente al campo en cada punto.

Unidades: [V/m] (Voltios por metro) = [N/C] (Newtons por Coulomb)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .vectors import Vector2D, Vector3D
from .particles import Particle
from .units import COULOMB_CONSTANT


# Distancia mínima para evitar la singularidad
MIN_DISTANCE: float = 1e-10  # [m]


@dataclass(frozen=True)
class FieldPoint:
    """El campo eléctrico evaluado en un punto del espacio.

    Contiene el punto donde se evaluó, el vector E, y su magnitud.
    """

    point: Union[Vector2D, Vector3D]        # Dónde se evaluó [m]
    E: Union[Vector2D, Vector3D]            # Vector campo eléctrico [V/m]
    magnitude: float                         # |E| [V/m]

    def to_dict(self) -> dict:
        return {
            "point": self.point.to_dict(),
            "E": self.E.to_dict(),
            "magnitude": self.magnitude,
        }


def electric_field_at(
    point: Union[Vector2D, Vector3D],
    source: Particle,
) -> FieldPoint:
    """Campo eléctrico creado por una carga puntual en un punto dado.

    Implementa:
                 k · q
        E(P) = ―――――― · r̂
                  r²

    donde r̂ apunta DESDE la carga fuente HACIA el punto P.

    Args:
        point: El punto P donde queremos conocer el campo [m].
        source: La carga que crea el campo.

    Returns:
        FieldPoint con E(P) y su magnitud.
    """
    # r = P - r_source — vector de la carga al punto de evaluación
    r = point - source.position
    distance = r.magnitude

    if distance < MIN_DISTANCE:
        # En el punto exacto de la carga, E → ∞ (singularidad).
        # Retornamos un campo "máximo" en la dirección estimada.
        if isinstance(point, Vector3D):
            return FieldPoint(point, Vector3D.zero(), 0.0)
        return FieldPoint(point, Vector2D.zero(), 0.0)

    # E = k · q / r² · r̂
    r_hat = r.unit
    E_magnitude = COULOMB_CONSTANT * source.charge / (distance ** 2)
    E_vector = r_hat * E_magnitude

    return FieldPoint(
        point=point,
        E=E_vector,
        magnitude=abs(E_magnitude),
    )


def electric_field_superposition(
    point: Union[Vector2D, Vector3D],
    sources: list[Particle],
) -> FieldPoint:
    """Campo eléctrico total en un punto, debido a múltiples cargas.

    Principio de superposición:
        E_total(P) = Σ E_i(P) = Σ (k · qᵢ / rᵢ²) · r̂ᵢ

    Cada carga contribuye independientemente.
    El campo total es la suma vectorial de todas las contribuciones.
    """
    if isinstance(point, Vector3D):
        total_E = Vector3D.zero()
    else:
        total_E = Vector2D.zero()

    for source in sources:
        fp = electric_field_at(point, source)
        total_E = total_E + fp.E

    return FieldPoint(
        point=point,
        E=total_E,
        magnitude=total_E.magnitude,
    )


def compute_field_grid_2d(
    sources: list[Particle],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: int,
) -> list[FieldPoint]:
    """Calcula el campo eléctrico en una grilla 2D de puntos.

    Crea una grilla uniforme de resolution × resolution puntos
    y evalúa E_total en cada uno. Esto es lo que el frontend usa
    para dibujar el campo vectorial o el mapa de calor.

    Args:
        sources: Lista de cargas que crean el campo.
        x_range: (x_min, x_max) en metros.
        y_range: (y_min, y_max) en metros.
        resolution: Número de puntos por eje (total = resolution²).

    Returns:
        Lista de FieldPoint, uno por cada punto de la grilla.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    dx = (x_max - x_min) / max(resolution - 1, 1)
    dy = (y_max - y_min) / max(resolution - 1, 1)

    field_points: list[FieldPoint] = []

    for i in range(resolution):
        for j in range(resolution):
            x = x_min + i * dx
            y = y_min + j * dy
            point = Vector2D(x, y)
            fp = electric_field_superposition(point, sources)
            field_points.append(fp)

    return field_points
