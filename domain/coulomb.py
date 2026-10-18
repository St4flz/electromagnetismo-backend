"""Ley de Coulomb — la fuerza entre dos cargas eléctricas.

         k · q₁ · q₂
  F₁₂ = ――――――――――――― · r̂₁₂
              r²

Donde:
    k   = 8.9875 × 10⁹ N·m²/C²   (constante de Coulomb)
    q₁  = carga de la partícula 1  [C]
    q₂  = carga de la partícula 2  [C]
    r   = distancia entre cargas    [m]
    r̂₁₂ = vector unitario de q₁ hacia q₂

Interpretación del signo:
    q₁·q₂ > 0  →  cargas del mismo signo  →  F apunta hacia afuera  →  REPULSIÓN
    q₁·q₂ < 0  →  cargas de signo opuesto →  F apunta hacia adentro →  ATRACCIÓN

La fuerza sigue la tercera ley de Newton:
    F₂₁ = -F₁₂  (igual magnitud, dirección opuesta)

Historia:
    Charles-Augustin de Coulomb la demostró experimentalmente en 1785
    usando una balanza de torsión.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .vectors import Vector2D, Vector3D
from .particles import Particle
from .units import COULOMB_CONSTANT


# Distancia mínima para evitar la singularidad en r → 0.
# Físicamente, las cargas puntuales no pueden superponerse.
MIN_DISTANCE: float = 1e-10  # [m]


@dataclass(frozen=True)
class CoulombResult:
    """Resultado completo de un cálculo de fuerza de Coulomb.

    Incluye no solo la fuerza, sino todos los valores intermedios
    para que el frontend pueda mostrar cada parte de la ecuación:

        F₁₂ = k · q₁ · q₂ / r² · r̂₁₂
              ↑   ↑    ↑    ↑     ↑
              k   q₁   q₂   r²   r̂₁₂

    Cada uno de estos valores está disponible para la UI.
    """

    force: Union[Vector2D, Vector3D]          # F₁₂ — vector de fuerza [N]
    force_magnitude: float                     # |F₁₂| [N]
    distance: float                            # r — distancia entre cargas [m]
    unit_vector: Union[Vector2D, Vector3D]     # r̂₁₂ — dirección de la fuerza
    is_attractive: bool                        # ¿La fuerza es atractiva?
    particle_from_id: str                      # ID de la partícula que EJERCE la fuerza
    particle_to_id: str                        # ID de la partícula que RECIBE la fuerza

    def to_dict(self) -> dict:
        return {
            "force": self.force.to_dict(),
            "forceMagnitude": self.force_magnitude,
            "distance": self.distance,
            "unitVector": self.unit_vector.to_dict(),
            "isAttractive": self.is_attractive,
            "particleFromId": self.particle_from_id,
            "particleToId": self.particle_to_id,
        }


def coulomb_force(p1: Particle, p2: Particle) -> CoulombResult:
    """Calcula la fuerza que p2 ejerce sobre p1.

    Implementa la Ley de Coulomb:

        F₁₂ = k · q₁ · q₂ / |r₁₂|² · r̂₁₂

    donde r₁₂ = r₂ - r₁ (vector que va DE p1 HACIA p2)

    Args:
        p1: La partícula que RECIBE la fuerza.
        p2: La partícula que EJERCE la fuerza.

    Returns:
        CoulombResult con la fuerza y todos los valores intermedios.

    Raises:
        ValueError: Si las cargas están demasiado cerca (r < MIN_DISTANCE).
    """
    # ┌─────────────────────────────────────────────────────────┐
    # │ Paso 1: r₁₂ = r₂ - r₁                                 │
    # │ Vector de separación: apunta de p1 hacia p2             │
    # └─────────────────────────────────────────────────────────┘
    r12 = p2.position - p1.position

    # ┌─────────────────────────────────────────────────────────┐
    # │ Paso 2: r = |r₁₂| — distancia entre las cargas         │
    # └─────────────────────────────────────────────────────────┘
    distance = r12.magnitude

    if distance < MIN_DISTANCE:
        raise ValueError(
            f"Las cargas {p1.label} y {p2.label} están demasiado cerca "
            f"(r = {distance:.2e} m). La ley de Coulomb diverge cuando r → 0."
        )

    # ┌─────────────────────────────────────────────────────────┐
    # │ Paso 3: r̂₁₂ = r₁₂ / |r₁₂| — dirección de la fuerza   │
    # └─────────────────────────────────────────────────────────┘
    unit = r12.unit

    # ┌─────────────────────────────────────────────────────────┐
    # │ Paso 4: |F| = k · q₁ · q₂ / r²                        │
    # │                                                         │
    # │ El signo de q₁·q₂ determina atracción vs repulsión:    │
    # │   (+)(+) = + → F en dirección r̂ → repulsión             │
    # │   (+)(-) = - → F en dirección -r̂ → atracción            │
    # │   (-)(+) = - → F en dirección -r̂ → atracción            │
    # │   (-)(-) = + → F en dirección r̂ → repulsión             │
    # └─────────────────────────────────────────────────────────┘
    force_magnitude_signed = COULOMB_CONSTANT * p1.charge * p2.charge / (distance ** 2)

    # ┌─────────────────────────────────────────────────────────┐
    # │ Paso 5: F₁₂ = |F| · r̂₁₂                               │
    # │                                                         │
    # │ El signo ya está en force_magnitude_signed:             │
    # │   + → fuerza en dirección r̂ (hacia p2) → repulsión     │
    # │   - → fuerza en dirección -r̂ (lejos de p2) → atracción │
    # │                                                         │
    # │ ESPERA — eso parece invertido. Veamos:                  │
    # │   Si las cargas se repelen, p1 se aleja de p2,          │
    # │   o sea F apunta en dirección OPUESTA a r̂₁₂.            │
    # │   Pero r̂₁₂ apunta DE p1 HACIA p2.                      │
    # │   Así que repulsión = F en dirección -r̂₁₂.              │
    # │                                                         │
    # │   q₁·q₂ > 0 → force_magnitude_signed > 0               │
    # │   F = (+) · r̂₁₂ apunta hacia p2 → ¡MAL!               │
    # │                                                         │
    # │   Solución: F₁₂ = -|F_signed| · r̂₁₂                    │
    # │   O equivalentemente, usar la convención de que         │
    # │   el vector va DE p2 HACIA p1 para la fuerza ON p1.     │
    # │                                                         │
    # │   La convención correcta es:                            │
    # │   F sobre p1 = k·q₁·q₂/r² · r̂₂₁ donde r̂₂₁ = -r̂₁₂     │
    # │                                                         │
    # │   Entonces: F = -(k·q₁·q₂/r²) · r̂₁₂                   │
    # │   O sea:    F = (k·q₁·q₂/r²) · r̂₂₁                    │
    # └─────────────────────────────────────────────────────────┘
    # Usando la convención r̂₂₁ = -r̂₁₂ (del que ejerce hacia el que recibe):
    # F_on_p1 = k·q₁·q₂/r² · (-r̂₁₂)
    force_vector = unit * (-force_magnitude_signed)

    is_attractive = (p1.charge * p2.charge) < 0

    return CoulombResult(
        force=force_vector,
        force_magnitude=abs(force_magnitude_signed),
        distance=distance,
        unit_vector=unit,
        is_attractive=is_attractive,
        particle_from_id=p2.id,
        particle_to_id=p1.id,
    )


def net_force_on(target: Particle, others: list[Particle]) -> Union[Vector2D, Vector3D]:
    """Calcula la fuerza neta sobre una partícula debida a todas las demás.

    Principio de superposición:
        F_total = Σ F_i  (la fuerza total es la suma vectorial de cada fuerza individual)

    Este principio es fundamental: las fuerzas eléctricas se suman linealmente.
    No importa cuántas cargas haya, cada par interactúa independientemente.

    Args:
        target: La partícula sobre la que calculamos la fuerza neta.
        others: Todas las demás partículas.

    Returns:
        Vector de fuerza neta [N].
    """
    if isinstance(target.position, Vector3D):
        total = Vector3D.zero()
    else:
        total = Vector2D.zero()

    for other in others:
        if other.id == target.id:
            continue
        result = coulomb_force(target, other)
        total = total + result.force

    return total


def all_forces(particles: list[Particle]) -> list[CoulombResult]:
    """Calcula TODAS las fuerzas entre todos los pares de partículas.

    Para N partículas hay N·(N-1) fuerzas dirigidas
    (cada par genera dos fuerzas: F₁₂ y F₂₁ = -F₁₂).

    Retorna la lista completa para que el frontend pueda dibujar
    la flecha de fuerza sobre cada partícula.
    """
    results = []
    for i, p1 in enumerate(particles):
        for j, p2 in enumerate(particles):
            if i == j:
                continue
            results.append(coulomb_force(p1, p2))
    return results
