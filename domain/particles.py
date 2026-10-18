"""Una partícula con carga eléctrica.

En electrostática, todo empieza con cargas puntuales:
objetos con carga q ubicados en una posición r.

La carga q se mide en Coulombs [C]:
    Protón:   q = +1.6 × 10⁻¹⁹ C
    Electrón: q = -1.6 × 10⁻¹⁹ C

Para la simulación usamos microcoulombs (µC = 10⁻⁶ C)
para que las fuerzas sean visibles a escala de centímetros.

Ejemplo:
    q₁ = +3 µC en (2 cm, 5 cm) → Particle(Vector2D(0.02, 0.05), 3e-6, "q₁")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union
import uuid

from .vectors import Vector2D, Vector3D


@dataclass
class Particle:
    """Carga puntual en el espacio.

    Attributes:
        position: Dónde está la carga en el espacio [m].
                  Puede ser Vector2D (plano) o Vector3D (espacio).
        charge:   Cuánta carga tiene [C] (Coulombs).
                  Positivo (+) para protones/cargas positivas.
                  Negativo (-) para electrones/cargas negativas.
        label:    Nombre para mostrar en la UI ("q₁", "q₂", etc.).
        id:       Identificador único para tracking en frontend.
    """

    position: Union[Vector2D, Vector3D]
    charge: float  # [C] — positivo o negativo
    label: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def is_positive(self) -> bool:
        """¿La carga es positiva? (como un protón)"""
        return self.charge > 0

    @property
    def is_negative(self) -> bool:
        """¿La carga es negativa? (como un electrón)"""
        return self.charge < 0

    @property
    def charge_microcoulombs(self) -> float:
        """Carga en microcoulombs (µC) — unidad más cómoda para la UI.

        1 µC = 10⁻⁶ C
        """
        return self.charge * 1e6

    def to_dict(self) -> dict:
        """Serializa para enviar por la API."""
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "charge": self.charge,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Particle:
        """Crea una partícula desde un diccionario de la API."""
        pos_data = data["position"]
        if "z" in pos_data:
            position = Vector3D.from_dict(pos_data)
        else:
            position = Vector2D.from_dict(pos_data)

        return cls(
            position=position,
            charge=data["charge"],
            label=data.get("label", ""),
            id=data.get("id", str(uuid.uuid4())[:8]),
        )

    def __repr__(self) -> str:
        sign = "+" if self.charge >= 0 else ""
        return (
            f"Particle({self.label or self.id}: "
            f"q={sign}{self.charge_microcoulombs:.2f} µC, "
            f"pos={self.position})"
        )
