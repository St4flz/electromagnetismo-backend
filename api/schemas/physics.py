"""Schemas Pydantic — contratos entre frontend y backend.

Estos modelos definen EXACTAMENTE qué datos viajan por la API.
Son el "idioma" compartido entre React y FastAPI.

Cada schema tiene un nombre que refleja su significado físico,
no su rol técnico ("ParticleData" en vez de "RequestBody").
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# VECTORES
# ═══════════════════════════════════════════════════════════════════

class Vector2DSchema(BaseModel):
    """Vector en 2D. Unidades dependen del contexto: [m], [N], [V/m]."""
    x: float
    y: float


class Vector3DSchema(BaseModel):
    """Vector en 3D. Unidades dependen del contexto."""
    x: float
    y: float
    z: float


# ═══════════════════════════════════════════════════════════════════
# PARTÍCULAS
# ═══════════════════════════════════════════════════════════════════

class ParticleSchema(BaseModel):
    """Carga puntual que el frontend envía/recibe.

    position: en metros [m]
    charge: en Coulombs [C] (positivo o negativo)
    """
    id: str
    position: Vector2DSchema | Vector3DSchema
    charge: float = Field(..., description="Carga en Coulombs [C]")
    label: str = ""


# ═══════════════════════════════════════════════════════════════════
# RESULTADOS DE COULOMB
# ═══════════════════════════════════════════════════════════════════

class CoulombForceSchema(BaseModel):
    """Resultado de calcular la fuerza de Coulomb entre dos cargas.

    Incluye todos los valores intermedios para que la UI pueda
    mostrar la ecuación paso a paso:

        F = k · q₁ · q₂ / r² · r̂
    """
    force: Vector2DSchema | Vector3DSchema          # F [N]
    forceMagnitude: float                            # |F| [N]
    distance: float                                  # r [m]
    unitVector: Vector2DSchema | Vector3DSchema      # r̂
    isAttractive: bool
    particleFromId: str   # Quién EJERCE la fuerza
    particleToId: str     # Quién la RECIBE


# ═══════════════════════════════════════════════════════════════════
# CAMPO ELÉCTRICO
# ═══════════════════════════════════════════════════════════════════

class FieldPointSchema(BaseModel):
    """Campo E evaluado en un punto del espacio."""
    point: Vector2DSchema | Vector3DSchema
    E: Vector2DSchema | Vector3DSchema      # [V/m]
    magnitude: float                         # |E| [V/m]


# ═══════════════════════════════════════════════════════════════════
# REQUESTS del WebSocket
# ═══════════════════════════════════════════════════════════════════

class FieldGridQuery(BaseModel):
    """Solicitud de campo eléctrico en una grilla de puntos."""
    xRange: tuple[float, float]
    yRange: tuple[float, float]
    resolution: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Puntos por eje (total = resolution²)"
    )


class PhysicsRequest(BaseModel):
    """Mensaje que el frontend envía por WebSocket.

    type determina qué cálculo hacer:
        "coulomb"        → calcula fuerzas entre todos los pares
        "electric_field" → calcula E en una grilla de puntos
    """
    type: str = Field(..., pattern="^(coulomb|electric_field)$")
    particles: list[ParticleSchema]
    dimensions: int = Field(default=2, ge=2, le=3)
    fieldQuery: FieldGridQuery | None = None


class PhysicsResponse(BaseModel):
    """Mensaje que el backend envía de vuelta por WebSocket."""
    type: str
    forces: list[CoulombForceSchema] | None = None
    field: list[FieldPointSchema] | None = None
    computeTimeMs: float  # Transparencia: cuánto tardó el cálculo
