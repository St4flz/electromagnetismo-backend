"""Schemas Pydantic — contratos entre frontend y backend.

Estos modelos definen EXACTAMENTE qué datos viajan por la API.
Son el "idioma" compartido entre React y FastAPI.

Cada schema tiene un nombre que refleja su significado físico,
no su rol técnico ("ParticleData" en vez de "RequestBody").
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ═══════════════════════════════════════════════════════════════════
# VECTORES
# ═══════════════════════════════════════════════════════════════════

class Vector2DSchema(BaseModel):
    """Vector en 2D. Unidades dependen del contexto: [m], [N], [V/m]."""
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float


class Vector3DSchema(BaseModel):
    """Vector en 3D. Unidades dependen del contexto."""
    model_config = ConfigDict(extra="forbid")
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
    position: Vector3DSchema | Vector2DSchema
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
    force: Vector3DSchema | Vector2DSchema          # F [N]
    forceMagnitude: float                            # |F| [N]
    distance: float                                  # r [m]
    unitVector: Vector3DSchema | Vector2DSchema      # r̂
    isAttractive: bool
    particleFromId: str   # Quién EJERCE la fuerza
    particleToId: str     # Quién la RECIBE


# ═══════════════════════════════════════════════════════════════════
# CAMPO ELÉCTRICO
# ═══════════════════════════════════════════════════════════════════

class FieldPointSchema(BaseModel):
    """Campo E evaluado en un punto del espacio."""
    point: Vector3DSchema | Vector2DSchema
    E: Vector3DSchema | Vector2DSchema      # [V/m]
    magnitude: float                         # |E| [V/m]


# ═══════════════════════════════════════════════════════════════════
# DINÁMICA Y TRAYECTORIAS (Capítulo 5)
# ═══════════════════════════════════════════════════════════════════

class DynamicParticleSchema(BaseModel):
    """Partícula con propiedades inerciales y cinemáticas."""
    particle: ParticleSchema
    mass: float = Field(default=1e-6, gt=0, description="Masa en kilogramos [kg]")
    velocity: Vector3DSchema | Vector2DSchema = Field(
        default_factory=lambda: Vector2DSchema(x=0.0, y=0.0),
        description="Velocidad inicial [m/s]"
    )


class UniformFieldSchema(BaseModel):
    """Campo eléctrico uniforme espacialmente delimitado (placas deflectoras)."""
    fieldVector: Vector3DSchema | Vector2DSchema
    xBounds: tuple[float, float] | None = None
    yBounds: tuple[float, float] | None = None
    zBounds: tuple[float, float] | None = None


class TrajectorySnapshotSchema(BaseModel):
    """Instantánea de la partícula a lo largo de su vuelo."""
    time: float
    position: Vector3DSchema | Vector2DSchema
    velocity: Vector3DSchema | Vector2DSchema
    acceleration: Vector3DSchema | Vector2DSchema
    speed: float
    kineticEnergy: float
    potentialEnergy: float
    totalEnergy: float


class TrajectoryQuery(BaseModel):
    """Parámetros para calcular la trayectoria dinámica."""
    dynamicParticle: DynamicParticleSchema
    uniformFields: list[UniformFieldSchema] | None = None
    totalTime: float = Field(default=0.05, gt=0, le=2.0, description="Tiempo de simulación [s]")
    dt: float = Field(default=0.0005, gt=0, le=0.01, description="Paso temporal de integración [s]")


# ═══════════════════════════════════════════════════════════════════
# REQUESTS del WebSocket y REST
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
    """Mensaje que el frontend envía por WebSocket o REST.

    type determina qué cálculo hacer:
        "coulomb"        → calcula fuerzas entre todos los pares
        "electric_field" → calcula E en una grilla de puntos
        "trajectory_sim" → simula el movimiento continuo de una carga en el tiempo
    """
    type: str = Field(..., pattern="^(coulomb|electric_field|trajectory_sim)$")
    particles: list[ParticleSchema] = Field(default_factory=list)
    dimensions: int = Field(default=2, ge=2, le=3)
    fieldQuery: FieldGridQuery | None = None
    trajectoryQuery: TrajectoryQuery | None = None


class PhysicsResponse(BaseModel):
    """Mensaje que el backend envía de vuelta por WebSocket o REST."""
    type: str
    forces: list[CoulombForceSchema] | None = None
    field: list[FieldPointSchema] | None = None
    trajectory: list[TrajectorySnapshotSchema] | None = None
    computeTimeMs: float  # Transparencia: cuánto tardó el cálculo
