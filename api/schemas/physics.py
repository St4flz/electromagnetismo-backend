"""Schemas Pydantic — contratos entre frontend y backend.

Estos modelos definen EXACTAMENTE qué datos viajan por la API (REST y WebSocket).
Son el contrato tipado y validado compartido entre React y FastAPI.

Cubre todos los capítulos:
    Capítulo 1 a 4: Cargas, Coulomb 2D/3D y Campo Eléctrico
    Capítulo 5: Cinemática, Velocity-Verlet y Deflexión de Thomson
    Capítulo 6: Ondas Magnéticas, Biot-Savart, Lorentz 2D y Ciclotrón
    Capítulo 7: Onda Electromagnética en Vacío 2D, Poynting y Equipartición
    Capítulo 8: Introducción a Metamateriales, Celda (a,d,w), Veselago DNG y TMM
    Capítulo 9: FDTD 2D, Rejilla Yee TM, Condición CFL y Espectro FFT
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# ═══════════════════════════════════════════════════════════════════
# VECTORES
# ═══════════════════════════════════════════════════════════════════

class Vector2DSchema(BaseModel):
    """Vector en 2D. Unidades dependen del contexto: [m], [N], [V/m], [T]."""
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float


class Vector3DSchema(BaseModel):
    """Vector en 3D."""
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    z: float


# ═══════════════════════════════════════════════════════════════════
# PARTÍCULAS
# ═══════════════════════════════════════════════════════════════════

class ParticleSchema(BaseModel):
    """Carga puntual que el frontend envía/recibe [m], [C]."""
    id: str
    position: Vector3DSchema | Vector2DSchema
    charge: float = Field(..., description="Carga en Coulombs [C]")
    label: str = ""


# ═══════════════════════════════════════════════════════════════════
# RESULTADOS DE COULOMB (Capítulos 2 y 3)
# ═══════════════════════════════════════════════════════════════════

class CoulombForceSchema(BaseModel):
    """Fuerza electrostática F₁₂ con datos paso a paso."""
    force: Vector3DSchema | Vector2DSchema
    forceMagnitude: float
    distance: float
    unitVector: Vector3DSchema | Vector2DSchema
    isAttractive: bool
    particleFromId: str
    particleToId: str


# ═══════════════════════════════════════════════════════════════════
# CAMPO ELÉCTRICO (Capítulo 4)
# ═══════════════════════════════════════════════════════════════════

class FieldPointSchema(BaseModel):
    """Campo E evaluado en un punto del espacio [V/m]."""
    point: Vector3DSchema | Vector2DSchema
    E: Vector3DSchema | Vector2DSchema
    magnitude: float


class FieldGridQuery(BaseModel):
    """Solicitud de campo eléctrico en una grilla de puntos."""
    xRange: tuple[float, float] = (-0.2, 0.2)
    yRange: tuple[float, float] = (-0.2, 0.2)
    resolution: int = Field(default=20, ge=5, le=60)


# ═══════════════════════════════════════════════════════════════════
# DINÁMICA Y DEFLEXIÓN (Capítulo 5)
# ═══════════════════════════════════════════════════════════════════

class DynamicParticleSchema(BaseModel):
    particle: ParticleSchema
    mass: float = Field(default=1e-6, gt=0)
    velocity: Vector3DSchema | Vector2DSchema = Field(
        default_factory=lambda: Vector2DSchema(x=0.0, y=0.0)
    )


class UniformFieldSchema(BaseModel):
    fieldVector: Vector3DSchema | Vector2DSchema
    xBounds: tuple[float, float] | None = None
    yBounds: tuple[float, float] | None = None
    zBounds: tuple[float, float] | None = None


class TrajectorySnapshotSchema(BaseModel):
    time: float
    position: Vector3DSchema | Vector2DSchema
    velocity: Vector3DSchema | Vector2DSchema
    acceleration: Vector3DSchema | Vector2DSchema
    speed: float
    kineticEnergy: float
    potentialEnergy: float
    totalEnergy: float


class TrajectoryQuery(BaseModel):
    dynamicParticle: DynamicParticleSchema
    uniformFields: list[UniformFieldSchema] | None = None
    totalTime: float = Field(default=0.05, gt=0, le=2.0)
    dt: float = Field(default=0.0005, gt=0, le=0.01)


class ThomsonVerificationQuery(BaseModel):
    charge: float = -1.602e-19
    mass: float = 9.109e-31
    v0x: float = 1e7
    Ey: float = 2000.0
    length: float = 0.05
    dt: float = 1e-11


# ═══════════════════════════════════════════════════════════════════
# MAGNETISMO Y LORENTZ 2D (Capítulo 6)
# ═══════════════════════════════════════════════════════════════════

class MagneticPointSchema(BaseModel):
    point: Vector2DSchema
    B: Vector2DSchema
    Bz: float
    magnitude: float


class LorentzSnapshotSchema(BaseModel):
    time: float
    position: Vector2DSchema
    velocity: Vector2DSchema
    acceleration: Vector2DSchema
    speed: float
    kineticEnergy: float
    workDone: float


class LorentzQuery(BaseModel):
    charge: float = 1e-6
    mass: float = 1e-6
    initialPos: Vector2DSchema = Field(default_factory=lambda: Vector2DSchema(x=0.0, y=0.0))
    initialVel: Vector2DSchema = Field(default_factory=lambda: Vector2DSchema(x=50.0, y=0.0))
    E: Vector2DSchema = Field(default_factory=lambda: Vector2DSchema(x=0.0, y=0.0))
    Bz: float = 0.5
    totalTime: float = Field(default=0.005, gt=0, le=1.0)
    dt: float = Field(default=0.00005, gt=0, le=0.001)


class CyclotronVerificationQuery(BaseModel):
    mass: float = 1e-6
    charge: float = 1e-6
    speed: float = 50.0
    Bz: float = 0.5
    dt: float = 0.00005


class WienFilterQuery(BaseModel):
    mass: float = 1e-6
    charge: float = 1e-6
    speed: float = 1000.0
    Ey: float = 100.0
    Bz: float = 0.1
    length: float = 0.2
    dt: float = 0.00005


# ═══════════════════════════════════════════════════════════════════
# ONDA ELECTROMAGNÉTICA EN EL VACÍO 2D (Capítulo 7)
# ═══════════════════════════════════════════════════════════════════

class EMWaveSnapshotSchema(BaseModel):
    point: Vector2DSchema
    Ez: float
    B: Vector2DSchema
    BMagnitude: float
    poynting: Vector2DSchema
    poyntingMagnitude: float
    energyDensityE: float
    energyDensityB: float
    totalEnergyDensity: float


class EMWaveGridQuery(BaseModel):
    amplitudeE0: float = 100.0
    frequency: float = 1.0e9
    propagationAngleRad: float = 0.0
    time: float = 0.0
    resolution: int = Field(default=16, ge=4, le=40)
    xRange: tuple[float, float] = (-0.15, 0.15)
    yRange: tuple[float, float] = (-0.15, 0.15)


class EMWaveVerificationQuery(BaseModel):
    amplitudeE0: float = 100.0
    frequency: float = 1.0e9
    angleRad: float = 0.5235987755982988


# ═══════════════════════════════════════════════════════════════════
# METAMATERIALES (Capítulo 8)
# ═══════════════════════════════════════════════════════════════════

class MetamaterialCellSchema(BaseModel):
    a: float = Field(default=10.0, description="Tamaño de celda en mm")
    d: float = Field(default=2.0, description="Separación en mm")
    w: float = Field(default=1.0, description="Ancho de elemento en mm")
    periods: int = Field(default=5, ge=1, le=20)
    lossFactor: float = Field(default=0.002, ge=0.0)


class SpectralPointSchema(BaseModel):
    frequencyGhz: float
    transmission: float
    reflection: float
    absorption: float
    transmissionDb: float
    epsReal: float
    epsImag: float
    muReal: float
    muImag: float
    nReal: float
    nImag: float
    isLeftHanded: bool


class MetamaterialsQuery(BaseModel):
    cell: MetamaterialCellSchema
    freqMinGhz: float = 5.0
    freqMaxGhz: float = 15.0
    points: int = Field(default=80, ge=10, le=200)


# ═══════════════════════════════════════════════════════════════════
# FDTD 2D (Capítulo 9)
# ═══════════════════════════════════════════════════════════════════

class FDTDSimQuery(BaseModel):
    Nx: int = Field(default=80, ge=30, le=120)
    Ny: int = Field(default=50, ge=20, le=80)
    dx: float = 0.002
    courantFactor: float = 0.5
    sourceType: Literal["gaussian", "sine"] = "gaussian"
    sourceFreqGhz: float = 10.0
    steps: int = Field(default=5, ge=1, le=50)
    addSlab: bool = False
    slabStart: int = 30
    slabEnd: int = 42
    slabEps: float = 4.0
    addMetamaterial: bool = False
    reset: bool = False


# ═══════════════════════════════════════════════════════════════════
# MENSAJES UNIFICADOS WEBSOCKET / REST
# ═══════════════════════════════════════════════════════════════════

class PhysicsRequest(BaseModel):
    """Petición unificada del cliente (React) al backend."""
    type: str = Field(
        ...,
        pattern="^(coulomb|electric_field|trajectory_sim|ch5_thomson|ch6_magnetic_sim|ch6_cyclotron_verify|ch6_wien_verify|ch7_em_wave_grid|ch7_em_verify|ch8_metamaterials_spectrum|ch8_metamaterials_verify|ch9_fdtd_step|ch9_fdtd_verify)$"
    )
    particles: list[ParticleSchema] = Field(default_factory=list)
    dimensions: int = Field(default=2, ge=2, le=3)
    fieldQuery: FieldGridQuery | None = None
    trajectoryQuery: TrajectoryQuery | None = None
    thomsonQuery: ThomsonVerificationQuery | None = None
    lorentzQuery: LorentzQuery | None = None
    cyclotronQuery: CyclotronVerificationQuery | None = None
    wienQuery: WienFilterQuery | None = None
    emWaveQuery: EMWaveGridQuery | None = None
    emWaveVerifyQuery: EMWaveVerificationQuery | None = None
    metaQuery: MetamaterialsQuery | None = None
    fdtdQuery: FDTDSimQuery | None = None


class PhysicsResponse(BaseModel):
    """Respuesta determinista del backend hacia el cliente con mediciones exactas."""
    type: str
    forces: list[CoulombForceSchema] | None = None
    field: list[FieldPointSchema] | None = None
    trajectory: list[TrajectorySnapshotSchema] | None = None
    thomsonResult: dict[str, Any] | None = None
    lorentzTrajectory: list[LorentzSnapshotSchema] | None = None
    cyclotronResult: dict[str, Any] | None = None
    wienResult: dict[str, Any] | None = None
    emWaveGrid: list[EMWaveSnapshotSchema] | None = None
    emWaveVerifyResult: dict[str, Any] | None = None
    spectrum: list[SpectralPointSchema] | None = None
    metamaterialsVerifyResult: dict[str, Any] | None = None
    fdtdGridEz: list[list[float]] | None = None
    fdtdTimeStep: int | None = None
    fdtdCurrentTime: float | None = None
    fdtdDt: float | None = None
    fdtdSpectrum: list[dict[str, Any]] | None = None
    fdtdVerifyResult: dict[str, Any] | None = None
    computeTimeMs: float
    error: str | None = None
