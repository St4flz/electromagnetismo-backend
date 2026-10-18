"""Representación de vectores en el espacio.

Un vector es la abstracción más básica de la física:
describe una magnitud con dirección y sentido.

Lo usamos para representar:
    - Posición de una carga:  r = (x, y) [metros]
    - Fuerza eléctrica:       F = (Fx, Fy) [Newtons]
    - Campo eléctrico:        E = (Ex, Ey) [V/m]

Todas las unidades están en el Sistema Internacional (SI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector2D:
    """Vector en el plano (x, y).

    Inmutable (frozen) porque los vectores representan valores,
    no estados. Para "mover" algo, se crea un nuevo vector.

    Ejemplos:
        >>> posicion = Vector2D(0.03, 0.05)   # 3 cm, 5 cm
        >>> fuerza = Vector2D(-2.1, 0.8)      # Fx = -2.1 N, Fy = 0.8 N
        >>> campo = Vector2D(1e3, -5e2)       # Ex = 1000 V/m, Ey = -500 V/m
    """

    x: float
    y: float

    # ── Operaciones vectoriales ──────────────────────────────────────

    def __add__(self, other: Vector2D) -> Vector2D:
        """Suma vectorial: a + b = (ax+bx, ay+by)"""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        """Resta vectorial: a - b = (ax-bx, ay-by)"""
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2D:
        """Producto escalar: α·v = (α·vx, α·vy)"""
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vector2D:
        """Producto escalar conmutativo: v·α = α·v"""
        return self.__mul__(scalar)

    def __neg__(self) -> Vector2D:
        """Negación: -v = (-vx, -vy)"""
        return Vector2D(-self.x, -self.y)

    def __truediv__(self, scalar: float) -> Vector2D:
        """División escalar: v/α = (vx/α, vy/α)"""
        if scalar == 0:
            raise ZeroDivisionError("No se puede dividir un vector por cero")
        return Vector2D(self.x / scalar, self.y / scalar)

    # ── Propiedades ──────────────────────────────────────────────────

    @property
    def magnitude(self) -> float:
        """Magnitud (norma euclidiana): ||v|| = √(x² + y²)

        Es la "longitud" del vector. Para una fuerza, es la intensidad.
        Para una posición, es la distancia al origen.
        """
        return math.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def magnitude_squared(self) -> float:
        """Magnitud al cuadrado: ||v||² = x² + y²

        Útil para evitar raíces cuadradas innecesarias,
        especialmente en la ley de Coulomb donde necesitamos r².
        """
        return self.x ** 2 + self.y ** 2

    @property
    def unit(self) -> Vector2D:
        """Vector unitario: v̂ = v / ||v||

        Tiene magnitud 1 y la misma dirección que v.
        Representa "la dirección" sin magnitud.

        En Coulomb: r̂₁₂ indica la dirección de la fuerza.
        """
        mag = self.magnitude
        if mag < 1e-15:
            raise ValueError(
                "El vector cero no tiene dirección definida. "
                "Físicamente, dos cargas no pueden ocupar el mismo punto."
            )
        return Vector2D(self.x / mag, self.y / mag)

    @property
    def angle(self) -> float:
        """Ángulo del vector respecto al eje x positivo, en radianes.

        θ = atan2(y, x) ∈ (-π, π]
        """
        return math.atan2(self.y, self.x)

    # ── Operaciones adicionales ──────────────────────────────────────

    def dot(self, other: Vector2D) -> float:
        """Producto punto: a·b = ax·bx + ay·by

        Mide cuánto se "alinean" dos vectores.
        Si a·b = 0, son perpendiculares.
        """
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: Vector2D) -> float:
        """Distancia euclidiana entre dos puntos: d = ||a - b||

        Usado en Coulomb para calcular r, la separación entre cargas.
        """
        return (self - other).magnitude

    # ── Serialización ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convierte a diccionario para enviar por la API."""
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> Vector2D:
        """Crea un vector desde un diccionario de la API."""
        return cls(x=data["x"], y=data["y"])

    @classmethod
    def zero(cls) -> Vector2D:
        """Vector cero: (0, 0). Sin magnitud ni dirección."""
        return cls(0.0, 0.0)

    def __repr__(self) -> str:
        return f"Vector2D(x={self.x:.6g}, y={self.y:.6g})"


@dataclass(frozen=True)
class Vector3D:
    """Vector en el espacio tridimensional (x, y, z).

    Extiende Vector2D a 3 dimensiones. La misma física,
    pero ahora r = (x, y, z) y las fuerzas tienen componente z.

    La ley de Coulomb NO cambia en 3D:
        F = k·q₁·q₂/r²·r̂
    Solo que r² = x² + y² + z² y r̂ tiene tres componentes.
    """

    x: float
    y: float
    z: float

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3D:
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3D:
        return self.__mul__(scalar)

    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)

    def __truediv__(self, scalar: float) -> Vector3D:
        if scalar == 0:
            raise ZeroDivisionError("No se puede dividir un vector por cero")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    @property
    def magnitude(self) -> float:
        """||v|| = √(x² + y² + z²)"""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    @property
    def magnitude_squared(self) -> float:
        """||v||² = x² + y² + z²"""
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    @property
    def unit(self) -> Vector3D:
        """v̂ = v / ||v||"""
        mag = self.magnitude
        if mag < 1e-15:
            raise ValueError(
                "El vector cero no tiene dirección definida."
            )
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)

    def dot(self, other: Vector3D) -> float:
        """a·b = ax·bx + ay·by + az·bz"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        """Producto cruz: a × b

        Resultado es un vector perpendicular a ambos.
        Importante en electromagnetismo: ∇ × E = -∂B/∂t
        """
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: Vector3D) -> float:
        return (self - other).magnitude

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data: dict) -> Vector3D:
        return cls(x=data["x"], y=data["y"], z=data["z"])

    @classmethod
    def zero(cls) -> Vector3D:
        return cls(0.0, 0.0, 0.0)

    def __repr__(self) -> str:
        return f"Vector3D(x={self.x:.6g}, y={self.y:.6g}, z={self.z:.6g})"
