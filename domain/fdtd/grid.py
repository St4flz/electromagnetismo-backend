"""Estructura de datos y malla espacial de Yee en 2D (Modo TM: Ez, Hx, Hy)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from domain.units import SPEED_OF_LIGHT, MU_0, EPSILON_0


@dataclass
class YeeGrid2D:
    """Rejilla discreta 2D de Kane Yee para modo TM.
    
    Variables de campo:
      Ez[i, j]:      Campo eléctrico en (i, j) en tiempo t = n * dt
      Hx[i, j]:      Campo magnético en (i, j + 1/2) en tiempo t = (n + 1/2) * dt
      Hy[i, j]:      Campo magnético en (i + 1/2, j) en tiempo t = (n + 1/2) * dt
    """

    Nx: int                      # Celdas en dirección de propagación (x)
    Ny: int                      # Celdas en dirección transversal periódica (y)
    dx: float                    # Resolución espacial en x [m]
    dy: float                    # Resolución espacial en y [m]
    courant_factor: float = 0.5  # S = c * dt * sqrt(1/dx^2 + 1/dy^2) <= 1.0 (en 2D S_max = 1.0 para CFL general)

    # Campos discretizados
    Ez: np.ndarray = field(init=False)
    Hx: np.ndarray = field(init=False)
    Hy: np.ndarray = field(init=False)

    dt: float = field(init=False)
    time_step: int = field(default=0, init=False)
    current_time: float = field(default=0.0, init=False)

    def __post_init__(self):
        if self.Nx < 10 or self.Ny < 2:
            raise ValueError(f"Dimensiones de grilla insuficientes: Nx={self.Nx}, Ny={self.Ny}")

        # Factor canónico de Courant: S = c * dt / dx
        # Condición de estabilidad CFL en 2D: S <= 1 / sqrt(1 + (dx/dy)^2)
        # En grilla cuadrada dx = dy: S_max = 1 / sqrt(2) ≈ 0.70710678
        self.dt = (self.courant_factor * self.dx) / SPEED_OF_LIGHT

        # Asignación contigua en memoria para máximo rendimiento en NumPy
        self.Ez = np.zeros((self.Nx, self.Ny), dtype=np.float64, order="C")
        self.Hx = np.zeros((self.Nx, self.Ny), dtype=np.float64, order="C")
        self.Hy = np.zeros((self.Nx, self.Ny), dtype=np.float64, order="C")

    def reset(self) -> None:
        """Reinicia todos los campos y el reloj a cero."""
        self.Ez.fill(0.0)
        self.Hx.fill(0.0)
        self.Hy.fill(0.0)
        self.time_step = 0
        self.current_time = 0.0

    @property
    def Lx(self) -> float:
        """Longitud física en x [m]."""
        return self.Nx * self.dx

    @property
    def Ly(self) -> float:
        """Longitud física en y [m]."""
        return self.Ny * self.dy

    def total_energy(self, eps_r: np.ndarray) -> float:
        """Calcula la energía electromagnética total U [Joules] en la rejilla 2D.
        
        U = (1/2) * integral (epsilon * Ez^2 + mu0 * (Hx^2 + Hy^2)) dx dy
        """
        cell_area = self.dx * self.dy
        u_elec = 0.5 * np.sum((eps_r * EPSILON_0) * (self.Ez ** 2)) * cell_area
        u_mag = 0.5 * MU_0 * np.sum((self.Hx ** 2) + (self.Hy ** 2)) * cell_area
        return float(u_elec + u_mag)
