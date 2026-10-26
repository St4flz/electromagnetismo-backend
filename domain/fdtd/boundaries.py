"""Condiciones de frontera para FDTD: Periódica (Floquet-Bloch en y) y CPML (en x)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from domain.units import EPSILON_0, MU_0, SPEED_OF_LIGHT


@dataclass
class CPMLBoundary:
    """Implementación de Convolutional PML (Roden & Gedney 2000) en el eje de propagación x."""

    Nx: int
    Ny: int
    dt: float
    dx: float
    pml_thickness: int = 12       # Número de celdas PML en cada extremo
    m_order: float = 3.5          # Grado polinómico del perfil de conductividad
    sigma_factor: float = 1.3     # Escala de sigma_max
    kappa_max: float = 7.0        # Factor máximo de absorción de ondas evanescentes
    alpha_max: float = 0.05       # Factor CFS para absorción en bajas frecuencias

    # Coeficientes CPML y variables de memoria
    # Extremo izquierdo (x_min, celdas 0 .. pml_thickness-1)
    be_left: np.ndarray = field(init=False)
    ae_left: np.ndarray = field(init=False)
    bh_left: np.ndarray = field(init=False)
    ah_left: np.ndarray = field(init=False)
    ke_left: np.ndarray = field(init=False)
    kh_left: np.ndarray = field(init=False)
    psi_Ez_left: np.ndarray = field(init=False)
    psi_Hy_left: np.ndarray = field(init=False)

    # Extremo derecho (x_max, celdas Nx - pml_thickness .. Nx - 1)
    be_right: np.ndarray = field(init=False)
    ae_right: np.ndarray = field(init=False)
    bh_right: np.ndarray = field(init=False)
    ah_right: np.ndarray = field(init=False)
    ke_right: np.ndarray = field(init=False)
    kh_right: np.ndarray = field(init=False)
    psi_Ez_right: np.ndarray = field(init=False)
    psi_Hy_right: np.ndarray = field(init=False)

    def __post_init__(self):
        L = self.pml_thickness
        eta0 = math.sqrt(MU_0 / EPSILON_0)
        sigma_max = self.sigma_factor * (self.m_order + 1.0) / (eta0 * self.dx)

        # Perfiles continuos de profundidad rho in [0, 1] desde la interfaz hacia el exterior
        # En la interfaz (borde interior con el vacío), la profundidad y conductividad son exactamente 0
        depth_e = np.array([L - (i + 1) for i in range(L)], dtype=np.float64)
        depth_h = np.array([L - (i + 0.5) for i in range(L)], dtype=np.float64)

        rho_e = depth_e / L
        rho_h = depth_h / L

        sigma_e = sigma_max * (rho_e ** self.m_order)
        sigma_h = sigma_max * (rho_h ** self.m_order)

        kappa_e = 1.0 + (self.kappa_max - 1.0) * (rho_e ** self.m_order)
        kappa_h = 1.0 + (self.kappa_max - 1.0) * (rho_h ** self.m_order)

        alpha_e = self.alpha_max * (1.0 - rho_e)
        alpha_h = self.alpha_max * (1.0 - rho_h)

        # Coeficientes a y b para convolución temporal
        def _calc_ab(sigma, kappa, alpha):
            b = np.exp(-((sigma / kappa) + alpha) * (self.dt / EPSILON_0))
            denom = sigma * kappa + (kappa ** 2) * alpha
            a = np.zeros_like(b)
            mask = denom > 1e-12
            a[mask] = (sigma[mask] / denom[mask]) * (b[mask] - 1.0)
            return b, a

        # Extremo izquierdo
        be, ae = _calc_ab(sigma_e, kappa_e, alpha_e)
        bh, ah = _calc_ab(sigma_h, kappa_h, alpha_h)
        self.be_left = be[:, np.newaxis]
        self.ae_left = ae[:, np.newaxis]
        self.bh_left = bh[:, np.newaxis]
        self.ah_left = ah[:, np.newaxis]
        self.ke_left = kappa_e[:, np.newaxis]
        self.kh_left = kappa_h[:, np.newaxis]

        # Extremo derecho (inversión espacial del perfil)
        self.be_right = be[::-1, np.newaxis]
        self.ae_right = ae[::-1, np.newaxis]
        self.bh_right = bh[::-1, np.newaxis]
        self.ah_right = ah[::-1, np.newaxis]
        self.ke_right = kappa_e[::-1, np.newaxis]
        self.kh_right = kappa_h[::-1, np.newaxis]

        # Matrices de memoria psi
        self.psi_Ez_left = np.zeros((L, self.Ny), dtype=np.float64, order="C")
        self.psi_Hy_left = np.zeros((L, self.Ny), dtype=np.float64, order="C")
        self.psi_Ez_right = np.zeros((L, self.Ny), dtype=np.float64, order="C")
        self.psi_Hy_right = np.zeros((L, self.Ny), dtype=np.float64, order="C")

    def reset(self) -> None:
        self.psi_Ez_left.fill(0.0)
        self.psi_Hy_left.fill(0.0)
        self.psi_Ez_right.fill(0.0)
        self.psi_Hy_right.fill(0.0)


@dataclass
class BoundaryController:
    """Maneja las fronteras periódicas en y y CPML en x."""

    grid_nx: int
    grid_ny: int
    dx: float
    dy: float
    dt: float
    pml_thickness: int = 12

    cpml: CPMLBoundary = field(init=False)

    def __post_init__(self):
        self.cpml = CPMLBoundary(
            Nx=self.grid_nx,
            Ny=self.grid_ny,
            dt=self.dt,
            dx=self.dx,
            pml_thickness=self.pml_thickness,
        )

    def reset(self) -> None:
        self.cpml.reset()
