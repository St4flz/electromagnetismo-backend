"""Distribución de propiedades constitutivas del medio y coeficientes de actualización de Yee."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from domain.units import EPSILON_0, MU_0


class MaterialPreset(str, Enum):
    VACUUM = "vacuum"
    PTFE = "ptfe"          # Teflon: eps_r = 2.1, tan_delta ~ 0.0002
    FR4 = "fr4"            # PCB substrato: eps_r = 4.4, tan_delta ~ 0.02
    ALUMINA = "alumina"    # Ceramica: eps_r = 9.8, tan_delta ~ 0.0001
    PEC = "pec"            # Conductor Eléctrico Perfecto


@dataclass
class MaterialMap:
    """Distribución espacial de permitividad, conductividad y máscaras de conductores."""

    Nx: int
    Ny: int
    dx: float
    dy: float
    dt: float

    # Matrices constitutivas
    eps_r: np.ndarray = field(init=False)
    sigma: np.ndarray = field(init=False)
    pec_mask: np.ndarray = field(init=False)

    # Coeficientes precalculados de Yee
    Ca: np.ndarray = field(init=False)
    Cbx: np.ndarray = field(init=False)
    Cby: np.ndarray = field(init=False)
    Dbx: float = field(init=False)
    Dby: float = field(init=False)

    def __post_init__(self):
        self.eps_r = np.ones((self.Nx, self.Ny), dtype=np.float64, order="C")
        self.sigma = np.zeros((self.Nx, self.Ny), dtype=np.float64, order="C")
        self.pec_mask = np.zeros((self.Nx, self.Ny), dtype=bool, order="C")
        self.recompute_coefficients()

    def reset_to_vacuum(self) -> None:
        """Restaura todo el dominio a vacío homogéneo."""
        self.eps_r.fill(1.0)
        self.sigma.fill(0.0)
        self.pec_mask.fill(False)
        self.recompute_coefficients()

    def set_dielectric_block(
        self,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
        eps_relative: float,
        conductivity: float = 0.0,
    ) -> None:
        """Inserta un bloque dieléctrico homogéneo."""
        xs = max(0, min(x_start, self.Nx))
        xe = max(0, min(x_end, self.Nx))
        ys = max(0, min(y_start, self.Ny))
        ye = max(0, min(y_end, self.Ny))
        if xs < xe and ys < ye:
            self.eps_r[xs:xe, ys:ye] = eps_relative
            self.sigma[xs:xe, ys:ye] = conductivity
            self.pec_mask[xs:xe, ys:ye] = False
            self.recompute_coefficients()

    def set_pec_block(
        self,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
    ) -> None:
        """Inserta un conductor eléctrico perfecto (PEC)."""
        xs = max(0, min(x_start, self.Nx))
        xe = max(0, min(x_end, self.Nx))
        ys = max(0, min(y_start, self.Ny))
        ye = max(0, min(y_end, self.Ny))
        if xs < xe and ys < ye:
            self.pec_mask[xs:xe, ys:ye] = True
            # En celdas PEC, Ez se fuerza a 0.0 en cada paso
            self.recompute_coefficients()

    def recompute_coefficients(self) -> None:
        """Precomputa coeficientes Ca, Cbx, Cby, Dbx, Dby optimizados para NumPy."""
        eps = self.eps_r * EPSILON_0
        # loss = sigma * dt / (2 * eps)
        loss = (self.sigma * self.dt) / (2.0 * eps)
        self.Ca = (1.0 - loss) / (1.0 + loss)
        self.Cbx = (self.dt / (eps * self.dx)) / (1.0 + loss)
        self.Cby = (self.dt / (eps * self.dy)) / (1.0 + loss)

        self.Dbx = self.dt / (MU_0 * self.dx)
        self.Dby = self.dt / (MU_0 * self.dy)
