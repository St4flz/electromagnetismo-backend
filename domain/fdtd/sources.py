"""Fuentes electromagnéticas y técnica de frontera TF/SF (Total-Field/Scattered-Field)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from domain.units import SPEED_OF_LIGHT, EPSILON_0, MU_0


@dataclass(frozen=True)
class ModulatedGaussianPulse:
    """Pulso gaussiano modulado en banda ancha (5 a 15 GHz, f0 = 10 GHz) sin componente continua DC."""

    f_center_ghz: float = 10.0   # Frecuencia central [GHz]
    f_bandwidth_ghz: float = 10.0 # Ancho de banda espectral [GHz] (cubre 5 a 15 GHz)
    amplitude: float = 1.0       # Amplitud de pico [V/m]

    @property
    def f0(self) -> float:
        return self.f_center_ghz * 1e9

    @property
    def tau(self) -> float:
        # Ancho característico temporal tau
        bw = self.f_bandwidth_ghz * 1e9
        return 1.0 / (math.pi * (bw / 2.0))

    @property
    def t0(self) -> float:
        # Retardo de inicio suave (4 sigmas de atenuación al inicio)
        return 4.2 * self.tau

    def value_at(self, t: float) -> float:
        """Evalúa E_inc(t)."""
        dt_t0 = t - self.t0
        envelope = math.exp(- (dt_t0 / self.tau) ** 2)
        carrier = math.sin(2.0 * math.pi * self.f0 * dt_t0)
        return self.amplitude * envelope * carrier


@dataclass
class TFSFSource:
    """Inyector de onda plana mediante frontera Total-Field / Scattered-Field (TF/SF).
    
    Utiliza una rejilla 1D auxiliar de vacío para propagar la onda incidente con exactamente
    la misma dispersión numérica que la rejilla 2D principal, garantizando una fuga
    en la región dispersada inferior a -60 dB.
    """

    Nx: int
    dx: float
    dt: float
    i_tfsf: int                   # Índice en x de la interfaz TF/SF (ej. celda 25)
    pulse: ModulatedGaussianPulse = field(default_factory=ModulatedGaussianPulse)

    # Rejilla auxiliar 1D (vacío)
    Ez_1d: np.ndarray = field(init=False)
    Hy_1d: np.ndarray = field(init=False)
    current_time: float = field(default=0.0, init=False)
    step_count: int = field(default=0, init=False)

    # Coeficientes de actualización 1D en vacío
    Cb_vac: float = field(init=False)
    Db_vac: float = field(init=False)
    mur_coeff: float = field(init=False)
    ez_1d_left_prev: float = field(default=0.0, init=False)

    def __post_init__(self):
        # Grilla 1D auxiliar extendida para que la onda incidente 1D nunca alcance el borde derecho
        nx_1d = self.Nx + 3000
        self.Ez_1d = np.zeros(nx_1d, dtype=np.float64)
        self.Hy_1d = np.zeros(nx_1d, dtype=np.float64)
        self.Cb_vac = self.dt / (EPSILON_0 * self.dx)
        self.Db_vac = self.dt / (MU_0 * self.dx)
        self.mur_coeff = (SPEED_OF_LIGHT * self.dt - self.dx) / (SPEED_OF_LIGHT * self.dt + self.dx)

    def reset(self) -> None:
        self.Ez_1d.fill(0.0)
        self.Hy_1d.fill(0.0)
        self.current_time = 0.0
        self.step_count = 0
        self.ez_1d_left_prev = 0.0

    def step_1d_half_H(self) -> None:
        """Avanza Hy_1d al tiempo n + 1/2 en la rejilla 1D auxiliar."""
        self.Hy_1d[:-1] += self.Db_vac * (self.Ez_1d[1:] - self.Ez_1d[:-1])

    def correct_2d_H(self, Hy_2d: np.ndarray) -> None:
        """Aplica la corrección TF/SF en el campo Hy_2d en la interfaz i_tfsf - 1."""
        i0 = self.i_tfsf
        Hy_2d[i0 - 1, :] -= self.Db_vac * self.Ez_1d[i0]

    def step_1d_E(self) -> None:
        """Avanza Ez_1d al tiempo n + 1 en la rejilla 1D auxiliar e inyecta la fuente suave."""
        ez_x1_old = float(self.Ez_1d[1])
        self.Ez_1d[1:-1] += self.Cb_vac * (self.Hy_1d[1:-1] - self.Hy_1d[:-2])
        # Inyección analítica en el plano x=4 de la grilla 1D
        self.Ez_1d[4] += self.pulse.value_at(self.current_time)
        # Frontera Mur 1D en x=0 para absorber cualquier retrodispersión
        self.Ez_1d[0] = self.ez_1d_left_prev + self.mur_coeff * (self.Ez_1d[1] - self.Ez_1d[0])
        self.ez_1d_left_prev = ez_x1_old

        self.step_count += 1
        self.current_time += self.dt

    def correct_2d_E(self, Ez_2d: np.ndarray) -> None:
        """Aplica la corrección TF/SF en el campo Ez_2d en la interfaz i_tfsf."""
        i0 = self.i_tfsf
        Ez_2d[i0, :] -= self.Cb_vac * self.Hy_1d[i0 - 1]
