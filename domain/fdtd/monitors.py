"""Monitores de campo en el dominio de la frecuencia mediante acumulación DFT en línea y flujo de Poynting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FrequencyDomainMonitor:
    """Acumula la Transformada Discreta de Fourier (DFT) en línea sobre una línea transversal x = x_idx."""

    x_idx: int
    Ny: int
    dt: float
    dy: float
    freqs_hz: np.ndarray       # Arreglo 1D de frecuencias a evaluar [Hz]

    # Matrices acumuladoras complejas [num_freqs, Ny]
    Ez_f: np.ndarray = field(init=False)
    Hy_f: np.ndarray = field(init=False)

    def __post_init__(self):
        self.freqs_hz = np.asarray(self.freqs_hz, dtype=np.float64)
        K = len(self.freqs_hz)
        self.Ez_f = np.zeros((K, self.Ny), dtype=np.complex128)
        self.Hy_f = np.zeros((K, self.Ny), dtype=np.complex128)

    def reset(self) -> None:
        self.Ez_f.fill(0.0)
        self.Hy_f.fill(0.0)

    def accumulate(self, Ez_col: np.ndarray, Hy_col: np.ndarray, time_step: int) -> None:
        """Acumula un paso temporal n. Respeta el desfase temporal de medio paso de Yee."""
        t_E = time_step * self.dt
        t_H = (time_step + 0.5) * self.dt

        # e^(-i * 2pi * f * t) * dt
        # Forma vectorizada [num_freqs, 1]
        phase_E = np.exp(-1j * (2.0 * math.pi * self.freqs_hz * t_E)) * self.dt
        phase_H = np.exp(-1j * (2.0 * math.pi * self.freqs_hz * t_H)) * self.dt

        self.Ez_f += phase_E[:, np.newaxis] * Ez_col[np.newaxis, :]
        self.Hy_f += phase_H[:, np.newaxis] * Hy_col[np.newaxis, :]

    def compute_poynting_power(self) -> np.ndarray:
        """Calcula el flujo de potencia espectral P(f) [Watts] integrado a lo largo del eje transversal y.
        
        En polarización TM (Ez, Hx, Hy):
          S = E x H = (Ez z^) x (Hx x^ + Hy y^) = -Ez*Hy x^ + Ez*Hx y^
          Sx = -(1/2) * Re { Ez * conj(Hy) }
        Flujo de propagación hacia +x: integral Sx dy.
        """
        sx = -0.5 * np.real(self.Ez_f * np.conj(self.Hy_f)) # [K, Ny]
        power = np.sum(sx, axis=1) * self.dy              # [K]
        return power


@dataclass(frozen=True)
class PoyntingSpectrum:
    """Espectro de coeficientes de transmisión T(f), reflexión R(f) y absorción A(f)."""

    frequencies_ghz: list[float]
    transmission: list[float]
    reflection: list[float]
    absorption: list[float]
    transmission_db: list[float]

    @classmethod
    def from_powers(
        cls,
        freqs_hz: np.ndarray,
        p_inc: np.ndarray,
        p_trans: np.ndarray,
        p_refl: np.ndarray,
    ) -> PoyntingSpectrum:
        """Calcula T, R, A a partir de los flujos de potencia medidos en los monitores."""
        freqs_ghz = [float(f * 1e-9) for f in freqs_hz]
        trans_list: list[float] = []
        refl_list: list[float] = []
        abs_list: list[float] = []
        tdb_list: list[float] = []

        peak_inc = float(np.max(p_inc))
        eps = max(1e-6 * peak_inc, 1e-35)

        for pi, pt, pr in zip(p_inc, p_trans, p_refl):
            if pi > eps:
                t_val = float(pt) / float(pi)
                # El flujo reflejado en la región SF viaja hacia -x (Sx < 0), por lo que la potencia es -pr
                r_val = -float(pr) / float(pi)
            else:
                t_val = 0.0
                r_val = 0.0

            a_val = 1.0 - t_val - r_val
            t_db = 10.0 * math.log10(max(t_val, 1e-12))

            trans_list.append(round(t_val, 5))
            refl_list.append(round(r_val, 5))
            abs_list.append(round(a_val, 5))
            tdb_list.append(round(t_db, 2))

        return cls(
            frequencies_ghz=freqs_ghz,
            transmission=trans_list,
            reflection=refl_list,
            absorption=abs_list,
            transmission_db=tdb_list,
        )
