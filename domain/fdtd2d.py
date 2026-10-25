"""Solucionador Electrodinámico FDTD 2D y Celda de Yee (Capítulo 9).

Implementa el método canónico de Kane Yee (1966) en diferencias finitas en el dominio
del tiempo para resolver las ecuaciones de Maxwell acopladas en 2D (modo TM: Ez, Hx, Hy).

Formulación Matemática Discreta (Polarización TM):
    ∂Hx/∂t = - (1/μ) · ∂Ez/∂y
    ∂Hy/∂t = + (1/μ) · ∂Ez/∂x
    ∂Ez/∂t = + (1/ε) · [∂Hy/∂x - ∂Hx/∂y - σ · Ez]

Condición de Estabilidad de Courant-Friedrichs-Lewy (CFL):
    c · Δt ≤ 1 / √(1/Δx² + 1/Δy²)
    En grilla cuadrada (Δx = Δy = Δ):
        S = c · Δt / Δ ≤ 1 / √2 ≈ 0.70710678

Condiciones de Frontera:
    - Mur ABC de 1er orden en los bordes de propagación x=0 y x=Nx-1 para absorber
      ondas salientes sin reflexiones artificiales en la caja computacional.
    - Fuentes de excitación: Pulso Gaussiano de banda ancha o señal armónica continua.
    - Espectro de transmisión T(f) mediante Transformada Rápida de Fourier (FFT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .units import SPEED_OF_LIGHT, EPSILON_0, MU_0


@dataclass
class FDTD2DSimulator:
    """Simulador bidimensional FDTD sobre rejilla escalonada de Yee (TM: Ez, Hx, Hy)."""

    Nx: int = 80             # Celdas espaciales en el eje x (propagación)
    Ny: int = 50             # Celdas espaciales en el eje y (transversal)
    dx: float = 0.002        # Resolución espacial Δx = Δy en metros [m] (2 mm)
    courant_factor: float = 0.5  # Factor de Courant S = c·Δt / Δx (Debe ser ≤ 1/√2 ≈ 0.707)
    source_type: Literal["gaussian", "sine"] = "gaussian"
    source_freq_ghz: float = 10.0   # Frecuencia central de microondas (10 GHz)
    source_x: int = 15              # Posición de la línea de inyección en x

    # Campos electromagnéticos discretizados
    Ez: np.ndarray = field(init=False)
    Hx: np.ndarray = field(init=False)
    Hy: np.ndarray = field(init=False)

    # Coeficientes materiales en cada celda
    eps_r: np.ndarray = field(init=False)
    sigma: np.ndarray = field(init=False)
    Ca: np.ndarray = field(init=False)
    Cb: np.ndarray = field(init=False)
    Db: float = field(init=False)

    # Fronteras absorbentes Mur ABC
    ez_left_prev: np.ndarray = field(init=False)
    ez_right_prev: np.ndarray = field(init=False)

    # Monitores de registro temporal
    time_step: int = 0
    current_time: float = 0.0
    dt: float = field(init=False)
    trans_probe_x: int = field(init=False)
    refl_probe_x: int = field(init=False)
    time_history_trans: list[float] = field(default_factory=list)
    time_history_refl: list[float] = field(default_factory=list)

    def __post_init__(self):
        # Paso temporal condicionado por CFL
        self.dt = (self.courant_factor * self.dx) / SPEED_OF_LIGHT

        # Inicialización de matrices de campo
        self.Ez = np.zeros((self.Nx, self.Ny), dtype=np.float64)
        self.Hx = np.zeros((self.Nx, self.Ny), dtype=np.float64)
        self.Hy = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        # Medios materiales (vacío por defecto: ε_r = 1, σ = 0)
        self.eps_r = np.ones((self.Nx, self.Ny), dtype=np.float64)
        self.sigma = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        # Buffers para condición absorbente de Mur
        self.ez_left_prev = np.zeros(self.Ny, dtype=np.float64)
        self.ez_right_prev = np.zeros(self.Ny, dtype=np.float64)

        self.trans_probe_x = self.Nx - 12
        self.refl_probe_x = self.source_x - 5

        self._recompute_coefficients()

    def _recompute_coefficients(self):
        """Precomputa coeficientes Ca, Cb, Db optimizados para NumPy."""
        eps = self.eps_r * EPSILON_0
        mu = MU_0

        # Ca = (1 - σ·dt / 2ε) / (1 + σ·dt / 2ε)
        # Cb = (dt / ε·dx) / (1 + σ·dt / 2ε)
        loss = (self.sigma * self.dt) / (2.0 * eps)
        self.Ca = (1.0 - loss) / (1.0 + loss)
        self.Cb = (self.dt / (eps * self.dx)) / (1.0 + loss)

        # Db = dt / (μ·dx)
        self.Db = self.dt / (mu * self.dx)

    def add_dielectric_slab(self, start_x: int, end_x: int, relative_permittivity: float = 4.0):
        """Inserta una lámina dieléctrica de permitividad ε_r entre start_x y end_x."""
        self.eps_r[start_x:end_x, :] = relative_permittivity
        self._recompute_coefficients()

    def add_metamaterial_periodic_array(
        self,
        start_x: int,
        periods: int = 4,
        cell_width: int = 4,
        rod_width: int = 2,
        rod_eps: float = 9.0,
    ):
        """Inserta un arreglo periódico de varillas/bloques dieléctricos de alta permitividad."""
        for p in range(periods):
            x0 = start_x + p * cell_width
            x1 = x0 + rod_width
            if x1 < self.Nx:
                # Arreglo periódico de inclusiones
                for y in range(8, self.Ny - 8, 8):
                    self.eps_r[x0:x1, y:y+4] = rod_eps
        self._recompute_coefficients()

    def reset_fields(self):
        """Reinicia todos los campos electromagnéticos a cero."""
        self.Ez.fill(0.0)
        self.Hx.fill(0.0)
        self.Hy.fill(0.0)
        self.ez_left_prev.fill(0.0)
        self.ez_right_prev.fill(0.0)
        self.time_step = 0
        self.current_time = 0.0
        self.time_history_trans.clear()
        self.time_history_refl.clear()

    def _source_value(self) -> float:
        """Calcula el valor instantáneo de la fuente de excitación."""
        t = self.current_time
        f0 = self.source_freq_ghz * 1e9

        if self.source_type == "gaussian":
            # Pulso Gaussiano de banda ancha centrado en t0
            t0 = 4.0 / f0
            tau = 1.0 / (2.0 * f0)
            return math.exp(-((t - t0)**2) / (2.0 * (tau**2)))
        else:
            # Onda senoidal continua suavemente encendida
            ramp = min(t * f0 * 0.25, 1.0)
            return ramp * math.sin(2.0 * math.pi * f0 * t)

    def step(self, num_steps: int = 1):
        """Ejecuta 'num_steps' iteraciones completas de Yee Leap-Frog vectorizadas."""
        mur_coeff = (SPEED_OF_LIGHT * self.dt - self.dx) / (SPEED_OF_LIGHT * self.dt + self.dx)

        for _ in range(num_steps):
            # 1. Actualización de campo magnético Hx en salto de rana (t + dt/2)
            # Hx[i, j] = Hx[i, j] - Db * (Ez[i, j+1] - Ez[i, j])
            self.Hx[:, :-1] -= self.Db * (self.Ez[:, 1:] - self.Ez[:, :-1])

            # 2. Actualización de campo magnético Hy en salto de rana (t + dt/2)
            # Hy[i, j] = Hy[i, j] + Db * (Ez[i+1, j] - Ez[i, j])
            self.Hy[:-1, :] += self.Db * (self.Ez[1:, :] - self.Ez[:-1, :])

            # Guardar valores de frontera para Mur ABC antes de actualizar Ez
            ez_x1_old = self.Ez[1, :].copy()
            ez_xend1_old = self.Ez[self.Nx - 2, :].copy()

            # 3. Actualización de campo eléctrico Ez en salto de rana (t + dt)
            # Ez[i, j] = Ca * Ez + Cb * [(Hy[i, j] - Hy[i-1, j]) - (Hx[i, j] - Hx[i, j-1])]
            curl_h = (
                (self.Hy[1:-1, 1:-1] - self.Hy[:-2, 1:-1])
                - (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, :-2])
            )
            self.Ez[1:-1, 1:-1] = (
                self.Ca[1:-1, 1:-1] * self.Ez[1:-1, 1:-1]
                + self.Cb[1:-1, 1:-1] * curl_h
            )

            # 4. Inyección de la fuente (línea suave transversal)
            val = self._source_value()
            self.Ez[self.source_x, 10:-10] += val

            # 5. Fronteras absorbentes Mur ABC de 1er orden en x=0 y x=Nx-1
            # Ez_0^{n+1} = Ez_1^n + mur_coeff * (Ez_1^{n+1} - Ez_0^n)
            self.Ez[0, :] = self.ez_left_prev + mur_coeff * (self.Ez[1, :] - self.Ez[0, :])
            self.Ez[-1, :] = self.ez_right_prev + mur_coeff * (self.Ez[-2, :] - self.Ez[-1, :])

            self.ez_left_prev = ez_x1_old
            self.ez_right_prev = ez_xend1_old

            # 6. Registro en monitores
            mid_y = self.Ny // 2
            self.time_history_trans.append(float(self.Ez[self.trans_probe_x, mid_y]))
            self.time_history_refl.append(float(self.Ez[self.refl_probe_x, mid_y]))

            self.time_step += 1
            self.current_time += self.dt

    def get_field_slice_subsampled(self, step_stride: int = 2) -> list[list[float]]:
        """Retorna una matriz 2D submuestreada de Ez lista para streaming por WebSocket/UI."""
        sub = self.Ez[::step_stride, ::step_stride]
        # Redondear para transmisión compacta en JSON
        return [[round(float(val), 4) for val in row] for row in sub]

    def compute_transmission_spectrum(self, max_freq_ghz: float = 20.0) -> list[dict]:
        """Calcula el espectro T(f) mediante FFT de la señal registrada en el monitor."""
        if len(self.time_history_trans) < 32:
            return []

        signal = np.array(self.time_history_trans, dtype=np.float64)
        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, d=self.dt) * 1e-9  # GHz

        # Normalizar espectro al pico máximo
        peak = np.max(fft_vals) if np.max(fft_vals) > 0 else 1.0
        normalized_fft = fft_vals / peak

        spectrum: list[dict] = []
        for f, mag in zip(freqs, normalized_fft):
            if 1.0 <= f <= max_freq_ghz:
                spectrum.append({
                    "frequencyGhz": round(float(f), 2),
                    "transmissionMagnitude": round(float(mag), 4),
                    "powerTransmission": round(float(mag**2), 4),
                })

        return spectrum


# ═══════════════════════════════════════════════════════════════════
# GUIONES DE COMPROBACIÓN FÍSICA ANALÍTICA (CAPÍTULO 9)
# ═══════════════════════════════════════════════════════════════════

def verify_cfl_stability_condition(courant_factor: float) -> dict:
    """Guión comprobable 1: Demuestra el teorema CFL en 2D.

    - Si S ≤ 1/√2 (≈ 0.7071): La solución numérica permanece acotada y estable.
    - Si S > 1/√2 (ej. 1.05): La solución FDTD explota numéricamente (|Ez| -> ∞).
    """
    sim = FDTD2DSimulator(
        Nx=40,
        Ny=30,
        dx=0.005,
        courant_factor=courant_factor,
        source_type="gaussian",
    )

    is_unstable = False
    max_field = 0.0

    for _ in range(60):
        sim.step(1)
        current_max = float(np.max(np.abs(sim.Ez)))
        if current_max > max_field:
            max_field = current_max
        if math.isnan(current_max) or math.isinf(current_max) or current_max > 1e4:
            is_unstable = True
            break

    cfl_limit_2d = 1.0 / math.sqrt(2.0)  # ≈ 0.7071
    theoretically_stable = courant_factor <= cfl_limit_2d

    return {
        "courant_factor": courant_factor,
        "cfl_limit_2d": cfl_limit_2d,
        "theoretically_stable": theoretically_stable,
        "numerically_stable": not is_unstable,
        "max_field_recorded": max_field if not is_unstable else "EXPLOSIÓN_NUMÉRICA_INF",
        "theorem_verified": (not is_unstable) == theoretically_stable,
    }


def verify_fdtd_pulse_velocity(Nx: int = 80, Ny: int = 40, dx: float = 0.002) -> dict:
    """Guión comprobable 2: Mide la velocidad de grupo de un pulso gaussiano en vacío.

    Registra el tiempo requerido para que el pico del pulso viaje entre dos sondas
    separadas una distancia Δx = (x₂ - x₁) · dx y lo compara con c = 2.998e8 m/s.
    """
    src_x = int(0.15 * Nx)
    x1 = int(0.30 * Nx)
    x2 = int(0.75 * Nx)
    sim = FDTD2DSimulator(Nx=Nx, Ny=Ny, dx=dx, courant_factor=0.5, source_type="gaussian", source_x=src_x)
    distance_m = (x2 - x1) * dx

    mid_y = Ny // 2
    history_x1: list[float] = []
    history_x2: list[float] = []

    total_steps = int((x2 / 0.5) * 1.8)
    for _ in range(total_steps):
        sim.step(1)
        history_x1.append(float(sim.Ez[x1, mid_y]))
        history_x2.append(float(sim.Ez[x2, mid_y]))

    def _find_peak_substep(history: list[float]) -> float:
        idx = int(np.argmax(history))
        if 0 < idx < len(history) - 1:
            y0, y1, y2 = history[idx - 1], history[idx], history[idx + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                offset = (y0 - y2) / (2.0 * denom)
                return idx + offset
        return float(idx)

    t1_sub = _find_peak_substep(history_x1)
    t2_sub = _find_peak_substep(history_x2)
    delta_t = (t2_sub - t1_sub) * sim.dt

    measured_c = distance_m / delta_t if delta_t > 0 else 0.0
    abs_err = abs(measured_c - SPEED_OF_LIGHT)
    rel_err = (abs_err / SPEED_OF_LIGHT) * 100.0

    return {
        "travel_distance_m": distance_m,
        "delta_time_seconds": delta_t,
        "measured_pulse_speed_m_s": measured_c,
        "speed_of_light_constant": SPEED_OF_LIGHT,
        "dispersion_error_percent": rel_err,
        "is_verified": rel_err < 5.5,  # Dispersión numérica física esperada en rejilla con 10 celdas por longitud de onda
    }
