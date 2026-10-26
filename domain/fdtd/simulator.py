"""Simulador electromagnético FDTD 2D integrado con rigor físico para metamateriales."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from domain.units import SPEED_OF_LIGHT, MU_0, EPSILON_0
from .grid import YeeGrid2D
from .materials import MaterialMap
from .boundaries import BoundaryController
from .sources import TFSFSource, ModulatedGaussianPulse
from .monitors import FrequencyDomainMonitor, PoyntingSpectrum


@dataclass
class FDTDSimulation:
    """Motor integral de simulación FDTD 2D sobre rejilla de Yee con fronteras CPML y TF/SF."""

    Nx: int = 240
    Ny: int = 20
    dx: float = 0.0005           # 0.5 mm de resolución espacial por defecto
    dy: float = 0.0005           # 0.5 mm (celda transversal Ny * dy = 10 mm)
    courant_factor: float = 0.70 # S = 0.70 (Estabilidad CFL garantizada < 1/sqrt(2) = 0.7071)
    pml_thickness: int = 14
    i_tfsf: int = 30             # Posición de inyección TF/SF
    f_min_ghz: float = 5.0
    f_max_ghz: float = 15.0
    num_freq_points: int = 101

    # Componentes del motor físico
    grid: YeeGrid2D = field(init=False)
    materials: MaterialMap = field(init=False)
    boundaries: BoundaryController = field(init=False)
    source: TFSFSource = field(init=False)

    # Monitores espectrales
    freqs_hz: np.ndarray = field(init=False)
    monitor_refl: FrequencyDomainMonitor = field(init=False)
    monitor_trans: FrequencyDomainMonitor = field(init=False)

    # Monitores de calibración de potencia incidente en vacío
    p_inc_calibrated: np.ndarray | None = field(default=None, init=False)

    # Registro de energía
    max_energy_recorded: float = field(default=0.0, init=False)
    energy_history: list[float] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.grid = YeeGrid2D(
            Nx=self.Nx,
            Ny=self.Ny,
            dx=self.dx,
            dy=self.dy,
            courant_factor=self.courant_factor,
        )
        self.materials = MaterialMap(
            Nx=self.Nx,
            Ny=self.Ny,
            dx=self.dx,
            dy=self.dy,
            dt=self.grid.dt,
        )
        self.boundaries = BoundaryController(
            grid_nx=self.Nx,
            grid_ny=self.Ny,
            dx=self.dx,
            dy=self.dy,
            dt=self.grid.dt,
            pml_thickness=self.pml_thickness,
        )
        self.source = TFSFSource(
            Nx=self.Nx,
            dx=self.dx,
            dt=self.grid.dt,
            i_tfsf=self.i_tfsf,
            pulse=ModulatedGaussianPulse(
                f_center_ghz=0.5 * (self.f_min_ghz + self.f_max_ghz),
                f_bandwidth_ghz=(self.f_max_ghz - self.f_min_ghz),
            ),
        )

        self.freqs_hz = np.linspace(
            self.f_min_ghz * 1e9,
            self.f_max_ghz * 1e9,
            self.num_freq_points,
            dtype=np.float64,
        )

        # Monitores de reflexión (en región SF) y transmisión (cerca de la PML derecha)
        x_refl = self.i_tfsf - 5
        x_trans = self.Nx - self.pml_thickness - 6
        self.monitor_refl = FrequencyDomainMonitor(
            x_idx=x_refl, Ny=self.Ny, dt=self.grid.dt, dy=self.dy, freqs_hz=self.freqs_hz
        )
        self.monitor_trans = FrequencyDomainMonitor(
            x_idx=x_trans, Ny=self.Ny, dt=self.grid.dt, dy=self.dy, freqs_hz=self.freqs_hz
        )

    def reset(self) -> None:
        """Reinicia campos, monitores y fuentes sin alterar la geometría de materiales."""
        self.grid.reset()
        self.boundaries.reset()
        self.source.reset()
        self.monitor_refl.reset()
        self.monitor_trans.reset()
        self.max_energy_recorded = 0.0
        self.energy_history.clear()

    def calibrate_incident_wave(self, max_steps: int = 1800) -> np.ndarray:
        """Ejecuta una corrida rápida en vacío homogéneo para registrar el flujo incidente exacto P_inc(f)."""
        # Guardar estado material actual
        old_eps = self.materials.eps_r.copy()
        old_sigma = self.materials.sigma.copy()
        old_pec = self.materials.pec_mask.copy()

        # Vacío puro
        self.materials.reset_to_vacuum()
        self.reset()

        # En vacío, el monitor en x_trans mide la onda incidente pura P_inc(f)
        self.run_until_decay(max_steps=max_steps, energy_threshold=1e-5)
        self.p_inc_calibrated = self.monitor_trans.compute_poynting_power().copy()

        # Restaurar estado material
        self.materials.eps_r = old_eps
        self.materials.sigma = old_sigma
        self.materials.pec_mask = old_pec
        self.materials.recompute_coefficients()
        self.reset()

        return self.p_inc_calibrated

    def step(self) -> None:
        """Ejecuta un ciclo completo de actualización de salto de rana (Yee Leap-Frog) de 2.º orden."""
        g = self.grid
        m = self.materials
        cpml = self.boundaries.cpml
        tfsf = self.source
        L = cpml.pml_thickness

        # ═════════════════════════════════════════════════════════════
        # 1. ACTUALIZACIÓN DE CAMPOS MAGNÉTICOS H (t = n + 1/2)
        # ═════════════════════════════════════════════════════════════
        # 1.1 Avanzar Hy auxiliar de la fuente 1D
        tfsf.step_1d_half_H()

        # 1.2 Actualización de Hx: ∂Hx/∂t = -(1/μ0) * ∂Ez/∂y
        # Condición de frontera periódica exacta en y usando np.roll
        dEz_dy = (np.roll(g.Ez, -1, axis=1) - g.Ez)
        g.Hx -= m.Dby * dEz_dy

        # 1.3 Actualización de Hy: ∂Hy/∂t = +(1/μ0) * ∂Ez/∂x
        # Derivada espacial en x: Ez[i+1, j] - Ez[i, j]
        dEz_dx = (g.Ez[1:, :] - g.Ez[:-1, :])

        # Actualización estándar de Hy en el interior
        g.Hy[:-1, :] += m.Dbx * dEz_dx

        # Corrección CPML en x para Hy (extremo izquierdo: i in 0 .. L-1)
        dEz_dx_left = dEz_dx[:L, :]
        cpml.psi_Hy_left = cpml.bh_left * cpml.psi_Hy_left + cpml.ah_left * dEz_dx_left
        g.Hy[:L, :] += m.Dbx * (dEz_dx_left / cpml.kh_left - dEz_dx_left + cpml.psi_Hy_left)

        # Corrección CPML en x para Hy (extremo derecho: i in Nx - 1 - L .. Nx - 2)
        dEz_dx_right = dEz_dx[-L:, :]
        cpml.psi_Hy_right = cpml.bh_right * cpml.psi_Hy_right + cpml.ah_right * dEz_dx_right
        g.Hy[-L-1:-1, :] += m.Dbx * (dEz_dx_right / cpml.kh_right - dEz_dx_right + cpml.psi_Hy_right)

        # 1.4 Corrección de frontera TF/SF para Hy
        tfsf.correct_2d_H(g.Hy)

        # ═════════════════════════════════════════════════════════════
        # 2. ACTUALIZACIÓN DE CAMPO ELÉCTRICO Ez (t = n + 1)
        # ═════════════════════════════════════════════════════════════
        # 2.1 Avanzar Ez auxiliar de la fuente 1D
        tfsf.step_1d_E()

        # 2.2 Rotacional de H en 2D: ∂Hy/∂x - ∂Hx/∂y
        # ∂Hy/∂x
        dHy_dx = g.Hy[1:-1, :] - g.Hy[:-2, :]
        # ∂Hx/∂y con frontera periódica transversal
        dHx_dy = g.Hx[1:-1, :] - np.roll(g.Hx[1:-1, :], 1, axis=1)

        # Actualización estándar de Ez
        g.Ez[1:-1, :] = (
            m.Ca[1:-1, :] * g.Ez[1:-1, :]
            + m.Cbx[1:-1, :] * dHy_dx
            - m.Cby[1:-1, :] * dHx_dy
        )

        # Corrección CPML en x para Ez (extremo izquierdo: celdas 1 .. L)
        dHy_dx_left = dHy_dx[:L, :]
        cpml.psi_Ez_left = cpml.be_left * cpml.psi_Ez_left + cpml.ae_left * dHy_dx_left
        g.Ez[1:L+1, :] += m.Cbx[1:L+1, :] * (dHy_dx_left / cpml.ke_left - dHy_dx_left + cpml.psi_Ez_left)

        # Corrección CPML en x para Ez (extremo derecho: celdas Nx - 1 - L .. Nx - 2)
        dHy_dx_right = dHy_dx[-L:, :]
        cpml.psi_Ez_right = cpml.be_right * cpml.psi_Ez_right + cpml.ae_right * dHy_dx_right
        g.Ez[-L-1:-1, :] += m.Cbx[-L-1:-1, :] * (dHy_dx_right / cpml.ke_right - dHy_dx_right + cpml.psi_Ez_right)

        # 2.3 Corrección de frontera TF/SF para Ez
        tfsf.correct_2d_E(g.Ez)

        # 2.4 Máscara de conductores perfectos PEC: Ez = 0
        g.Ez[m.pec_mask] = 0.0

        # ═════════════════════════════════════════════════════════════
        # 3. REGISTRO EN MONITORES Y CONTROL DE ENERGÍA
        # ═════════════════════════════════════════════════════════════
        step_n = g.time_step
        # Monitor de reflexión
        xr = self.monitor_refl.x_idx
        self.monitor_refl.accumulate(g.Ez[xr, :], g.Hy[xr, :], step_n)

        # Monitor de transmisión
        xt = self.monitor_trans.x_idx
        self.monitor_trans.accumulate(g.Ez[xt, :], g.Hy[xt, :], step_n)

        # Seguimiento de energía
        u_now = g.total_energy(m.eps_r)
        if u_now > self.max_energy_recorded:
            self.max_energy_recorded = u_now
        self.energy_history.append(u_now)

        # Incrementar reloj
        g.time_step += 1
        g.current_time += g.dt

    def run_steps(self, num_steps: int) -> None:
        """Ejecuta un número fijo de pasos temporales."""
        for _ in range(num_steps):
            self.step()

    def run_until_decay(self, max_steps: int = 2500, energy_threshold: float = 1e-4) -> int:
        """Avanza la simulación hasta que la energía electromagnética caiga por debajo del umbral o alcance max_steps."""
        steps_run = 0
        # Tiempo mínimo físico: duración de inyección (2*t0) + tiempo de tránsito por toda la caja (Nx / S)
        transit_steps = int(self.Nx / max(self.grid.courant_factor, 0.1))
        injection_steps = int(2.2 * self.source.pulse.t0 / self.grid.dt)
        min_steps_before_check = injection_steps + transit_steps + 40

        for _ in range(max_steps):
            self.step()
            steps_run += 1
            if steps_run > min_steps_before_check and self.max_energy_recorded > 1e-18:
                current_energy = self.energy_history[-1]
                if current_energy / self.max_energy_recorded < energy_threshold:
                    break

        return steps_run

    def get_spectrum(self) -> PoyntingSpectrum:
        """Calcula y retorna el espectro físico de Poynting T(f), R(f) y A(f)."""
        if self.p_inc_calibrated is None:
            # Calibración automática si no se había ejecutado
            self.calibrate_incident_wave()

        p_refl = self.monitor_refl.compute_poynting_power()
        p_trans = self.monitor_trans.compute_poynting_power()

        return PoyntingSpectrum.from_powers(
            freqs_hz=self.freqs_hz,
            p_inc=self.p_inc_calibrated,
            p_trans=p_trans,
            p_refl=p_refl,
        )
