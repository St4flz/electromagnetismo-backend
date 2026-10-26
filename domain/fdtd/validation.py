"""Suite formal de validación física y matemática del simulador FDTD (Casos V1 a V11)."""

from __future__ import annotations

from typing import Any
import math
import numpy as np

from domain.units import SPEED_OF_LIGHT
from .simulator import FDTDSimulation
from .analytic import (
    airy_dielectric_slab_transmission,
    tmm_multilayer_transmission,
    yee_numerical_phase_velocity,
    wire_medium_plasma_frequency,
)
from .geometry import UnitCellGeometry, CellFamily, rasterize_unit_cell


def validate_v1_vacuum_transmission() -> dict:
    """V1: Propagación en vacío puro -> T(f) == 1.000 (error < 1.0%)."""
    sim = FDTDSimulation(Nx=180, Ny=16, dx=0.001, dy=0.001, courant_factor=0.70)
    sim.calibrate_incident_wave(max_steps=1200)

    # Correr de nuevo en vacío y medir con los monitores
    sim.run_until_decay(max_steps=1200)
    spec = sim.get_spectrum()

    # En la banda 7 a 13 GHz (donde el pulso gaussiano tiene alta densidad espectral)
    mask = [7.0 <= f <= 13.0 for f in spec.frequencies_ghz]
    t_band = [t for t, m in zip(spec.transmission, mask) if m]

    max_err_pct = max(abs(t - 1.0) * 100.0 for t in t_band)
    mean_err_pct = float(np.mean([abs(t - 1.0) * 100.0 for t in t_band]))

    return {
        "case": "V1_vacuum_transmission",
        "mean_error_percent": round(mean_err_pct, 3),
        "max_error_percent": round(max_err_pct, 3),
        "tolerance_percent": 1.0,
        "is_verified": max_err_pct < 1.0,
    }


def validate_v2_cpml_absorption() -> dict:
    """V2: Absorción de la frontera CPML -> Reflexión residual R < -40 dB."""
    sim = FDTDSimulation(Nx=200, Ny=16, dx=0.001, dy=0.001, pml_thickness=16)
    sim.calibrate_incident_wave(max_steps=1400)
    sim.run_until_decay(max_steps=1400)
    spec = sim.get_spectrum()

    # Evaluar R en dB en la banda 7 a 13 GHz
    r_vals = [r for r, f in zip(spec.reflection, spec.frequencies_ghz) if 7.0 <= f <= 13.0]
    max_refl_linear = max(r_vals)
    max_refl_db = 10.0 * math.log10(max(max_refl_linear, 1e-12))

    return {
        "case": "V2_cpml_absorption",
        "max_reflection_db": round(max_refl_db, 2),
        "threshold_db": -40.0,
        "is_verified": max_refl_db < -40.0,
    }


def validate_v3_tfsf_leakage() -> dict:
    """V3: Supresión de fuga en región Scattered-Field en vacío -> Fuga residual < -60 dB."""
    sim = FDTDSimulation(Nx=140, Ny=8, dx=0.001, dy=0.001, courant_factor=0.50)
    max_sf = 0.0
    max_tf = 0.0

    # Registrar el pico máximo temporal en la región dispersada y en la región total
    for _ in range(350):
        sim.step()
        sf = float(np.max(np.abs(sim.grid.Ez[:sim.i_tfsf - 2, :])))
        tf = float(np.max(np.abs(sim.grid.Ez[sim.i_tfsf + 2:, :])))
        if sf > max_sf:
            max_sf = sf
        if tf > max_tf:
            max_tf = tf

    ratio = max_sf / max_tf if max_tf > 0 else 1.0
    leakage_db = 20.0 * math.log10(max(ratio, 1e-15))

    return {
        "case": "V3_tfsf_leakage",
        "max_sf_field": max_sf,
        "max_tf_field": max_tf,
        "leakage_db": round(leakage_db, 2),
        "threshold_db": -60.0,
        "is_verified": leakage_db < -60.0,
    }


def validate_v4_phase_velocity_dispersion() -> dict:
    """V4: Velocidad de propagación FDTD comparada contra la dispersión numérica teórica de Yee."""
    dx = 0.001 # 1 mm
    courant = 0.70
    sim = FDTDSimulation(Nx=220, Ny=16, dx=dx, dy=dx, courant_factor=courant)
    f_test = 10.0e9 # 10 GHz
    v_theo = yee_numerical_phase_velocity(f_test, dx, sim.grid.dt)

    # Dos sondas separadas Delta_x
    x1, x2 = 60, 150
    dist_m = (x2 - x1) * dx

    # Inyección sinusoidal en x=i_tfsf para medir fase pura
    omega = 2.0 * math.pi * f_test
    h1, h2 = [], []
    for _ in range(700):
        sim.step()
        h1.append(float(sim.grid.Ez[x1, 8]))
        h2.append(float(sim.grid.Ez[x2, 8]))

    # Medir retraso temporal por correlación cruzada de las señales completas
    s1 = np.asarray(h1, dtype=np.float64)
    s2 = np.asarray(h2, dtype=np.float64)
    corr = np.correlate(s2, s1, mode="full")
    lags = np.arange(-len(s1) + 1, len(s2))
    best_lag = lags[np.argmax(corr)]
    delta_t = best_lag * sim.grid.dt
    v_measured = dist_m / delta_t if delta_t > 0 else 0.0

    err_vs_dispersion = abs(v_measured - v_theo) / v_theo * 100.0
    err_vs_c = abs(v_measured - SPEED_OF_LIGHT) / SPEED_OF_LIGHT * 100.0

    return {
        "case": "V4_phase_velocity_dispersion",
        "v_measured_m_s": round(v_measured, 1),
        "v_theo_dispersion_m_s": round(v_theo, 1),
        "c_constant_m_s": SPEED_OF_LIGHT,
        "error_vs_dispersion_percent": round(err_vs_dispersion, 3),
        "error_vs_c_percent": round(err_vs_c, 3),
        "is_verified": err_vs_dispersion < 1.0,
    }


def validate_v5_dielectric_slab_airy() -> dict:
    """V5: Lámina dieléctrica homogénea (eps_r = 4.0, L = 10 mm) comparada contra fórmula analítica de Airy."""
    dx = 0.0005 # 0.5 mm
    sim = FDTDSimulation(Nx=220, Ny=16, dx=dx, dy=dx, courant_factor=0.70)
    sim.calibrate_incident_wave(max_steps=1600)

    # Insertar lámina de 10 mm (20 celdas) de eps_r = 4.0
    L_m = 0.010
    start_cell = 60
    end_cell = start_cell + int(round(L_m / dx))
    sim.materials.set_dielectric_block(start_cell, end_cell, 0, sim.Ny, eps_relative=4.0)
    sim.reset()

    sim.run_until_decay(max_steps=2200)
    spec = sim.get_spectrum()

    # Solución analítica exacta de Airy evaluada en las mismas frecuencias
    airy_t = airy_dielectric_slab_transmission(4.0, L_m, sim.freqs_hz)
    fdtd_t = np.array(spec.transmission, dtype=np.float64)

    # Filtrar rango central 7 a 13 GHz
    mask = (sim.freqs_hz >= 7.0e9) & (sim.freqs_hz <= 13.0e9)
    rms_error = float(np.sqrt(np.mean((fdtd_t[mask] - airy_t[mask]) ** 2)))
    max_error = float(np.max(np.abs(fdtd_t[mask] - airy_t[mask])))

    return {
        "case": "V5_dielectric_slab_airy",
        "rms_error": round(rms_error, 4),
        "max_error": round(max_error, 4),
        "tolerance_rms": 0.025,
        "is_verified": rms_error < 0.025,
    }


def validate_v6_bragg_multicell_tmm() -> dict:
    """V6: Espejo multicapa de Bragg 1D periódico comparado contra la Matriz de Transferencia TMM."""
    dx = 0.0005
    sim = FDTDSimulation(Nx=260, Ny=16, dx=dx, dy=dx, courant_factor=0.70)
    sim.calibrate_incident_wave(max_steps=1800)

    # Apilamiento de 4 pares: capa eps_r=4.0 (d=4mm) y vacío eps_r=1.0 (d=4mm)
    pair_layers = 4
    d_m = 0.004
    cur = 60
    eps_list = []
    d_list = []
    for _ in range(pair_layers):
        c1 = cur + int(round(d_m / dx))
        sim.materials.set_dielectric_block(cur, c1, 0, sim.Ny, eps_relative=4.0)
        eps_list.append(4.0)
        d_list.append(d_m)

        c2 = c1 + int(round(d_m / dx))
        sim.materials.set_dielectric_block(c1, c2, 0, sim.Ny, eps_relative=1.0)
        eps_list.append(1.0)
        d_list.append(d_m)
        cur = c2

    sim.reset()
    sim.run_until_decay(max_steps=2400)
    spec = sim.get_spectrum()

    tmm_t = tmm_multilayer_transmission(eps_list, d_list, sim.freqs_hz)
    fdtd_t = np.array(spec.transmission, dtype=np.float64)

    mask = (sim.freqs_hz >= 7.0e9) & (sim.freqs_hz <= 13.0e9)
    rms_error = float(np.sqrt(np.mean((fdtd_t[mask] - tmm_t[mask]) ** 2)))

    return {
        "case": "V6_bragg_multicell_tmm",
        "rms_error": round(rms_error, 4),
        "tolerance_rms": 0.04,
        "is_verified": rms_error < 0.04,
    }


def validate_v7_energy_conservation_lossless() -> dict:
    """V7: Conservación rigurosa de energía R(f) + T(f) == 1.000 con R y T medidos independientemente."""
    sim = FDTDSimulation(Nx=220, Ny=20, dx=0.0005, dy=0.0005)
    sim.calibrate_incident_wave(max_steps=1600)

    # Insertar estructura periódica sin pérdidas (Familia B: Varillas de Alúmina eps=9.8)
    geom = UnitCellGeometry(a_mm=10.0, d_mm=3.0, w_mm=2.0, periods=3, family=CellFamily.ROD_ARRAY, relative_permittivity=9.8)
    rasterize_unit_cell(geom, sim.materials, x_start_idx=60)
    sim.reset()

    sim.run_until_decay(max_steps=2400)
    spec = sim.get_spectrum()

    mask = [7.0 <= f <= 13.0 for f in spec.frequencies_ghz]
    total_energy_dev = [abs(t + r - 1.0) for t, r, m in zip(spec.transmission, spec.reflection, mask) if m]
    max_dev = max(total_energy_dev)
    mean_dev = float(np.mean(total_energy_dev))

    return {
        "case": "V7_energy_conservation_lossless",
        "mean_energy_deviation": round(mean_dev, 4),
        "max_energy_deviation": round(max_dev, 4),
        "tolerance": 0.035, # Medición FDTD numérica con CPML y dispersión
        "is_verified": max_dev < 0.035,
    }


def validate_v8_passivity_lossy() -> dict:
    """V8: Pasividad en medio disipativo (sigma > 0) -> A(f) = 1 - T - R >= -0.01 y A > 0 en la banda."""
    sim = FDTDSimulation(Nx=200, Ny=16, dx=0.001, dy=0.001)
    sim.calibrate_incident_wave(max_steps=1200)

    # Lámina con conductividad sigma = 0.5 S/m
    sim.materials.set_dielectric_block(50, 70, 0, sim.Ny, eps_relative=4.0, conductivity=0.5)
    sim.reset()

    sim.run_until_decay(max_steps=1600)
    spec = sim.get_spectrum()

    mask = [7.0 <= f <= 13.0 for f in spec.frequencies_ghz]
    abs_band = [a for a, m in zip(spec.absorption, mask) if m]

    min_abs = min(abs_band)
    mean_abs = float(np.mean(abs_band))

    return {
        "case": "V8_passivity_lossy",
        "min_absorption": round(min_abs, 4),
        "mean_absorption": round(mean_abs, 4),
        "is_verified": min_abs >= -0.01 and mean_abs > 0.05,
    }


def validate_v9_mesh_convergence_order2() -> dict:
    """V9: Convergencia espacial de orden 2 de Yee al refinar la malla Delta_x, Delta_x/2."""
    # Comparar solución de lámina dieléctrica con dx1=1.0mm y dx2=0.5mm frente a Airy exacto
    errs = []
    for dx_val in (0.001, 0.0005):
        sim = FDTDSimulation(Nx=int(0.12 / dx_val), Ny=16, dx=dx_val, dy=dx_val, courant_factor=0.70)
        sim.calibrate_incident_wave(max_steps=1400)
        start = int(0.03 / dx_val)
        end = start + int(0.01 / dx_val) # L = 10 mm
        sim.materials.set_dielectric_block(start, end, 0, sim.Ny, eps_relative=4.0)
        sim.reset()
        sim.run_until_decay(max_steps=2000)
        spec = sim.get_spectrum()
        airy_t = airy_dielectric_slab_transmission(4.0, 0.01, sim.freqs_hz)
        mask = (sim.freqs_hz >= 8.0e9) & (sim.freqs_hz <= 12.0e9)
        rms = float(np.sqrt(np.mean((np.array(spec.transmission)[mask] - airy_t[mask]) ** 2)))
        errs.append(rms)

    ratio = errs[0] / max(errs[1], 1e-12)
    estimated_order = math.log2(ratio)

    return {
        "case": "V9_mesh_convergence_order2",
        "error_coarse": round(errs[0], 4),
        "error_fine": round(errs[1], 4),
        "error_reduction_ratio": round(ratio, 2),
        "estimated_order": round(estimated_order, 2),
        "is_verified": estimated_order > 1.4, # Segundo orden teórico (~2.0)
    }


def validate_v10_cfl_stability_theorem() -> dict:
    """V10: Teorema CFL en 2D: Estable con S=0.70 (<1/√2 ≈ 0.7071) e inestable con S=1.05 (>1/√2)."""
    # 1. Caso estable S = 0.70
    sim_s = FDTDSimulation(Nx=80, Ny=10, dx=0.002, dy=0.002, courant_factor=0.70)
    sim_s.run_steps(120)
    max_s = float(np.max(np.abs(sim_s.grid.Ez)))
    stable_ok = not (math.isnan(max_s) or math.isinf(max_s) or max_s > 1e4)

    # 2. Caso inestable S = 1.05 (> 0.7071)
    sim_u = FDTDSimulation(Nx=80, Ny=10, dx=0.002, dy=0.002, courant_factor=1.05)
    unstable_exploded = False
    for _ in range(120):
        sim_u.step()
        val = float(np.max(np.abs(sim_u.grid.Ez)))
        if math.isnan(val) or math.isinf(val) or val > 1e4:
            unstable_exploded = True
            break

    return {
        "case": "V10_cfl_stability_theorem",
        "cfl_limit_2d": round(1.0 / math.sqrt(2.0), 4),
        "s_stable": 0.70,
        "s_unstable": 1.05,
        "stable_remains_bounded": stable_ok,
        "unstable_explodes": unstable_exploded,
        "is_verified": stable_ok and unstable_exploded,
    }


def validate_v11_pec_wire_medium() -> dict:
    """V11: Medio de varillas conductoras PEC -> Corte espectral de transmisión cerca de la frecuencia de plasma."""
    dx = 0.0005 # 0.5 mm
    sim = FDTDSimulation(Nx=220, Ny=20, dx=dx, dy=dx, courant_factor=0.70) # a = 20 * 0.5 = 10 mm
    sim.calibrate_incident_wave(max_steps=1500)

    a_m = 0.010
    w_m = 0.001 # 1 mm (2 celdas)
    r_eff = 0.59 * (w_m / 2.0) # Radio equivalente de sección transversal cuadrada
    fp_theo_ghz = wire_medium_plasma_frequency(a_m, r_eff) * 1e-9

    geom = UnitCellGeometry(a_mm=10.0, d_mm=3.0, w_mm=1.0, periods=3, family=CellFamily.ROD_ARRAY, is_pec=True)
    rasterize_unit_cell(geom, sim.materials, x_start_idx=60)
    sim.reset()

    sim.run_until_decay(max_steps=2200)
    spec = sim.get_spectrum()

    # Transmisión muy baja por debajo de fp_theo (comportamiento plasma eps_eff < 0)
    sub_plasma_t = [t for t, f in zip(spec.transmission, spec.frequencies_ghz) if f < (fp_theo_ghz - 1.0)]
    mean_sub_plasma_t = float(np.mean(sub_plasma_t)) if sub_plasma_t else 0.0

    return {
        "case": "V11_pec_wire_medium",
        "plasma_freq_theo_ghz": round(fp_theo_ghz, 2),
        "mean_transmission_below_plasma": round(mean_sub_plasma_t, 3),
        "is_verified": mean_sub_plasma_t < 0.15,
    }


VALIDATION_REGISTRY = {
    "V1": validate_v1_vacuum_transmission,
    "V2": validate_v2_cpml_absorption,
    "V3": validate_v3_tfsf_leakage,
    "V4": validate_v4_phase_velocity_dispersion,
    "V5": validate_v5_dielectric_slab_airy,
    "V6": validate_v6_bragg_multicell_tmm,
    "V7": validate_v7_energy_conservation_lossless,
    "V8": validate_v8_passivity_lossy,
    "V9": validate_v9_mesh_convergence_order2,
    "V10": validate_v10_cfl_stability_theorem,
    "V11": validate_v11_pec_wire_medium,
}


def _sanitize_for_json(val: Any) -> Any:
    """Convierte recursivamente tipos escalares de NumPy a tipos nativos de Python para JSON."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, dict):
        return {k: _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_for_json(v) for v in val]
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    return val


def run_validation_case(case_id: str) -> dict:
    func = VALIDATION_REGISTRY.get(case_id.upper())
    if not func:
        raise ValueError(f"Caso de validación desconocido: {case_id}")
    raw = func()
    return _sanitize_for_json(raw)


def run_all_validations() -> dict[str, dict]:
    return {cid: run_validation_case(cid) for cid in VALIDATION_REGISTRY}

