"""Pruebas unitarias completas para el Motor FDTD 2D (Capítulo 10)."""

import math
import numpy as np
import pytest

from domain.fdtd import (
    YeeGrid2D,
    MaterialMap,
    FDTDSimulation,
    UnitCellGeometry,
    CellFamily,
    rasterize_unit_cell,
    airy_dielectric_slab_transmission,
    tmm_multilayer_transmission,
    yee_numerical_phase_velocity,
    wire_medium_plasma_frequency,
    run_validation_case,
)


def test_yee_grid_cfl_dt():
    """Verifica que el paso de tiempo dt cumpla estrictamente el teorema CFL 2D."""
    grid = YeeGrid2D(Nx=80, Ny=20, dx=0.001, dy=0.001, courant_factor=0.70)
    # Factor de Courant canonico S = c * dt / dx => dt = S * dx / c
    expected_dt = 0.70 * (0.001 / 299792458.0)
    assert abs(grid.dt - expected_dt) / expected_dt < 1e-6
    assert grid.Ez.shape == (80, 20)
    assert grid.Hx.shape == (80, 20)
    assert grid.Hy.shape == (80, 20)


def test_material_map_coefficients():
    """Verifica que los coeficientes Ca, Cb, Db se computen sin pérdida de precisión."""
    m = MaterialMap(Nx=40, Ny=10, dx=0.001, dy=0.001, dt=1e-12)
    # Vacío: sigma=0 => Ca=1.0, Cb = dt / (eps0 * dx)
    assert np.allclose(m.Ca, 1.0)
    assert m.Dbx > 0.0 and m.Dby > 0.0

    # Inserción de bloque con pérdidas
    m.set_dielectric_block(10, 20, 0, 10, eps_relative=4.0, conductivity=1.0)
    assert m.eps_r[15, 5] == 4.0
    assert m.sigma[15, 5] == 1.0
    assert m.Ca[15, 5] < 1.0 # Pérdidas causan Ca < 1.0


def test_geometry_rasterization_families():
    """Verifica la rasterización paramétrica de las Familias B (varillas) y C (anillos)."""
    m = MaterialMap(Nx=100, Ny=20, dx=0.0005, dy=0.0005, dt=1e-12)

    # Familia B: Varillas cuadradas
    geom_b = UnitCellGeometry(a_mm=10.0, d_mm=2.0, w_mm=2.0, periods=3, family=CellFamily.ROD_ARRAY, relative_permittivity=9.8)
    x0, x1 = rasterize_unit_cell(geom_b, m, x_start_idx=20)
    assert x1 > x0
    assert np.any(m.eps_r == 9.8)

    # Familia C: Anillo cuadrado concéntrico PEC
    m.reset_to_vacuum()
    geom_c = UnitCellGeometry(a_mm=10.0, d_mm=2.0, w_mm=1.0, periods=2, family=CellFamily.SQUARE_RING, is_pec=True)
    rasterize_unit_cell(geom_c, m, x_start_idx=20)
    assert np.any(m.pec_mask == True)
    # El centro del anillo debe permanecer vacío
    mid_y = m.Ny // 2
    assert m.pec_mask[20 + 4, mid_y] == False # hueco interior


def test_analytic_airy_peaks():
    """Verifica que la solución analítica de Airy prediga picos de transmisión unitaria T=1 en resonancia."""
    freqs = np.linspace(5e9, 15e9, 200)
    # Lamina L = 15 mm, eps = 4.0 (n = 2.0). Picos en f_m = m * c / (2 * n * L) = m * 3e8 / (2 * 2 * 0.015) = m * 5.0 GHz
    # f = 5 GHz, 10 GHz, 15 GHz son picos T = 1.0
    t_airy = airy_dielectric_slab_transmission(4.0, 0.015, np.array([5e9, 10e9, 15e9]))
    assert np.allclose(t_airy, 1.0, atol=1e-4)


def test_validation_v1_vacuum():
    res = run_validation_case("V1")
    assert res["is_verified"], f"V1 fallo con error: {res}"


def test_validation_v2_cpml():
    res = run_validation_case("V2")
    assert res["is_verified"], f"V2 fallo con reflexion: {res}"


def test_validation_v3_tfsf():
    res = run_validation_case("V3")
    assert res["is_verified"], f"V3 fallo con fuga: {res}"


def test_validation_v4_phase_velocity():
    res = run_validation_case("V4")
    assert res["is_verified"], f"V4 fallo: {res}"


def test_validation_v5_airy_slab():
    res = run_validation_case("V5")
    assert res["is_verified"], f"V5 fallo: {res}"


def test_validation_v6_bragg_tmm():
    res = run_validation_case("V6")
    assert res["is_verified"], f"V6 fallo: {res}"


def test_validation_v7_energy_conservation():
    res = run_validation_case("V7")
    assert res["is_verified"], f"V7 fallo: {res}"


def test_validation_v8_passivity():
    res = run_validation_case("V8")
    assert res["is_verified"], f"V8 fallo: {res}"


def test_validation_v9_convergence():
    res = run_validation_case("V9")
    assert res["is_verified"], f"V9 fallo: {res}"


def test_validation_v10_cfl():
    res = run_validation_case("V10")
    assert res["is_verified"], f"V10 fallo: {res}"


def test_validation_v11_wire_medium():
    res = run_validation_case("V11")
    assert res["is_verified"], f"V11 fallo: {res}"

