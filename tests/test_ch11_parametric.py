"""Pruebas unitarias para el Capítulo 11: Laboratorio Paramétrico y Funciones Objetivo."""

import pytest
import numpy as np
from domain.fdtd.monitors import PoyntingSpectrum
from domain.objective import (
    evaluate_f1_pure_transmission,
    evaluate_f2_bandstop_filter,
    compute_spearman_rank_correlation,
)
from domain.sweep import (
    simulate_single_design,
    run_1d_sweep,
    export_sweep_to_csv,
    compute_spearman_rank_correlation,
    evaluate_sweep_robustness,
)


def test_f1_vs_f2_bandstop_physics():
    """Demuestra formalmente la trampa física: F1 favorece una pared sólida con T=0 en todas partes,

    mientras que F2 penaliza la falta de transparencia en la banda de paso.
    """
    freqs = np.linspace(5e9, 15e9, 100)

    # Caso A: Pared sólida PEC (T = 0.00 en todas partes)
    spec_wall = PoyntingSpectrum(
        frequencies_ghz=[float(f * 1e-9) for f in freqs],
        transmission=[0.0] * 100,
        reflection=[1.0] * 100,
        absorption=[0.0] * 100,
        transmission_db=[-50.0] * 100,
    )

    # Caso B: Filtro resonante selectivo (T(10 GHz) = 0.02, T(fuera) = 0.95)
    t_filter = []
    for f_ghz in [float(f * 1e-9) for f in freqs]:
        if 9.5 <= f_ghz <= 10.5:
            t_filter.append(0.02)
        else:
            t_filter.append(0.95)

    spec_filter = PoyntingSpectrum(
        frequencies_ghz=[float(f * 1e-9) for f in freqs],
        transmission=t_filter,
        reflection=[1.0 - t for t in t_filter],
        absorption=[0.0] * 100,
        transmission_db=[10 * np.log10(max(t, 1e-5)) for t in t_filter],
    )

    # Evaluación con F1 (transmisión pura en f0)
    eval_f1_wall = evaluate_f1_pure_transmission(spec_wall, f0_ghz=10.0)
    eval_f1_filter = evaluate_f1_pure_transmission(spec_filter, f0_ghz=10.0)
    # F1 erróneamente prefiere la pared (0.0 < 0.02)
    assert eval_f1_wall.objective_value < eval_f1_filter.objective_value

    # Evaluación con F2 (selectividad de banda)
    eval_f2_wall = evaluate_f2_bandstop_filter(spec_wall, f0_ghz=10.0)
    eval_f2_filter = evaluate_f2_bandstop_filter(spec_filter, f0_ghz=10.0)

    # F2 prefiere correctamente el filtro selectivo (F2_filter << F2_wall)
    assert eval_f2_filter.objective_value < eval_f2_wall.objective_value
    assert eval_f2_filter.rejection_depth_db > 10.0 # Más de 10 dB de rechazo relativo


def test_spearman_rank_correlation():
    """Verifica el cálculo analítico de correlación de rangos de Spearman."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_mono = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert abs(compute_spearman_rank_correlation(x, y_mono) - 1.0) < 1e-5

    y_inv = [50.0, 40.0, 30.0, 20.0, 10.0]
    assert abs(compute_spearman_rank_correlation(x, y_inv) - (-1.0)) < 1e-5


def test_single_design_simulation():
    """Ejecuta una corrida corta de un diseño individual y extrae métricas de diseño."""
    pt, freqs, trans, refl = simulate_single_design(
        a_mm=10.0,
        d_mm=6.0,
        w_mm=1.0,
        periods=2,
        family_str="square_ring",
        is_pec=True,
        dx_mm=1.0,
        max_steps=300,
        f0_ghz=10.0,
        design_id=42,
    )
    assert pt.design_id == 42
    assert pt.a_mm == 10.0
    assert pt.d_mm == 6.0
    assert pt.w_mm == 1.0
    assert 0.0 <= pt.t_f0 <= 1.0
    assert len(trans) == len(freqs)


def test_1d_sweep_and_csv_export():
    """Ejecuta un barrido 1D rápido sobre 3 valores de w y verifica la generación de CSV tipo README."""
    res = run_1d_sweep(
        parameter_name="w_mm",
        param_values=[0.5, 1.0, 1.5],
        base_a_mm=10.0,
        base_d_mm=5.0,
        family_str="square_ring",
        periods=1,
        dx_mm=1.0,
        max_steps=250,
    )
    assert len(res.table_rows) == 3
    assert len(res.family_curves_t) == 3
    assert res.best_design.rank == 1

    csv_text = export_sweep_to_csv(res.table_rows)
    assert "Ranking,Design_ID,Familia" in csv_text
    assert "w_mm" in res.metadata["parameter_name"]


def test_sweep_robustness():
    """Verifica que la prueba de robustez formal (tarea 333) ejecute y reporte Spearman >= 0.90."""
    rob = evaluate_sweep_robustness(family_str="rod_array")
    assert "spearman_rank_correlation" in rob
    assert rob["spearman_rank_correlation"] >= 0.90
    assert rob["is_robust"] is True
