"""Pruebas unitarias para el Capítulo 8: Introducción a Metamateriales."""

import pytest
from domain.metamaterials import (
    MetamaterialUnitCell,
    compute_effective_parameters,
    compute_spectral_response,
    evaluate_objective_function,
    verify_energy_conservation_spectrum,
    verify_veselago_negative_refraction,
    verify_bragg_bandgap,
)


def test_energy_conservation_spectrum():
    cell = MetamaterialUnitCell(a=10.0, d=2.0, w=1.0, periods=5)
    res = verify_energy_conservation_spectrum(cell)
    assert res["is_conserved"]
    assert res["max_energy_deviation"] < 1e-5


def test_veselago_negative_refraction():
    cell = MetamaterialUnitCell(a=12.0, d=2.5, w=1.2, periods=4)
    res = verify_veselago_negative_refraction(cell, incident_angle_deg=30.0)
    assert res["is_negative_refraction"]
    assert res["refractive_index_n2"] < 0.0
    assert res["refracted_angle_deg"] < 0.0


def test_bragg_bandgap():
    res = verify_bragg_bandgap(cell_size_mm=10.0, host_eps_r=2.2)
    assert res["is_verified"]
    assert res["theoretical_bragg_frequency_ghz"] > 0.0


def test_objective_function_evaluation():
    cell = MetamaterialUnitCell(a=10.0, d=2.0, w=1.0, periods=5)
    f_val = evaluate_objective_function(cell, target_frequency_ghz=10.0)
    assert 0.0 <= f_val <= 1.0
