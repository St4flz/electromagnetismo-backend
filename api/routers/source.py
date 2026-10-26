"""Endpoint de trazabilidad Ecuación → Código.

Devuelve el código fuente REAL (vía ``inspect``) de funciones del dominio físico, para que
la interfaz muestre junto a cada ecuación el mismo código que el servidor está ejecutando.

Seguridad: solo se sirven los símbolos listados explícitamente en ``ALLOWED_SYMBOLS``
(lista blanca). Cualquier otro nombre responde 404; no se importa nada dinámicamente a
partir de texto arbitrario del cliente.
"""

from __future__ import annotations

import importlib
import inspect

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/source", tags=["Source"])

# símbolo público → (módulo, ruta de atributos). Añadir aquí los símbolos de nuevos módulos.
ALLOWED_SYMBOLS: dict[str, tuple[str, str]] = {
    # Capítulo 8 — metamateriales (modelo de medio efectivo)
    "metamaterials.compute_effective_parameters": ("domain.metamaterials", "compute_effective_parameters"),
    "metamaterials.compute_spectral_response": ("domain.metamaterials", "compute_spectral_response"),
    "metamaterials.verify_energy_conservation_spectrum": ("domain.metamaterials", "verify_energy_conservation_spectrum"),
    "metamaterials.verify_bragg_bandgap": ("domain.metamaterials", "verify_bragg_bandgap"),
    # Capítulo 9 — FDTD introductorio
    "fdtd2d.FDTD2DSimulator._recompute_coefficients": ("domain.fdtd2d", "FDTD2DSimulator._recompute_coefficients"),
    "fdtd2d.FDTD2DSimulator._source_value": ("domain.fdtd2d", "FDTD2DSimulator._source_value"),
    "fdtd2d.FDTD2DSimulator.step": ("domain.fdtd2d", "FDTD2DSimulator.step"),
    "fdtd2d.FDTD2DSimulator.compute_transmission_spectrum": ("domain.fdtd2d", "FDTD2DSimulator.compute_transmission_spectrum"),
    "fdtd2d.verify_cfl_stability_condition": ("domain.fdtd2d", "verify_cfl_stability_condition"),
    "fdtd2d.verify_fdtd_pulse_velocity": ("domain.fdtd2d", "verify_fdtd_pulse_velocity"),
    # Capítulo 10 — Motor FDTD v2 (Rigor físico y validaciones V1 a V11)
    "fdtd.FDTDSimulation.step": ("domain.fdtd.simulator", "FDTDSimulation.step"),
    "fdtd.FDTDSimulation.calibrate_incident_wave": ("domain.fdtd.simulator", "FDTDSimulation.calibrate_incident_wave"),
    "fdtd.CPMLBoundary.__post_init__": ("domain.fdtd.boundaries", "CPMLBoundary.__post_init__"),
    "fdtd.TFSFSource.correct_2d_H": ("domain.fdtd.sources", "TFSFSource.correct_2d_H"),
    "fdtd.TFSFSource.correct_2d_E": ("domain.fdtd.sources", "TFSFSource.correct_2d_E"),
    "fdtd.PoyntingSpectrum.from_powers": ("domain.fdtd.monitors", "PoyntingSpectrum.from_powers"),
    "fdtd.airy_dielectric_slab_transmission": ("domain.fdtd.analytic", "airy_dielectric_slab_transmission"),
    "fdtd.tmm_multilayer_transmission": ("domain.fdtd.analytic", "tmm_multilayer_transmission"),
    "fdtd.yee_numerical_phase_velocity": ("domain.fdtd.analytic", "yee_numerical_phase_velocity"),
    "fdtd.wire_medium_plasma_frequency": ("domain.fdtd.analytic", "wire_medium_plasma_frequency"),
    "fdtd.rasterize_unit_cell": ("domain.fdtd.geometry", "rasterize_unit_cell"),
}


def _resolve(module_name: str, attr_path: str):
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


@router.get("")
def list_symbols() -> dict:
    """Lista los símbolos cuyo código fuente se puede consultar."""
    return {"symbols": sorted(ALLOWED_SYMBOLS)}


@router.get("/{symbol}")
def get_source(symbol: str) -> dict:
    """Devuelve el código fuente de un símbolo de la lista blanca."""
    target = ALLOWED_SYMBOLS.get(symbol)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Símbolo no disponible: {symbol}")

    obj = _resolve(*target)
    lines, start_line = inspect.getsourcelines(obj)
    return {
        "symbol": symbol,
        "module": target[0],
        "file": inspect.getsourcefile(obj),
        "startLine": start_line,
        "endLine": start_line + len(lines) - 1,
        "source": "".join(lines),
    }
