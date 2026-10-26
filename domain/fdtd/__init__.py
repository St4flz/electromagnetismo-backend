"""Motor FDTD 2D de Alta Fidelidad para Metamateriales Electromagnéticos.

Resuelve las ecuaciones acopladas de Maxwell en diferencias finitas (modo TM)
utilizando la celda escalonada de Yee (1966) con condiciones periódicas transversales (Floquet),
capas absorbentes CPML (Convolutional Perfectly Matched Layer) y fuentes TF/SF (Total-Field/Scattered-Field).
"""

from .grid import YeeGrid2D
from .materials import MaterialMap, MaterialPreset
from .boundaries import BoundaryController, CPMLBoundary
from .sources import TFSFSource, ModulatedGaussianPulse
from .monitors import FrequencyDomainMonitor, PoyntingSpectrum
from .analytic import (
    airy_dielectric_slab_transmission,
    tmm_multilayer_transmission,
    yee_numerical_phase_velocity,
    wire_medium_plasma_frequency,
)
from .geometry import UnitCellGeometry, CellFamily, rasterize_unit_cell
from .simulator import FDTDSimulation
from .validation import run_all_validations, run_validation_case, VALIDATION_REGISTRY

__all__ = [
    "YeeGrid2D",
    "MaterialMap",
    "MaterialPreset",
    "BoundaryController",
    "CPMLBoundary",
    "TFSFSource",
    "ModulatedGaussianPulse",
    "FrequencyDomainMonitor",
    "PoyntingSpectrum",
    "airy_dielectric_slab_transmission",
    "tmm_multilayer_transmission",
    "yee_numerical_phase_velocity",
    "wire_medium_plasma_frequency",
    "UnitCellGeometry",
    "CellFamily",
    "rasterize_unit_cell",
    "FDTDSimulation",
    "run_all_validations",
    "run_validation_case",
    "VALIDATION_REGISTRY",
]
