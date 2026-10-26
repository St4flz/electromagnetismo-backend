"""Modelado geométrico y rasterización de celdas unitarias de metamateriales."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from .materials import MaterialMap


class CellFamily(str, Enum):
    ROD_ARRAY = "rod_array"       # Familia B: Arreglo de varillas cuadradas (Cristal fotónico / Medios de hilos)
    SQUARE_RING = "square_ring"   # Familia C: Anillo cuadrado concéntrico (README Fase 4)


@dataclass(frozen=True)
class UnitCellGeometry:
    """Parámetros geométricos de la celda unitaria del metamaterial en milímetros [mm]."""

    a_mm: float                   # Tamaño transversal de la celda periódica [mm] (eje y)
    d_mm: float                   # Separación entre elementos [mm]
    w_mm: float                   # Ancho / espesor del elemento conductor o dieléctrico [mm]
    periods: int = 4              # Número de celdas en la dirección de propagación (eje x)
    family: CellFamily = CellFamily.ROD_ARRAY
    relative_permittivity: float = 9.8  # Permitividad del dieléctrico (ej. Alúmina 9.8, FR4 4.4, Titanato)
    is_pec: bool = False                # True si el elemento es metálico PEC (conductor perfecto)
    conductivity: float = 0.0           # Pérdidas óhmicas [S/m]

    def validate(self) -> None:
        """Verifica restricciones físicas y geométricas con auto-reparación no destructiva."""
        if self.a_mm <= 0:
            self.a_mm = 10.0
        if self.w_mm < 0:
            self.w_mm = 0.0
        # Permitir w == a para lámina continua homogénea (Slab Airy)
        if self.w_mm > self.a_mm:
            self.w_mm = self.a_mm
        if self.family == CellFamily.SQUARE_RING and self.w_mm > 0:
            # Garantizar hueco interior para el anillo concéntrico
            if (self.a_mm - self.d_mm) <= 2.0 * self.w_mm:
                self.d_mm = max(0.0, self.a_mm - 2.0 * self.w_mm - 0.5)
                if (self.a_mm - self.d_mm) <= 2.0 * self.w_mm:
                    self.w_mm = max(0.25, (self.a_mm - self.d_mm - 0.5) / 2.0)


def rasterize_unit_cell(
    geom: UnitCellGeometry,
    materials: MaterialMap,
    x_start_idx: int,
) -> tuple[int, int]:
    """Rasteriza la estructura sobre MaterialMap cuantizada con la resolución de la grilla.
    
    Retorna el rango espacial (x_start_idx, x_end_idx) ocupado por la estructura.
    """
    geom.validate()
    dx_mm = materials.dx * 1e3
    dy_mm = materials.dy * 1e3

    # Cuantización de dimensiones continuas a celdas enteras de grilla
    cell_nx = max(1, int(round(geom.a_mm / dx_mm)))
    cell_ny = materials.Ny # Debe coincidir con el periodo transversal completo Ny * dy = a

    if geom.w_mm <= 0:
        # Vacío: sin inclusiones materiales
        return x_start_idx, x_start_idx + cell_nx * geom.periods

    w_cells_x = max(1, int(round(geom.w_mm / dx_mm)))
    w_cells_y = max(1, int(round(geom.w_mm / dy_mm)))
    d_cells_x = max(0, int(round(geom.d_mm / dx_mm)))

    current_x = x_start_idx

    for p in range(geom.periods):
        if geom.family == CellFamily.ROD_ARRAY:
            # Familia B: Varilla cuadrada centrada en cada celda periódica
            x0 = current_x + d_cells_x
            x1 = x0 + w_cells_x
            # Centrado en el eje transversal y
            mid_y = cell_ny // 2
            y0 = max(0, mid_y - (w_cells_y // 2))
            y1 = min(cell_ny, y0 + w_cells_y)

            if geom.is_pec:
                materials.set_pec_block(x0, x1, y0, y1)
            else:
                materials.set_dielectric_block(
                    x0, x1, y0, y1,
                    eps_relative=geom.relative_permittivity,
                    conductivity=geom.conductivity,
                )
            current_x = x1 + d_cells_x

        elif geom.family == CellFamily.SQUARE_RING:
            # Familia C: Anillo cuadrado concéntrico
            # Marco exterior de tamaño (a - d) y grosor w
            ring_size_cells_x = max(2 * w_cells_x + 1, cell_nx - d_cells_x)
            ring_size_cells_y = max(2 * w_cells_y + 1, cell_ny - d_cells_x)

            x_out0 = current_x + (d_cells_x // 2)
            x_out1 = x_out0 + ring_size_cells_x

            mid_y = cell_ny // 2
            y_out0 = max(0, mid_y - (ring_size_cells_y // 2))
            y_out1 = min(cell_ny, y_out0 + ring_size_cells_y)

            # Hueco interior
            x_in0 = x_out0 + w_cells_x
            x_in1 = x_out1 - w_cells_x
            y_in0 = y_out0 + w_cells_y
            y_in1 = y_out1 - w_cells_y

            if geom.is_pec:
                # 1. Todo el bloque exterior PEC
                materials.set_pec_block(x_out0, x_out1, y_out0, y_out1)
                # 2. Vaciado interior (vuelve a ser vacío)
                if x_in0 < x_in1 and y_in0 < y_in1:
                    materials.set_dielectric_block(x_in0, x_in1, y_in0, y_in1, eps_relative=1.0)
            else:
                # Dieléctrico
                materials.set_dielectric_block(
                    x_out0, x_out1, y_out0, y_out1,
                    eps_relative=geom.relative_permittivity,
                    conductivity=geom.conductivity,
                )
                if x_in0 < x_in1 and y_in0 < y_in1:
                    materials.set_dielectric_block(x_in0, x_in1, y_in0, y_in1, eps_relative=1.0)

            current_x = x_out1 + (d_cells_x // 2)

    return x_start_idx, current_x
