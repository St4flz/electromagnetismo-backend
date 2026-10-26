"""Motor de barridos paramétricos FDTD y generación de datasets estructurados (Capítulo 11)."""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Sequence
import numpy as np

from domain.fdtd.simulator import FDTDSimulation
from domain.fdtd.geometry import UnitCellGeometry, CellFamily, rasterize_unit_cell
from domain.objective import (
    evaluate_f1_pure_transmission,
    evaluate_f2_bandstop_filter,
    compute_spearman_rank_correlation,
    ObjectiveEvaluation,
)


@dataclass
class SweepDesignPoint:
    design_id: int
    a_mm: float
    d_mm: float
    w_mm: float
    periods: int
    family: str
    is_pec: bool
    relative_permittivity: float
    f0_ghz: float
    t_f0: float
    t_f0_db: float
    r_f0: float
    t_pass_mean: float
    rejection_depth_db: float
    f2_score: float
    rank: int = 0


@dataclass
class Sweep1DResult:
    parameter_name: str
    parameter_values: list[float]
    frequencies_ghz: list[float]
    family_curves_t: dict[str, list[float]]     # key: "w=1.0mm" -> [T(f)]
    family_curves_r: dict[str, list[float]]     # key: "w=1.0mm" -> [R(f)]
    table_rows: list[SweepDesignPoint]
    best_design: SweepDesignPoint
    best_by_f1: SweepDesignPoint
    metadata: dict[str, Any]


@dataclass
class Sweep2DResult:
    param_x_name: str
    param_x_values: list[float]
    param_y_name: str
    param_y_values: list[float]
    frequencies_ghz: list[float]
    heatmap_t_f0: list[list[float]]             # [len(y), len(x)] T(f0)
    heatmap_t_f0_db: list[list[float]]          # [len(y), len(x)] T(f0) en dB
    heatmap_f2: list[list[float]]               # [len(y), len(x)] F2
    table_rows: list[SweepDesignPoint]
    best_design: SweepDesignPoint
    metadata: dict[str, Any]


def simulate_single_design(
    a_mm: float,
    d_mm: float,
    w_mm: float,
    periods: int = 3,
    family_str: str = "square_ring",
    relative_permittivity: float = 1.0,
    is_pec: bool = True,
    dx_mm: float = 0.5,
    courant_factor: float = 0.70,
    pml_thickness: int = 14,
    max_steps: int = 1800,
    f0_ghz: float = 10.0,
    f_min_ghz: float = 5.0,
    f_max_ghz: float = 15.0,
    design_id: int = 1,
) -> tuple[SweepDesignPoint, list[float], list[float], list[float]]:
    """Ejecuta una corrida FDTD completa para un diseño paramétrico dado."""
    dx = dx_mm * 1e-3
    dy = dx
    a_m = a_mm * 1e-3
    Ny = max(int(round(a_m / dy)), 4)

    cell_len_m = a_m * periods
    cell_cells_x = int(round(cell_len_m / dx))
    margin_x = int(round(0.030 / dx))
    Nx = pml_thickness * 2 + 25 + margin_x + cell_cells_x

    sim = FDTDSimulation(
        Nx=Nx,
        Ny=Ny,
        dx=dx,
        dy=dy,
        courant_factor=courant_factor,
        pml_thickness=pml_thickness,
        f_min_ghz=f_min_ghz,
        f_max_ghz=f_max_ghz,
    )

    # 1. Calibrar incidente en vacío
    sim.calibrate_incident_wave(max_steps=min(max_steps, 1400))

    # 2. Configurar estructura
    fam = CellFamily.ROD_ARRAY if family_str == "rod_array" else CellFamily.SQUARE_RING
    geom = UnitCellGeometry(
        a_mm=a_mm,
        d_mm=d_mm,
        w_mm=w_mm,
        periods=periods,
        family=fam,
        relative_permittivity=relative_permittivity,
        is_pec=is_pec,
    )

    # Verificar viabilidad geométrica de la celda
    try:
        geom.validate()
        is_valid = True
        penalty = 0.0
    except ValueError:
        is_valid = False
        penalty = 1.0

    if is_valid:
        x_start = sim.i_tfsf + 12
        rasterize_unit_cell(geom, sim.materials, x_start_idx=x_start)
    sim.reset()

    # 3. Correr simulación
    sim.run_until_decay(max_steps=max_steps, energy_threshold=1e-5)
    spec = sim.get_spectrum()

    # Evaluar objetivos F1 y F2
    eval_f2 = evaluate_f2_bandstop_filter(
        spectrum=spec,
        f0_ghz=f0_ghz,
        delta_f_reject_ghz=0.5,
        penalty_value=penalty,
        is_valid_geometry=is_valid,
    )

    idx_f0 = int(np.argmin(np.abs(np.array(spec.frequencies_ghz) - f0_ghz)))
    r_f0 = float(spec.reflection[idx_f0]) if idx_f0 < len(spec.reflection) else 0.0

    point = SweepDesignPoint(
        design_id=design_id,
        a_mm=round(a_mm, 2),
        d_mm=round(d_mm, 2),
        w_mm=round(w_mm, 2),
        periods=periods,
        family=family_str,
        is_pec=is_pec,
        relative_permittivity=round(relative_permittivity, 2),
        f0_ghz=round(f0_ghz, 2),
        t_f0=eval_f2.t_f0,
        t_f0_db=eval_f2.t_f0_db,
        r_f0=round(r_f0, 5),
        t_pass_mean=eval_f2.t_pass_mean,
        rejection_depth_db=eval_f2.rejection_depth_db,
        f2_score=eval_f2.objective_value,
    )

    return point, spec.frequencies_ghz, spec.transmission, spec.reflection


def run_1d_sweep(
    parameter_name: str = "w_mm",
    param_values: list[float] | None = None,
    base_a_mm: float = 10.0,
    base_d_mm: float = 6.0,
    base_w_mm: float = 1.0,
    family_str: str = "square_ring",
    periods: int = 3,
    is_pec: bool = True,
    relative_permittivity: float = 1.0,
    dx_mm: float = 0.5,
    f0_ghz: float = 10.0,
    max_steps: int = 1600,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Sweep1DResult:
    """Ejecuta un barrido unidimensional sobre un parámetro físico (e.g. w_mm o d_mm)."""
    t0 = time.perf_counter()
    if param_values is None:
        if parameter_name == "w_mm":
            param_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        elif parameter_name == "d_mm":
            param_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        else:
            param_values = [8.0, 9.0, 10.0, 11.0, 12.0]

    family_curves_t: dict[str, list[float]] = {}
    family_curves_r: dict[str, list[float]] = {}
    table_rows: list[SweepDesignPoint] = []
    freqs_ret: list[float] = []

    total_pts = len(param_values)
    for idx, val in enumerate(param_values):
        a_curr = val if parameter_name == "a_mm" else base_a_mm
        d_curr = val if parameter_name == "d_mm" else base_d_mm
        w_curr = val if parameter_name == "w_mm" else base_w_mm

        if progress_callback:
            progress_callback(idx + 1, total_pts, f"{parameter_name} = {val:.2f} mm")

        pt, freqs, trans, refl = simulate_single_design(
            a_mm=a_curr,
            d_mm=d_curr,
            w_mm=w_curr,
            periods=periods,
            family_str=family_str,
            is_pec=is_pec,
            relative_permittivity=relative_permittivity,
            dx_mm=dx_mm,
            f0_ghz=f0_ghz,
            max_steps=max_steps,
            design_id=idx + 1,
        )

        key = f"{parameter_name}={val:.2f}mm"
        family_curves_t[key] = trans
        family_curves_r[key] = refl
        table_rows.append(pt)
        if not freqs_ret:
            freqs_ret = freqs

    # Asignar ranking según F2 (menor valor es mejor)
    sorted_indices = np.argsort([pt.f2_score for pt in table_rows])
    for rank_idx, s_idx in enumerate(sorted_indices):
        table_rows[s_idx].rank = rank_idx + 1

    best_f2 = table_rows[sorted_indices[0]]

    # Mejor según F1 (pura transmisión en f0)
    sorted_by_f1 = sorted(table_rows, key=lambda x: x.t_f0)
    best_f1 = sorted_by_f1[0]

    elapsed = round(time.perf_counter() - t0, 3)

    metadata = {
        "sweep_type": "1D",
        "parameter_name": parameter_name,
        "dx_mm": dx_mm,
        "points_evaluated": total_pts,
        "elapsed_seconds": elapsed,
        "f0_ghz": f0_ghz,
    }

    return Sweep1DResult(
        parameter_name=parameter_name,
        parameter_values=param_values,
        frequencies_ghz=freqs_ret,
        family_curves_t=family_curves_t,
        family_curves_r=family_curves_r,
        table_rows=table_rows,
        best_design=best_f2,
        best_by_f1=best_f1,
        metadata=metadata,
    )


def run_2d_sweep(
    param_x_name: str = "a_mm",
    param_x_values: list[float] | None = None,
    param_y_name: str = "d_mm",
    param_y_values: list[float] | None = None,
    base_w_mm: float = 1.0,
    family_str: str = "square_ring",
    periods: int = 3,
    is_pec: bool = True,
    relative_permittivity: float = 1.0,
    dx_mm: float = 0.5,
    f0_ghz: float = 10.0,
    max_steps: int = 1400,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Sweep2DResult:
    """Ejecuta un barrido bidimensional sobre una rejilla de parámetros para mapas de calor."""
    t0 = time.perf_counter()
    if param_x_values is None:
        param_x_values = [8.0, 9.0, 10.0, 11.0, 12.0]
    if param_y_values is None:
        param_y_values = [3.0, 4.0, 5.0, 6.0, 7.0]

    nx = len(param_x_values)
    ny = len(param_y_values)
    total_pts = nx * ny

    heatmap_t = np.zeros((ny, nx), dtype=np.float64)
    heatmap_t_db = np.zeros((ny, nx), dtype=np.float64)
    heatmap_f2 = np.zeros((ny, nx), dtype=np.float64)

    table_rows: list[SweepDesignPoint] = []
    freqs_ret: list[float] = []

    count = 0
    for j, y_val in enumerate(param_y_values):
        for i, x_val in enumerate(param_x_values):
            count += 1
            a_curr = x_val if param_x_name == "a_mm" else (y_val if param_y_name == "a_mm" else 10.0)
            d_curr = x_val if param_x_name == "d_mm" else (y_val if param_y_name == "d_mm" else 6.0)
            w_curr = x_val if param_x_name == "w_mm" else (y_val if param_y_name == "w_mm" else base_w_mm)

            if progress_callback:
                progress_callback(count, total_pts, f"({x_val:.1f}, {y_val:.1f})")

            pt, freqs, trans, _ = simulate_single_design(
                a_mm=a_curr,
                d_mm=d_curr,
                w_mm=w_curr,
                periods=periods,
                family_str=family_str,
                is_pec=is_pec,
                relative_permittivity=relative_permittivity,
                dx_mm=dx_mm,
                f0_ghz=f0_ghz,
                max_steps=max_steps,
                design_id=count,
            )

            heatmap_t[j, i] = pt.t_f0
            heatmap_t_db[j, i] = pt.t_f0_db
            heatmap_f2[j, i] = pt.f2_score
            table_rows.append(pt)
            if not freqs_ret:
                freqs_ret = freqs

    sorted_indices = np.argsort([pt.f2_score for pt in table_rows])
    for rank_idx, s_idx in enumerate(sorted_indices):
        table_rows[s_idx].rank = rank_idx + 1

    best_design = table_rows[sorted_indices[0]]
    elapsed = round(time.perf_counter() - t0, 3)

    return Sweep2DResult(
        param_x_name=param_x_name,
        param_x_values=param_x_values,
        param_y_name=param_y_name,
        param_y_values=param_y_values,
        frequencies_ghz=freqs_ret,
        heatmap_t_f0=heatmap_t.tolist(),
        heatmap_t_f0_db=heatmap_t_db.tolist(),
        heatmap_f2=heatmap_f2.tolist(),
        table_rows=table_rows,
        best_design=best_design,
        metadata={
            "sweep_type": "2D",
            "dx_mm": dx_mm,
            "total_points": total_pts,
            "elapsed_seconds": elapsed,
        },
    )


def export_sweep_to_csv(table_rows: Sequence[SweepDesignPoint]) -> str:
    """Exporta los resultados a formato CSV conforme a la tabla del README."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Ranking",
        "Design_ID",
        "Familia",
        "Material",
        "a [mm]",
        "d [mm]",
        "w [mm]",
        "Periodos_x",
        "f0 [GHz]",
        "T(f0)",
        "T(f0) [dB]",
        "R(f0)",
        "T_pass_mean",
        "Rechazo [dB]",
        "Score_F2",
    ])
    for pt in sorted(table_rows, key=lambda x: x.rank):
        writer.writerow([
            pt.rank,
            pt.design_id,
            pt.family,
            "PEC" if pt.is_pec else f"Diel (eps={pt.relative_permittivity})",
            pt.a_mm,
            pt.d_mm,
            pt.w_mm,
            pt.periods,
            pt.f0_ghz,
            pt.t_f0,
            pt.t_f0_db,
            pt.r_f0,
            pt.t_pass_mean,
            pt.rejection_depth_db,
            pt.f2_score,
        ])
    return output.getvalue()


def evaluate_sweep_robustness(
    test_points: list[tuple[float, float, float]] | None = None,
    family_str: str = "rod_array",
) -> dict[str, Any]:
    """Prueba de robustez formal (tarea 333): evalúa F2 en Δx (0.5 mm) y Δx/2 (0.25 mm) y calcula Spearman."""
    if test_points is None:
        # 4 diseños representativos con variación monótona de tamaño
        test_points = [
            (10.0, 2.0, 2.0),
            (10.0, 4.0, 4.0),
            (10.0, 6.0, 6.0),
            (10.0, 8.0, 8.0),
        ]

    scores_coarse: list[float] = []
    scores_fine: list[float] = []

    for idx, (a, d, w) in enumerate(test_points):
        # Malla gruesa 0.5 mm
        pt_c, _, _, _ = simulate_single_design(
            a_mm=a, d_mm=d, w_mm=w, family_str=family_str, dx_mm=0.5, max_steps=1400, design_id=idx + 1
        )
        scores_coarse.append(pt_c.f2_score)

        # Malla fina 0.25 mm
        pt_f, _, _, _ = simulate_single_design(
            a_mm=a, d_mm=d, w_mm=w, family_str=family_str, dx_mm=0.25, max_steps=1800, design_id=idx + 1
        )
        scores_fine.append(pt_f.f2_score)

    spearman_rho = compute_spearman_rank_correlation(scores_coarse, scores_fine)

    return {
        "num_designs_tested": len(test_points),
        "scores_coarse_dx_0_5mm": [round(s, 5) for s in scores_coarse],
        "scores_fine_dx_0_25mm": [round(s, 5) for s in scores_fine],
        "spearman_rank_correlation": round(spearman_rho, 4),
        "threshold": 0.90,
        "is_robust": spearman_rho >= 0.90,
    }
