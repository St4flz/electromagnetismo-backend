"""Dominio físico del simulador electromagnético.

Este paquete contiene la física pura: vectores, partículas, leyes y campos.
No hay HTTP, no hay API, no hay frameworks — solo matemáticas y física.

Estructura:
    vectors    — Vector2D, Vector3D (la abstracción más fundamental)
    units      — Constantes del SI y conversiones
    particles  — Carga puntual: posición + carga eléctrica
    coulomb    — Ley de Coulomb: F = k·q₁·q₂/r²·r̂
    electric_field — Campo eléctrico: E = k·q/r²·r̂
"""
