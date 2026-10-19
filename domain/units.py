"""Constantes fundamentales del electromagnetismo (SI).

Cada constante incluye:
    - Su símbolo estándar
    - Su valor en unidades SI
    - Las unidades
    - En qué ecuación aparece y qué significa físicamente

Referencia: CODATA 2018 (NIST)
"""

import math

# ═══════════════════════════════════════════════════════════════════
# CONSTANTE DE COULOMB  k = 1/(4πε₀)
# ═══════════════════════════════════════════════════════════════════
#
# Aparece en la Ley de Coulomb:   F = k · q₁ · q₂ / r²
# y en el campo eléctrico:        E = k · q / r²
#
# Determina la intensidad de la interacción eléctrica en el vacío.
# Es una constante enorme (~9 × 10⁹) lo que significa que la fuerza
# eléctrica es extraordinariamente intensa comparada con la gravedad.
#
COULOMB_CONSTANT: float = 8.9875517873681764e9  # [N·m²/C²]

# ═══════════════════════════════════════════════════════════════════
# PERMITIVIDAD DEL VACÍO  ε₀
# ═══════════════════════════════════════════════════════════════════
#
# Mide la capacidad del vacío para "permitir" un campo eléctrico.
# Aparece en:  ∇·D = ρ   y   D = ε₀·E
#
# Un material con ε > ε₀ "apantalla" el campo eléctrico,
# reduciendo la fuerza entre cargas dentro de él.
#
EPSILON_0: float = 8.8541878128e-12  # [F/m] (Faradios por metro)

# ═══════════════════════════════════════════════════════════════════
# PERMEABILIDAD DEL VACÍO  μ₀
# ═══════════════════════════════════════════════════════════════════
#
# Mide la capacidad del vacío para soportar un campo magnético.
# Aparece en:  B = μ₀·H   y   ∇×H = J + ∂D/∂t
#
MU_0: float = 1.25663706212e-6  # [H/m] (Henrios por metro)

# ═══════════════════════════════════════════════════════════════════
# VELOCIDAD DE LA LUZ  c
# ═══════════════════════════════════════════════════════════════════
#
# c = 1/√(ε₀·μ₀) — esta relación es una predicción directa
# de las ecuaciones de Maxwell: las ondas electromagnéticas
# viajan a la velocidad de la luz.
#
SPEED_OF_LIGHT: float = 299_792_458.0  # [m/s]

# ═══════════════════════════════════════════════════════════════════
# CARGA ELEMENTAL Y MASAS DE PARTÍCULAS
# ═══════════════════════════════════════════════════════════════════
#
# La carga más pequeña que existe libre en la naturaleza.
#   Protón:   q = +e = +1.602 × 10⁻¹⁹ C
#   Electrón: q = -e = -1.602 × 10⁻¹⁹ C
#
ELEMENTARY_CHARGE: float = 1.602176634e-19  # [C] (Coulombs)
ELECTRON_MASS: float = 9.1093837015e-31      # [kg]
PROTON_MASS: float = 1.67262192369e-27       # [kg]

# ═══════════════════════════════════════════════════════════════════
# VERIFICACIÓN DE CONSISTENCIA
# ═══════════════════════════════════════════════════════════════════
# k = 1/(4πε₀) — si esto no se cumple, algo está mal.
_k_from_epsilon = 1.0 / (4.0 * math.pi * EPSILON_0)
assert abs(COULOMB_CONSTANT - _k_from_epsilon) / COULOMB_CONSTANT < 1e-6, (
    f"Inconsistencia: k = {COULOMB_CONSTANT} pero 1/(4πε₀) = {_k_from_epsilon}"
)

# c = 1/√(ε₀·μ₀) — la relación de Maxwell
_c_from_em = 1.0 / math.sqrt(EPSILON_0 * MU_0)
assert abs(SPEED_OF_LIGHT - _c_from_em) / SPEED_OF_LIGHT < 1e-6, (
    f"Inconsistencia: c = {SPEED_OF_LIGHT} pero 1/√(ε₀μ₀) = {_c_from_em}"
)
