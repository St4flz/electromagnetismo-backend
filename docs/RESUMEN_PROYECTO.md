# Resumen Ejecutivo: Metodología, Estado Actual y Tecnologías

Este documento resume la metodología de rigor físico, el estado actual de implementación y el stack tecnológico del proyecto de **Electromagnetismo y Optimización de Metamateriales**.

---

## 1. Visión y Metodología del Proyecto

El objetivo general es el **diseño y optimización computacional de metamateriales electromagnéticos** capaces de moldear y controlar la propagación de ondas (reflexión, transmisión, bandas prohibidas y resonancia), tal como se especifica en [segunda_entrega_proyecto.md](file:///electromagnetismo/backend/docs/segunda_entrega_proyecto.md) y [frontend/others/docs/README.md](file:///electromagnetismo/frontend/others/docs/README.md).

Para evitar que las herramientas complejas operen como "cajas negras", se utiliza una **metodología computacional basada en el rigor físico**, estructurada de manera incremental tanto en conocimiento como en el software:

### A. Rigor Físico en el Dominio del Software
- **Un tipo por cada concepto físico:** No se usan tipos genéricos primitivos (`float`, diccionarios sin esquema) de forma aislada. Todo concepto físico se modela con su propia entidad (`Vector2D`, `Vector3D`, `Particle`, `CoulombResult`, `FieldPoint`, `DynamicParticle`, `PlaneWave2D`, `MetamaterialUnitCell`, `FDTD2DSimulator`).
- **Unidades SI consistentes:** Toda magnitud interna respeta el Sistema Internacional de Unidades (Coulombs, Newtons, Metros, Voltios por metro, Teslas, Henrios por metro, Faradios por metro) con transformaciones controladas a microescalas cuando la visualización lo requiere.
- **Auto-documentación pedagógica y guiones comprobables:** Cada ley física está respaldada por una formulación analítica exacta y un guión comprobable que compara la simulación numérica contra la teoría cerrada, evaluando el error relativo porcentual.

### B. Enfoque Incremental por Capítulos
El avance cubre los 9 capítulos del programa:
1. **Capítulo 1 — Partículas y Carga Eléctrica:** Naturaleza discreta de la carga elemental y polaridades duales en 2D.
2. **Capítulo 2 — Ley de Coulomb 2D:** Fuerza electrostática vectorial mutua y 3ª Ley de Newton ($F_{12} = -F_{21}$).
3. **Capítulo 3 — Interacción Tridimensional:** Espacio euclidiano continuo 3D renderizado con Three.js / WebGL.
4. **Capítulo 4 — Campo Eléctrico Vectorial:** Perturbación local del espacio y principio de superposición lineal analítica.
5. **Capítulo 5 — Campo Eléctrico en Movimiento (2D):** Cinemática de cargas aceleradas por campo $\vec{E}$, integrador simpléctico Velocity-Verlet y deflexión balística de Thomson ($y(L) = \frac{q E_y L^2}{2 m v_{0x}^2}$).
6. **Capítulo 6 — Ondas y Campo Magnético (2D):** Ley de Biot-Savart, Fuerza de Lorentz $\vec{F} = q(\vec{E} + \vec{v} \times \vec{B})$, algoritmo simpléctico de Boris, órbita de ciclotrón ($R_c = \frac{m v}{|q| B_z}$), selector de velocidades de Wien y teorema de trabajo nulo ($\Delta K = 0$).
7. **Capítulo 7 — Onda Electromagnética Ideal en el Vacío (2D):** Acoplamiento rotacional de Maxwell ($\nabla \times \vec{E} = -\partial\vec{B}/\partial t$, $\nabla \times \vec{B} = \mu_0\varepsilon_0\partial\vec{E}/\partial t$), velocidad $c = 1/\sqrt{\varepsilon_0\mu_0} = 299,792,458\text{ m/s}$, impedancia del vacío $\eta_0 = \sqrt{\mu_0/\varepsilon_0} \approx 376.73\,\Omega$, vector de Poynting $\vec{S} = \frac{1}{\mu_0}(\vec{E} \times \vec{B})$ y equipartición de energía ($u_E = u_B$).
8. **Capítulo 8 — Introducción a Metamateriales:** Estructuras periódicas con celda unitaria de parámetros $(a, d, w)$, modelos constitutivos Drude-Lorentz, medios zurdos DNG ($\text{Re}(\varepsilon) < 0$ y $\text{Re}(\mu) < 0 \implies n_{\text{eff}} < 0$), refracción negativa de Veselago, método TMM para transmisión $T(f)$ y reflexión $R(f)$ con conservación estricta de energía ($R + T + A = 1$).
9. **Capítulo 9 — Introducción al Método FDTD:** Discretización en celda escalonada de Yee en 2D (modo TM: $E_z, H_x, H_y$), condición de estabilidad de Courant-Friedrichs-Lewy ($S \le 1/\sqrt{2} \approx 0.7071$), fronteras absorbentes Mur ABC de 1er orden, y espectro FFT de potencia transmitida.

---

## 2. Componentes en Funcionamiento

### En el Backend (`backend/`)
- **Dominio Físico Puro ([backend/domain/](file:///electromagnetismo/backend/domain)):**
  - [vectors.py](file:///electromagnetismo/backend/domain/vectors.py): Álgebra vectorial en 2D y 3D.
  - [units.py](file:///electromagnetismo/backend/domain/units.py): Constantes físicas universales ($\varepsilon_0$, $\mu_0$, $c$, $k_e$).
  - [particles.py](file:///electromagnetismo/backend/domain/particles.py): Modelado formal de fuentes puntuales.
  - [coulomb.py](file:///electromagnetismo/backend/domain/coulomb.py): Ley de Coulomb e interacciones por pares.
  - [electric_field.py](file:///electromagnetismo/backend/domain/electric_field.py): Muestreo espacial de campo $\mathbf{E}$.
  - [dynamics.py](file:///electromagnetismo/backend/domain/dynamics.py): Velocity-Verlet y deflexión balística de Thomson.
  - [magnetic.py](file:///electromagnetismo/backend/domain/magnetic.py): Biot-Savart 2D, Fuerza de Lorentz, integrador de Boris, ciclotrón y filtro de Wien.
  - [em_wave.py](file:///electromagnetismo/backend/domain/em_wave.py): Onda plana ideal en el vacío 2D, vector de Poynting, equipartición e impedancia $\eta_0$.
  - [metamaterials.py](file:///electromagnetismo/backend/domain/metamaterials.py): Celda periódica $(a, d, w)$, medios DNG y matriz de transferencia TMM para $T(f)$ y $R(f)$.
  - [fdtd2d.py](file:///electromagnetismo/backend/domain/fdtd2d.py): Algoritmo de Yee 2D TM, Mur ABC, condición CFL y cálculo FFT de transmisión.
- **Servicios y APIs ([backend/api/](file:///electromagnetismo/backend/api)):**
  - **Endpoints REST (FastAPI):** Endpoints validados con Pydantic v2 para consultas y guiones de verificación física.
  - **Canal WebSocket en Tiempo Real (`/ws/physics`):** Streaming bidireccional a 60 fps para todos los capítulos.
- **Suite de Pruebas Automatizadas ([backend/tests/](file:///electromagnetismo/backend/tests)):**
  - 30 pruebas unitarias con `pytest` que verifican la consistencia matemática, conservación de energía y límites numéricos de cada ley física.

### En el Frontend (`frontend/`)
- **Sistema de Tipos Espejo ([frontend/src/types/physics.ts](file:///electromagnetismo/frontend/src/types/physics.ts)):**
  - Modelos TypeScript sincronizados con los schemas Pydantic.
- **Páginas de Capítulos Interactivos ([frontend/src/pages/chapters/](file:///electromagnetismo/frontend/src/pages/chapters)):**
  - [Chapter1_Charges.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter1_Charges.tsx): Cargas y cuantización elemental.
  - [Chapter2_Coulomb.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter2_Coulomb.tsx): Ley de Coulomb y 3ª Ley de Newton en 2D.
  - [Chapter3_Forces3D.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter3_Forces3D.tsx): Fuerzas en 3D con Three.js / WebGL.
  - [Chapter4_ElectricField.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter4_ElectricField.tsx): Campo eléctrico continuo en 2D.
  - [Chapter5_MovingCharge.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter5_MovingCharge.tsx): Cinemática y deflexión balística de Thomson con guión analítico.
  - [Chapter6_MagneticField.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter6_MagneticField.tsx): Campo magnético 2D, integrador Boris, órbita de ciclotrón y filtro de Wien.
  - [Chapter7_EMWaveVacuum.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter7_EMWaveVacuum.tsx): Onda en el vacío 2D, equipartición, impedancia $\eta_0$ y flujo de Poynting.
  - [Chapter8_Metamaterials.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter8_Metamaterials.tsx): Estructura periódica $(a, d, w)$, gráfica TMM de $T(f)$, medios DNG y refracción negativa.
  - [Chapter9_FDTDIntro.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter9_FDTDIntro.tsx): Rejilla Yee FDTD 2D TM en tiempo real, colormap divergente, verificación CFL y FFT espectral.
- **Portal Principal ([frontend/src/pages/HomePage.tsx](file:///electromagnetismo/frontend/src/pages/HomePage.tsx)):**
  - Catálogo general con los 9 capítulos y navegación lateral interactiva ([Sidebar.tsx](file:///electromagnetismo/frontend/src/components/layout/Sidebar.tsx)).

---

## 3. Guiones de Comprobación Física Implementados

Cada capítulo incluye un panel interactivo de verificación que ejecuta experimentos parametrizados en el backend y los compara frente a la solución analítica exacta:

1. **Guión 05 (Deflexión de J.J. Thomson):**
   $$\text{Error Relativo} = \frac{|y_{\text{num}} - y_{\text{analítico}}|}{y_{\text{analítico}}} \times 100\% < 0.5\%$$
2. **Guión 06 (Radio Ciclotrónico y Trabajo Magnético Nulo):**
   $$R_c = \frac{m v}{|q| B_z}, \quad T_c = \frac{2\pi m}{|q| B_z}, \quad \Delta K = 0.00000\text{ J}$$
3. **Guión 07 (Constante c e Impedancia de Vacío de Maxwell):**
   $$v = \lambda f \equiv c = \frac{1}{\sqrt{\varepsilon_0 \mu_0}}, \quad \frac{E}{B} = c, \quad \eta_0 = \sqrt{\frac{\mu_0}{\varepsilon_0}} \approx 376.73\,\Omega$$
4. **Guión 08 (Conservación de Energía TMM y Refracción Negativa de Veselago):**
   $$R(f) + T(f) + A(f) \equiv 1.000000, \quad \theta_2 = \arcsin\left(\frac{1}{n_{\text{eff}}}\sin\theta_1\right) < 0$$
5. **Guión 09 (Teorema de Estabilidad CFL y Rapidez de Grupo FDTD):**
   $$S = c \frac{\Delta t}{\Delta x} \le \frac{1}{\sqrt{2}} \approx 0.7071, \quad v_{\text{pulso}} = \frac{\Delta x}{\Delta t} \approx c$$
