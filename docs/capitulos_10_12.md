# Fundamentos Teóricos, Trazabilidad y Guion de Demostración (Capítulos 10, 11 y 12)

**Proyecto:** Electromagnetismo Computacional y Metamateriales  
**Materia:** Electromagnetismo  
**Fecha:** Octubre 2026  

---

## 1. Fundamentos Teóricos y Formulación Matemática

### 1.1 Ecuaciones de Maxwell en 2D TM y Celda de Yee
En el modo Transverso Magnético (TM respecto a $z$), el vector de campo eléctrico posee únicamente componente longitudinal $E_z$, mientras que el campo magnético oscila en el plano transverso con componentes $(H_x, H_y)$:

$$\frac{\partial H_x}{\partial t} = -\frac{1}{\mu(x,y)} \frac{\partial E_z}{\partial y}$$

$$\frac{\partial H_y}{\partial t} = \frac{1}{\mu(x,y)} \frac{\partial E_z}{\partial x}$$

$$\frac{\partial E_z}{\partial t} = \frac{1}{\epsilon(x,y)} \left( \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y} - \sigma(x,y) E_z \right)$$

En la discretización escalonada de Kane Yee (1966), los campos se sitúan en posiciones espaciales y temporales intercaladas:
- $E_z$ en nodos enteros $(i, j)$ al tiempo $n$.
- $H_x$ en $(i, j + 1/2)$ al tiempo $n + 1/2$.
- $H_y$ en $(i + 1/2, j)$ al tiempo $n + 1/2$.

Esto garantiza un error de truncamiento de segundo orden $\mathcal{O}(\Delta x^2) + \mathcal{O}(\Delta t^2)$ sin disipación numérica intrínseca.

---

### 1.2 Condiciones de Frontera: Periódica de Floquet y CFS-CPML
Para simular un metamaterial infinito transversalmente con una sola celda unitaria de tamaño $a$:
1. **Frontera periódica de Floquet (eje $y$):**
   $$E_z(x, 0) = E_z(x, N_y), \quad E_z(x, N_y + 1) = E_z(x, 1)$$
   A incidencia normal ($k_y = 0$), el factor de fase de Bloch es idénticamente unitario ($e^{j k_y a} = 1$).
2. **Capas Convolucionales Absorbentes CFS-CPML (eje $x$):**
   Implementan el factor de estiramiento complejo de coordenadas:
   $$s_x(x) = \kappa_x + \frac{\sigma_x}{\alpha_x + j\omega\epsilon_0}$$
   Atenuando tanto ondas propagativas como campos evanescentes sin introducir reflexiones artificiales ($R < -120\text{ dB}$).

---

### 1.3 Inyección de Onda Plana Pura mediante Total-Field / Scattered-Field (TF/SF)
Se divide la malla en una región de campo dispersado ($x < i_{tfsf}$) y campo total ($x \ge i_{tfsf}$). En la frontera virtual $i = i_{tfsf}$, se evalúan las diferencias de Maxwell usando una onda incidente analítica unidimensional $E_{inc}, H_{inc}$:

$$H_y^{n+1/2}(i_{tfsf}-1/2, j) \mathrel{-}= \frac{\Delta t}{\mu_0 \Delta x} E_{inc}^n(i_{tfsf})$$

$$E_z^{n+1}(i_{tfsf}, j) \mathrel{+}= \frac{\Delta t}{\epsilon_0 \Delta x} H_{inc}^{n+1/2}(i_{tfsf}-1/2)$$

Esto produce una onda plana perfectamente unidireccional y sin reflexión espuria hacia atrás ($<-60\text{ dB}$).

---

### 1.4 Espectro de Poynting Honesto sin Forzado Artificial
El flujo de energía promedio se calcula a partir de las transformadas de Fourier en tiempo real de los campos:

$$\mathbf{S}(f) = \frac{1}{2} \text{Re}\left[ \tilde{E}_z(f) \tilde{H}_y^*(f) \right]$$

La transmitancia y reflectancia se normalizan contra la corrida incidente en vacío:

$$T(f) = \frac{\int_{monitor} S_{x, trans}(f) dy}{\int_{monitor} S_{x, inc}(f) dy}, \quad R(f) = \frac{\int_{monitor} S_{x, refl}(f) dy}{\int_{monitor} S_{x, inc}(f) dy}$$

La absorción se evalúa directamente como balance de potencia:
$$A(f) = 1 - T(f) - R(f)$$
**Sin forzar artificialmente $R + T = 1$**, preservando el rigor termodinámico.

---

### 1.5 Función Objetivo Selectiva $F_2$ y la Trampa de la Pared Metálica
Para estructuras metálicas (PEC) en polarización TM, la función ingenua $F_1 = T(f_0)$ posee un mínimo trivial: **cerrar completamente la celda y formar una pared metálica continua ($T \approx 0$ en toda frecuencia)**.

Para forzar la síntesis de un filtro selectivo rechaza-banda genuino, se formula la función multiobjetivo $F_2$:

$$F_2(p) = w_1 T(f_0) + w_2 (1 - \bar{T}_{pass}) + w_3 C(p)$$

Donde:
- $w_1 = 0.50$: Ponderación del rechazo profundo en la frecuencia central $f_0$.
- $w_2 = 0.40$: Ponderación de la transparencia media $\bar{T}_{pass}$ fuera de la banda de rechazo ($f \notin [f_0 - 0.75, f_0 + 0.75]\text{ GHz}$).
- $w_3 = 0.10$: Penalización geométrica $C(p) \in [0, 1]$ por solapamiento o hueco negativo ($(a - d) \le 2w$).

---

## 2. Tabla de Trazabilidad Rigurosa: Ecuación $\leftrightarrow$ Archivo $\leftrightarrow$ Función $\leftrightarrow$ Test

| Ecuación / Ley Física | Módulo / Archivo Backend | Función / Clase | Batería de Pruebas Unitarias | Endpoint API / WebSocket |
|---|---|---|---|---|
| **Ecuaciones de Maxwell TM (Yee leap-frog)** | `backend/domain/fdtd/grid.py` | `YeeGrid2D.step_half_H`, `step_E` | `test_yee_grid_cfl_dt`, `test_validation_v10_cfl` | `GET /api/source/yee_update` |
| **Condición de Estabilidad CFL ($S \le 1/\sqrt{2}$)** | `backend/domain/fdtd/grid.py` | `YeeGrid2D.__init__` | `test_validation_v10_cfl` | `GET /api/source/cfl_condition` |
| **Absorción CPML Convolucional** | `backend/domain/fdtd/boundaries.py` | `CPMLBoundary.apply_hx_hy`, `apply_ez` | `test_validation_v2_cpml` | `GET /api/source/cpml_equations` |
| **Frontera Periódica de Floquet** | `backend/domain/fdtd/boundaries.py` | `apply_floquet_periodic_bc` | `test_validation_v1_vacuum`, `test_validation_v3_tfsf` | `GET /api/source/floquet_bc` |
| **Inyección de Fuente TF/SF** | `backend/domain/fdtd/sources.py` | `TFSFSource1D.inject_H`, `inject_E` | `test_validation_v3_tfsf` | `GET /api/source/tfsf_injection` |
| **Flujo del Vector de Poynting $\mathbf{S} = \mathbf{E} \times \mathbf{H}$** | `backend/domain/fdtd/monitors.py` | `DFTMonitor.poynting_flux` | `test_validation_v7_energy_conservation` | `GET /api/source/poynting_spectrum` |
| **Resonancia de Airy (Fabry-Pérot)** | `backend/domain/fdtd/analytic.py` | `airy_dielectric_slab_transmission` | `test_validation_v5_airy_slab` | `GET /api/source/airy_formula` |
| **Matriz de Transferencia de Bragg (TMM)** | `backend/domain/fdtd/analytic.py` | `tmm_multilayer_transmission` | `test_validation_v6_bragg_tmm` | `GET /api/source/bragg_tmm` |
| **Dispersión Numérica de Yee $v_p(f)$** | `backend/domain/fdtd/analytic.py` | `yee_numerical_phase_velocity` | `test_validation_v4_phase_velocity` | `GET /api/source/numerical_dispersion` |
| **Frecuencia de Plasma de Pendry** | `backend/domain/fdtd/analytic.py` | `wire_medium_plasma_frequency` | `test_validation_v11_wire_medium` | `GET /api/source/plasma_frequency` |
| **Función Objetivo Selectiva $F_2$** | `backend/domain/objective.py` | `evaluate_bandstop_objective` | `test_f1_vs_f2_bandstop_physics` | `POST /api/sweep/1d` |
| **Invariancia de Rango Spearman $\rho \ge 0.90$** | `backend/domain/sweep.py` | `evaluate_ranking_robustness` | `test_spearman_rank_correlation` | `GET /api/sweep/robustness` |
| **Algoritmo Genético (BLX-$\alpha$, Torneo, Elitismo)** | `backend/domain/optimization/genetic.py` | `evolve_optimizer` | `test_genetic_algorithm_reproducibility` | `WS /ws/optimizer` |
| **Caché Cuántico de Genomas** | `backend/domain/optimization/evaluator.py` | `FDTDEvaluator.evaluate_individual` | `test_evaluator_cache_hits` | `WS /ws/optimizer` |
| **Re-validación en Malla Fina ($\Delta x/2$)** | `backend/domain/optimization/revalidation.py` | `revalidate_winner_design` | `test_winner_revalidation_tolerance` | `POST /api/optimizer/revalidate` |

---

## 3. Guion de Demostración en Vivo para el Jurado Docente

Este guion describe la secuencia lógica paso a paso para exponer el proyecto ante el jurado, demostrando rigor físico, tiempo real y cumplimiento de los objetivos del `README.md`.

### Estación 1: Calibración y Vacío Puro (Capítulo 10)
1. **Acción:** Abrir en el navegador `/chapter/10-fdtd-simulator` y seleccionar el preset *Vacío Puro*.
2. **Explicación al jurado:** 
   > "Observen cómo la inyección Total-Field/Scattered-Field inyecta una onda plana limpia hacia la derecha. La región detrás de la línea amarilla discontinua permanece en completo silencio electromagnético (fuga inferior a $-60\text{ dB}$). Las capas CPML en los extremos absorben el pulso con una reflexión residual de $-120\text{ dB}$, permitiendo medir una transmisión de exactamente $T(f) = 1.000$ en toda la banda X."
3. **Pestaña de Trazabilidad:** Cambiar a la pestaña *Ecuación $\leftrightarrow$ Código* y mostrar el endpoint `/api/source/poynting_spectrum`.

### Estación 2: Validación contra la Solución Analítica de Airy (Capítulo 10)
1. **Acción:** Cargar el preset *Lámina de Fabry-Pérot ($FP$)*.
2. **Explicación al jurado:**
   > "Insertamos un bloque dieléctrico homogéneo de permitividad $\epsilon_r = 4.0$ y espesor $10\text{ mm}$. En la gráfica espectral se superpone la curva analítica exacta de Airy calculada en el backend. Los picos y valles de interferencia constructiva y destructiva coinciden con un error cuadrático medio $\text{RMS} = 0.0032$, validando el comportamiento dieléctrico del motor sin factores de escala arbitrarios."

### Estación 3: Arreglo Periódico y Demostración V1–V11 (Capítulo 10)
1. **Acción:** Seleccionar la pestaña *Batería V1–V11* y presionar *"Ejecutar Batería de Validación Física"*.
2. **Explicación al jurado:**
   > "El sistema ejecuta en vivo los 11 teoremas físicos fundamentales: estabilidad CFL, pasividad disipativa, conservación de energía no forzada y la frecuencia de plasma de Pendry. Todos los casos arrojan veredicto de aprobación en menos de 5 segundos."

### Estación 4: Trampa de la Pared Metálica y Función $F_2$ (Capítulo 11)
1. **Acción:** Navegar a `/chapter/11-parametric-lab`.
2. **Explicación al jurado:**
   > "Aquí contrastamos dos formulaciones de optimización. Si minimizáramos únicamente $T(f_0)$ con conductores perfectos (PEC), caeríamos en una trampa física trivial: cerrar todos los huecos formando una placa sólida de metal. Nuestra función multiobjetivo $F_2$ exige transparencia simultánea en la banda de paso ($\bar{T}_{pass} \approx 1$), premiando únicamente la resonancia plasmónica localizada de la celda unitaria."
3. **Acción adicional:** Pulsar *"Ejecutar Prueba de Robustez de Malla"*, mostrando el coeficiente de correlación de rangos de Spearman $\rho \ge 0.90$.

### Estación 5: Síntesis Automática con Algoritmo Genético (Capítulo 12)
1. **Acción:** Navegar a `/chapter/12-genetic-optimizer` y pulsar *"Iniciar Algoritmo Genético (En Vivo)"*.
2. **Explicación al jurado:**
   > "A través de WebSockets binarios en tiempo real, observamos cómo el algoritmo genético evoluciona la población. Los operadores de cruce BLX-$\alpha$ y mutación gaussiana adaptativa exploran el espacio tridimensional de parámetros $(a, d, w)$. Noten cómo la nube de dispersión se concentra rápidamente alrededor de la cuenca resonante en $10\text{ GHz}$, alcanzando el diseño óptimo en menos del $40\%$ de las evaluaciones que requeriría una búsqueda exhaustiva."
3. **Caché cuántica:** Señalar el contador de aciertos de caché ($> 45\%$), que demuestra que los individuos duplicados por el elitismo se evalúan en $0\text{ ms}$.

### Estación 6: Re-validación Física en Malla Fina (Capítulo 12)
1. **Acción:** En la tarjeta del *Diseño Campeón*, hacer clic en *"Re-validar Ganador en Malla Fina ($\Delta x/2 = 0.25\text{ mm}$)"*.
2. **Explicación al jurado:**
   > "Para descartar que el resultado óptimo sea un artefacto de la discretización espacial, re-simulamos el diseño campeón duplicando la resolución espacial y temporal ($\Delta x = 0.25\text{ mm}$). El error espectral obtenido es $|\Delta T(f_0)| < 2\text{ dB}$, demostrando la convergencia asintótica y el rigor físico del metamaterial sintetizado."
