# Resumen Ejecutivo: Metodología, Estado Actual y Tecnologías

Este documento resume la metodología de rigor físico, el estado actual de implementación y el stack tecnológico del proyecto de **Electromagnetismo y Optimización de Metamateriales**.

---

## 1. Visión y Metodología del Proyecto

El objetivo general es el **diseño y optimización computacional de metamateriales electromagnéticos** capaces de moldear y controlar la propagación de ondas (reflexión, transmisión, bandas prohibidas y resonancia), tal como se especifica en [frontend/others/docs/README.md](file:///electromagnetismo/frontend/others/docs/README.md).

Para evitar que las herramientas complejas operen como "cajas negras", se utiliza una **metodología computacional basada en el rigor físico**, estructurada de manera incremental tanto en conocimiento como en el software:

### A. Rigor Físico en el Dominio del Software
- **Un tipo por cada concepto físico:** No se usan tipos genéricos primitivos (`float`, diccionarios sin esquema) de forma aislada. Todo concepto físico se modela con su propia entidad (`Vector2D`, `Vector3D`, `Particle`, `CoulombForce`, `ElectricField`).
- **Unidades SI consistentes:** Toda magnitud interna respeta el Sistema Internacional de Unidades (Coulombs, Newtons, Metros, Voltios por metro) con transformaciones controladas a microescalas cuando la visualización lo requiere.
- **Auto-documentación pedagógica:** Cada fórmula física (Ley de Coulomb $\mathbf{F} = k\frac{q_1 q_2}{r^2}\hat{\mathbf{r}}$, superposición lineal de campos $\mathbf{E} = \sum \mathbf{E}_i$) está explícita y aislada en una capa de dominio independiente de frameworks web.

### B. Enfoque Incremental
El avance sigue una escalera de complejidad progresiva descrita en [backend/docs/plan_de_desarrollo.md](file:///electromagnetismo/backend/docs/plan_de_desarrollo.md):
1. **Paso 0 / Fundamentos:** Abstracciones algebraicas de vectores y partículas.
2. **Nivel 1 — Electrostática Discreta:** Cargas puntuales, Ley de Coulomb en 2D y 3D, prevención de singularidades ($r \to 0$).
3. **Nivel 2 — Campos Vectoriales:** Cálculo continuo de campos $\mathbf{E}(\mathbf{r})$, mallas de muestreo espacial y líneas de fuerza.
4. **Nivel 3 — Electrodinámica (Siguiente etapa):** Ecuaciones de Maxwell completas ($\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$, $\nabla \times \mathbf{H} = \mathbf{J} + \frac{\partial \mathbf{D}}{\partial t}$) y propagación temporal.
5. **Nivel 4 — Metamateriales y Optimización (Meta final):** Simulación numérica (FDTD / FEM), celdas unitarias periódicas, obtención de curvas de transmisión $T(f)$ y algoritmos de optimización inversa para encontrar geometrías óptimas.

---

## 2. Lo que Llevamos Implementado

Actualmente se encuentra culminado y funcional el bloque fundamental correspondiente a **Electrostática (Nivel 1 y Nivel 2)**:

### En el Backend (`backend/`)
- **Dominio Físico Puro ([backend/domain/](file:///electromagnetismo/backend/domain)):**
  - [vectors.py](file:///electromagnetismo/backend/domain/vectors.py): Álgebra vectorial completa en 2D y 3D (`Vector2D`, `Vector3D`), cálculo de normas, vectores unitarios y operaciones aritméticas sobrecargadas.
  - [units.py](file:///electromagnetismo/backend/domain/units.py): Constantes físicas ($\varepsilon_0$, $\mu_0$, $c$, $k_e$) y conversión de prefijos de escala.
  - [particles.py](file:///electromagnetismo/backend/domain/particles.py): Modelado formal de cargas puntuales con posición vectorial y magnitud de carga en Coulombs.
  - [coulomb.py](file:///electromagnetismo/backend/domain/coulomb.py): Ley de Coulomb, interacciones atractivas/repulsivas por pares y superposición neta.
  - [electric_field.py](file:///electromagnetismo/backend/domain/electric_field.py): Cálculo del campo electrostático $\mathbf{E}$ en puntos arbitrarios del espacio y generación de mallas vectoriales 2D/3D.
- **Servicios y APIs ([backend/api/](file:///electromagnetismo/backend/api)):**
  - **Endpoints REST (FastAPI):** Exposición de constantes, cálculo de fuerzas y mallas de campo eléctrico en `/api/physics/*`.
  - **WebSockets en Tiempo Real:** Canal reactivo `/ws/physics` para cómputo dinámico y continuo al interactuar con las partículas desde el frontend.

### En el Frontend (`frontend/`)
- **Sistema de Tipos Espejo ([frontend/src/types/](file:///electromagnetismo/frontend/src/types)):**
  - Modelos en TypeScript que reflejan fielmente las entidades físicas del backend (`Vector2D`, `Vector3D`, `Particle`, `FieldPoint`).
- **Capítulos Interactivos ([frontend/src/pages/chapters/](file:///electromagnetismo/frontend/src/pages/chapters)):**
  - **Capítulo 1 ([Chapter1_Charges.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter1_Charges.tsx)):** Introducción a la carga eléctrica, polaridades e interacción básica.
  - **Capítulo 2 ([Chapter2_Coulomb.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter2_Coulomb.tsx)):** Ley de Coulomb en 2D interactiva, variación de distancias y cargas.
  - **Capítulo 3 ([Chapter3_Forces3D.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter3_Forces3D.tsx)):** Fuerzas en 3D con visualización espacial tridimensional interactiva y control de cámara orbital.
  - **Capítulo 4 ([Chapter4_ElectricField.tsx](file:///electromagnetismo/frontend/src/pages/chapters/Chapter4_ElectricField.tsx)):** Visualización de campos vectoriales en el plano y muestreo continuo.
- **Navegación y Documentación en UI:**
  - [HomePage.tsx](file:///electromagnetismo/frontend/src/pages/HomePage.tsx) con el índice temático y barra de navegación por etapas conceptuales.

---

## 3. Tecnologías que Estamos Usando

| Área | Tecnología | Propósito |
| :--- | :--- | :--- |
| **Backend Core** | **Python 3.12+** | Lenguaje para modelar la lógica y ecuaciones de la física de dominio. |
| **Backend Framework** | **FastAPI** + **Uvicorn** | Framework asíncrono para exponer endpoints REST y WebSockets de alto rendimiento. |
| **Cómputo Numérico** | **NumPy** | Operaciones matriciales y cálculo vectorizado sobre grillas de simulación. |
| **Validación de Datos** | **Pydantic v2** | Validación estricta de esquemas, tipos y contratos entre frontend y backend. |
| **Testing** | **Pytest** | Verificación de algoritmos y leyes físicas (pruebas de Coulomb, vectores, singularidades). |
| **Frontend Framework** | **React 19** + **TypeScript** | Interfaz de usuario reactiva, modular y con tipado estricto. |
| **Bundler & Tooling** | **Vite** | Entorno de desarrollo rápido y empaquetador moderno. |
| **Visualización 3D** | **Three.js**, **@react-three/fiber**, **@react-three/drei** | Renderizado WebGL de escenas 3D (cargas, vectores espaciales de fuerza, cámaras orbitales). |
| **Notación Científica** | **KaTeX** | Renderizado tipográfico preciso de fórmulas matemáticas y ecuaciones en LaTeX. |
| **Gráficos y Curvas** | **Recharts** | Visualización de gráficos 2D, curvas de transmisión y análisis de respuesta física. |
| **Estilos** | **Vanilla CSS Moderno** | Diseño oscuro ("dark mode"), componentes translúcidos tipo glassmorphism y diseño responsive sin dependencias pesadas. |
