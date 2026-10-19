# Simulación numérica y optimización computacional de estructuras periódicas (metamateriales teóricos) para el control de ondas electromagnéticas

*Modelado bidimensional mediante diferencias finitas en el dominio del tiempo (FDTD) y algoritmos genéticos*

---

## I. Planteamiento del Proyecto (1a Revisión Actualizada)

### 1. Resumen

El diseño de estructuras electromagnéticas periódicas (metamateriales teóricos) permite controlar selectivamente la reflexión y transmisión de ondas a partir de la geometría de su celda unitaria, pero hallar configuraciones óptimas por ensayo y error resulta ineficiente. Este proyecto exclusivamente computacional desarrolla una plataforma interactiva para simular y optimizar numéricamente dichas estructuras en la banda de microondas sin requerir fabricación física. Mediante el método de diferencias finitas en el dominio del tiempo (FDTD) en 2D, el sistema resuelve las ecuaciones de Maxwell sobre una rejilla discreta, valida el modelo frente a tres casos de referencia (vacío, medio convencional y arreglo periódico) y evalúa cómo las dimensiones geométricas $(a, d, w)$ alteran el espectro de transmisión $T(f)$. Finalmente, integra una búsqueda sistemática y un algoritmo genético que minimizan automáticamente una función objetivo (como la transmisión a $f_0 = 10\text{ GHz}$) para identificar la geometría óptima *in silico*.

* **Palabras clave:** simulación electromagnética, estructuras periódicas, metamateriales teóricos, ecuaciones de Maxwell, FDTD, algoritmos genéticos, optimización computacional.
* **Modalidad:** Computacional (simulación numérica y modelado teórico).

### 2. Objetivos

#### Objetivo general
Desarrollar un sistema computacional capaz de simular, diseñar y optimizar numéricamente una estructura periódica teórica (metamaterial simulado) para controlar la propagación de ondas electromagnéticas, minimizando su transmisión a una frecuencia objetivo $f_0$ (por ejemplo, $10\text{ GHz}$) en la banda de microondas mediante simulación FDTD bidimensional y algoritmos de optimización.

#### Objetivos específicos
1. **Implementar un simulador FDTD bidimensional** que resuelva numéricamente las ecuaciones de Maxwell sobre una rejilla espacial, calcule los espectros de reflexión $R(f)$ y transmisión $T(f)$ de una estructura teórica y visualice la propagación del campo electromagnético.
2. **Validar el simulador computacional** con tres casos de referencia (espacio vacío, lámina de material convencional y estructura periódica), comprobando que los resultados numéricos tengan sentido físico.
3. **Determinar la influencia geométrica** del tamaño de la celda $(a)$, la separación $(d)$ y el ancho de los elementos $(w)$ sobre la curva de transmisión $T(f)$, construyendo un conjunto de datos que relacione la geometría con la respuesta electromagnética.
4. **Definir una función objetivo $F$** que cuantifique el desempeño del diseño: inicialmente $F = T(f_0)$ y, posteriormente, combinaciones ponderadas de transmisión y reflexión.
5. **Implementar la optimización automática** mediante búsqueda sistemática y algoritmo genético, comparando la eficiencia y los diseños teóricos obtenidos con cada estrategia.

### 3. Método elegido

**Enfoque computacional incremental.** Se resuelven numéricamente las ecuaciones de Maxwell dependientes del tiempo:

$$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t} \quad \text{y} \quad \nabla \times \vec{H} = \vec{J} + \frac{\partial \vec{D}}{\partial t}$$

empleando el método de diferencias finitas en el dominio del tiempo (FDTD) dentro de un motor de simulación propio con arquitectura cliente-servidor. Para evitar que el simulador opere como una "caja negra", el desarrollo se estructura de forma incremental en cuatro niveles de complejidad física (Electrostática Discreta $\to$ Campos Vectoriales Continuos $\to$ Electrodinámica FDTD 2D $\to$ Optimización Inversa de Metamateriales) y cuatro etapas metodológicas:

1. **Simulador FDTD 2D:** Discretización espacial y temporal mediante la celda de Yee (polarización TM: $E_z, H_x, H_y$), caracterizando cada punto por su permitividad $\varepsilon$, permeabilidad $\mu$ y conductividad $\sigma$. Se inyecta un pulso gaussiano de banda ancha con fronteras absorbentes en el eje de propagación y condiciones periódicas transversales, obteniendo $R(f)$ y $T(f)$ mediante la transformada de Fourier de los campos normalizados respecto al vacío.
2. **Validación física:** Contraste riguroso frente a propagación en espacio vacío (velocidad $c$ y ausencia de reflexiones espurias), lámina dieléctrica homogénea (comparación con solución analítica) y estructura periódica sin pérdidas (conservación de energía $R + T = 1$).
3. **Exploración paramétrica y función objetivo:** Barrido sobre las variables geométricas $\mathbf{p} = (a, d, w)$ y la frecuencia $f$ para construir curvas $T(f)$ y evaluar $F(\mathbf{p}) = T(f_0; \mathbf{p})$.
4. **Optimización automática:** Generación algorítmica de candidatos comparando un barrido sistemático de referencia frente a un algoritmo genético (poblaciones de $\sim 100$ diseños a lo largo de $\sim 50$ generaciones con operadores de selección, cruce y mutación).

---

## II. Desarrollo del Proyecto (2a Revisión)

### 1. Marco teórico

#### 1.1. Electromagnetismo y campos

El electromagnetismo estudia las interacciones entre las cargas eléctricas, las corrientes y los campos eléctricos y magnéticos. Una carga eléctrica es una propiedad física de la materia que puede ser positiva o negativa. Las cargas del mismo signo se repelen y las de signo opuesto se atraen. La carga eléctrica se conserva y, para partículas elementales como el electrón y el protón, su magnitud es ($e \approx 1{,}602 \times 10^{-19}\,\mathrm{C}$), con signo negativo para el electrón y positivo para el protón.

Un **campo** es una magnitud física definida en cada punto del espacio y, cuando corresponde, en cada instante. El campo eléctrico ($\vec{E}$) representa la fuerza eléctrica por unidad de carga positiva de prueba. Por ello, la fuerza que experimenta una carga ($q$) en un campo eléctrico es:

$$\vec{F} = q\vec{E}.$$

El campo eléctrico es vectorial: tiene magnitud, dirección y sentido. El campo generado por varias fuentes se calcula mediante el **principio de superposición**, que establece que el campo total es la suma vectorial de los campos individuales:

$$\vec{E}_{\mathrm{total}} = \sum_{i=1}^{N} \vec{E}_i.$$

Por tanto, deben sumarse por separado las componentes de cada campo; no se suman simplemente sus magnitudes salvo que tengan la misma dirección y sentido.

Para una carga puntual ($Q$), la magnitud del campo eléctrico en el vacío a una distancia ($r$) es:

$$E = \frac{1}{4\pi\varepsilon_0}\frac{|Q|}{r^2} = k\frac{|Q|}{r^2},$$

donde ($\varepsilon_0$) es la permitividad eléctrica del vacío y ($k = 1/(4\pi\varepsilon_0)$). El campo apunta hacia afuera de una carga positiva y hacia la carga si esta es negativa. La dependencia ($1/r^2$) indica que, al duplicar la distancia a una carga puntual, la magnitud del campo se reduce a la cuarta parte.

La ley de Coulomb describe la fuerza entre dos cargas puntuales:

$$F = k\frac{|q_1 q_2|}{r^2}.$$

La forma vectorial incorpora la dirección y el signo de las cargas, lo que permite distinguir la atracción entre cargas de signos opuestos de la repulsión entre cargas del mismo signo. Estas relaciones introducen los conceptos de fuente, dirección e intensidad del campo que se emplean al interpretar las simulaciones electromagnéticas.

#### 1.2. Ecuaciones de Maxwell

Las ecuaciones de Maxwell constituyen la descripción clásica fundamental de los campos eléctricos y magnéticos. En forma diferencial, en el sistema SI y usando ($\vec{E}$), ($\vec{B}$), la densidad de carga libre ($\rho$) y la densidad de corriente ($\vec{J}$), se escriben:

**Ley de Gauss para el campo eléctrico**

$$\nabla \cdot \vec{E} = \frac{\rho}{\varepsilon_0}.$$

Relaciona las cargas eléctricas con la divergencia del campo eléctrico: las cargas positivas actúan como fuentes y las negativas como sumideros.

**Ley de Gauss para el magnetismo**

$$\nabla \cdot \vec{B} = 0.$$

Expresa que, en el electromagnetismo clásico, no se observan monopolos magnéticos aislados. Las líneas del campo magnético no comienzan ni terminan en una carga magnética individual, sino que forman lazos cerrados.

**Ley de Faraday**

$$\nabla \times \vec{E} = -\frac{\partial\vec{B}}{\partial t}.$$

Un campo magnético que cambia con el tiempo está asociado con un campo eléctrico circulante. El signo negativo expresa la ley de Lenz: la inducción tiene una orientación que se opone al cambio de flujo magnético que la produce.

**Ley de Ampère-Maxwell**

$$\nabla \times \vec{B} = \mu_0\vec{J} + \mu_0\varepsilon_0\frac{\partial\vec{E}}{\partial t}.$$

Las corrientes eléctricas y los campos eléctricos variables en el tiempo están asociados con la circulación del campo magnético. El término añadido por Maxwell, ($\mu_0\varepsilon_0\,\partial\vec{E}/\partial t$), es esencial para describir la propagación de ondas electromagnéticas.

En estas expresiones, ($\nabla \cdot$) (divergencia) describe localmente la condición de fuente o sumidero de un campo, mientras que ($\nabla \times$) (rotacional) describe su circulación local. ($\partial/\partial t$) representa el cambio temporal. Las constantes ($\mu_0$) y ($\varepsilon_0$) son, respectivamente, la permeabilidad y la permitividad del vacío.

En el vacío y lejos de cargas y corrientes, las ecuaciones de Faraday y Ampère-Maxwell se acoplan: los campos eléctricos y magnéticos variables pueden propagarse como una onda electromagnética. En una onda plana ideal, los campos ($\vec{E}$) y ($\vec{B}$) son perpendiculares entre sí y a la dirección de propagación. La rapidez de la luz en el vacío es aproximadamente ($c = 3{,}00 \times 10^8\,\mathrm{m/s}$).

#### 1.3. Ondas electromagnéticas: frecuencia y longitud de onda

Una onda electromagnética es una perturbación propagante de los campos eléctrico y magnético. No requiere un medio material para viajar. Las ondas de radio, microondas, luz visible, rayos X y otras regiones del espectro electromagnético son manifestaciones del mismo fenómeno; se diferencian principalmente por su frecuencia y longitud de onda.

La **frecuencia** ($f$), medida en hercios (Hz), indica cuántos ciclos ocurren por segundo. La **longitud de onda** ($\lambda$), medida en metros, es la distancia entre puntos equivalentes de ciclos consecutivos. La rapidez de propagación ($v$), la frecuencia y la longitud de onda se relacionan mediante:

$$v = f\lambda.$$

En el vacío:

$$c = f\lambda.$$

Por ejemplo, una onda de ($1\,\mathrm{GHz} = 10^9\,\mathrm{Hz}$) en el vacío tiene una longitud de onda aproximada de ($0{,}30\,\mathrm{m}$). A una rapidez fija, aumentar la frecuencia disminuye la longitud de onda. En materiales, la rapidez depende de sus propiedades electromagnéticas y, en medios dispersivos, también puede depender de la frecuencia.

Estas relaciones son importantes para el diseño de metamateriales, porque el tamaño de sus elementos y la separación entre ellos, comparados con la longitud de onda de operación, influyen en la interacción con la onda. Si las dimensiones de la estructura son mucho menores que la longitud de onda, a menudo puede describirse mediante parámetros efectivos; cuando las dimensiones no son suficientemente pequeñas, pueden aparecer efectos de difracción y respuesta espacial que exigen un análisis más detallado.

#### 1.4. Materiales y respuesta electromagnética

La respuesta de un material a los campos electromagnéticos se caracteriza, en un modelo macroscópico, mediante propiedades como la permitividad eléctrica ($\varepsilon$), la permeabilidad magnética ($\mu$) y la conductividad eléctrica ($\sigma$). La permitividad relaciona el campo eléctrico con la respuesta eléctrica del material; la permeabilidad caracteriza su respuesta magnética; y la conductividad describe la facilidad con que se establecen corrientes bajo un campo eléctrico.

En medios lineales e isotrópicos sencillos se emplean las relaciones constitutivas:

$$\vec{D} = \varepsilon\vec{E}, \qquad \vec{B} = \mu\vec{H}, \qquad \vec{J} = \sigma\vec{E},$$

donde ($\vec{D}$) es el desplazamiento eléctrico y ($\vec{H}$) el campo magnético auxiliar. Estas relaciones son aproximaciones: en materiales reales, las propiedades pueden depender de la frecuencia, la dirección, la intensidad del campo y la historia temporal. En simulación, la elección del modelo material debe ser coherente con el rango de frecuencias estudiado.

Cuando una onda encuentra una interfaz entre materiales con distintas propiedades electromagnéticas, parte de su energía puede reflejarse y parte transmitirse. También puede haber absorción, dispersión, interferencia y resonancias. La magnitud de cada efecto depende de la frecuencia, la geometría, las propiedades del material y las condiciones de incidencia.

#### 1.5. Metamateriales electromagnéticos

Un **metamaterial electromagnético** es una estructura diseñada cuya respuesta a las ondas electromagnéticas depende no solo de la composición química de sus materiales, sino también de su arquitectura, geometría y organización espacial. Frecuentemente está formado por elementos repetidos llamados celdas unitarias, cuyas dimensiones y separaciones pueden ajustarse para obtener una respuesta específica.

A diferencia de un material homogéneo convencional, una estructura metamaterial puede exhibir respuestas resonantes diseñadas, permitiendo controlar de manera selectiva la reflexión, la transmisión, la fase, la polarización o la dirección de propagación en determinadas bandas de frecuencia. En ciertas condiciones, la respuesta colectiva puede describirse mediante parámetros efectivos como una permitividad ($\varepsilon_{\mathrm{eff}}$) y una permeabilidad ($\mu_{\mathrm{eff}}$). Sin embargo, esa aproximación debe justificarse según la escala de la celda, la longitud de onda y la dispersión del medio.

Los trabajos de Veselago sobre medios con permitividad y permeabilidad negativas, y las demostraciones experimentales de estructuras artificiales resonantes, contribuyeron al desarrollo moderno del área. Estos resultados no implican que todos los metamateriales presenten índices negativos: el comportamiento depende de la geometría, los materiales constituyentes y la frecuencia.

Para el presente proyecto, se considera una estructura periódica con parámetros geométricos ajustables, por ejemplo: tamaño de la celda ($a$), separación entre elementos ($d$) y ancho de una sección ($w$). La frecuencia de operación ($f$) constituye una variable de análisis. La variación sistemática de estos parámetros permite estudiar qué configuraciones favorecen la transmisión o producen una respuesta resonante en una banda deseada.

#### 1.6. Simulación electromagnética mediante FDTD

La simulación computacional permite aproximar la solución de las ecuaciones de Maxwell para una geometría y unas condiciones de excitación determinadas. Un método especialmente utilizado es **FDTD** (*Finite-Difference Time-Domain*, método de diferencias finitas en el dominio del tiempo), introducido de forma clásica por Kane Yee en 1966.

FDTD discretiza el espacio en una malla y el tiempo en pasos pequeños. Las derivadas espaciales y temporales de las ecuaciones de Maxwell se aproximan mediante diferencias finitas, y los campos se actualizan sucesivamente. En la disposición de Yee, las componentes eléctricas y magnéticas se ubican en posiciones espaciales alternadas y se actualizan en instantes temporales escalonados, lo que permite resolver de forma eficiente la evolución de la onda.

Un flujo de simulación típico incluye:

1. Definir el dominio computacional, la malla espacial y el paso temporal.
2. Especificar la geometría de la estructura y las propiedades electromagnéticas de cada región.
3. Definir la fuente de excitación, por ejemplo, un pulso o una señal de frecuencia conocida.
4. Establecer condiciones de frontera que representen adecuadamente el problema. Las fronteras absorbentes, como las capas perfectamente acopladas (*PML*), reducen las reflexiones artificiales; las fronteras periódicas son útiles cuando se modela una celda repetitiva bajo condiciones compatibles.
5. Actualizar los campos eléctricos y magnéticos durante el tiempo de simulación.
6. Registrar los campos y calcular magnitudes de interés, como la intensidad, la reflexión, la transmisión y la respuesta en frecuencia.

La resolución espacial debe ser suficiente para representar la longitud de onda más corta relevante y los detalles geométricos más pequeños. El paso temporal debe cumplir la condición de estabilidad del esquema numérico, como la condición de Courant-Friedrichs-Lewy (CFL) en FDTD explícito. Además, se deben realizar pruebas de convergencia al refinar la malla y verificar que las fronteras y el tiempo de simulación no alteren artificialmente los resultados.

Para validar el simulador, conviene comenzar con casos conocidos: propagación en el vacío, incidencia sobre un material convencional y propagación o dispersión en una estructura periódica sencilla. La comparación con soluciones analíticas cuando existan, o con resultados publicados y simuladores de referencia, ayuda a identificar errores de implementación.

#### 1.7. Transmisión, reflexión y resonancia

Para evaluar el comportamiento de una estructura, se pueden comparar las ondas incidentes, reflejadas y transmitidas. En condiciones adecuadas, los coeficientes de reflexión ($R$) y transmisión ($T$) se definen como cocientes de potencia reflejada y transmitida respecto a la potencia incidente. En ese caso, para un sistema pasivo, la fracción absorbida ($A$) cumple:

$$A = 1 - R - T,$$

si ($R$), ($T$) y ($A$) representan fracciones de potencia y se contabilizan todos los canales relevantes. En situaciones con varios modos, difracción o diferentes medios de entrada y salida, la definición debe adaptarse para asegurar una comparación correcta.

La transmisión puede representarse en función de la frecuencia mediante una curva ($T(f)$). Los máximos o mínimos de esa curva pueden indicar bandas de transmisión, rechazo o resonancias, aunque su interpretación requiere distinguir resonancia, pérdidas, interferencia y acoplamiento con los modos de propagación. La visualización de mapas de campo también permite localizar zonas de concentración de energía y comprender los mecanismos físicos detrás de una respuesta espectral.

#### 1.8. Optimización computacional de la geometría

La optimización consiste en buscar, dentro de un conjunto de diseños posibles, la geometría que mejor satisface un objetivo cuantificable. Para este proyecto, una solución candidata se describe mediante parámetros como:

$$\mathbf{p} = (a, d, w),$$

donde ($a$) es el tamaño de celda, ($d$) la separación y ($w$) el ancho de un elemento. El simulador calcula la respuesta electromagnética de cada candidato y permite evaluar una función objetivo.

Si el propósito es maximizar la transmisión a una frecuencia objetivo ($f_0$), una función sencilla es:

$$F(\mathbf{p}) = T(f_0; \mathbf{p}).$$

En esta expresión, ($T(f_0; \mathbf{p})$) es la transmisión calculada a la frecuencia objetivo para la geometría ($\mathbf{p}$). El algoritmo busca valores de ($\mathbf{p}$) que hagan máxima la función. Si se desea una banda de transmisión, un rechazo o limitar la reflexión, puede definirse una función más completa que combine varios términos y penalizaciones. La función debe reflejar de manera explícita el comportamiento físico deseado.

Una estrategia de optimización es el **algoritmo genético**, que trabaja con una población de diseños candidatos. En términos generales, se generan diseños iniciales, se evalúa su función objetivo, se seleccionan candidatos prometedores y se producen nuevas soluciones mediante cruce y mutación. El proceso se repite durante varias generaciones o hasta cumplir un criterio de parada. Los algoritmos genéticos son útiles cuando el espacio de búsqueda es complejo o no se dispone de gradientes sencillos, pero no garantizan encontrar el óptimo global y pueden requerir numerosas simulaciones.

El flujo integrado del proyecto sería: generar una geometría, ejecutar la simulación FDTD, extraer la respuesta espectral, calcular la función objetivo, actualizar los candidatos y guardar el mejor diseño. Una interfaz puede mostrar la frecuencia objetivo, los parámetros geométricos, la generación actual, la transmisión obtenida y la evolución del mejor resultado.

#### 1.9. Consideraciones de evaluación y alcance

La calidad del diseño no debe juzgarse únicamente por el valor máximo de la función objetivo. También conviene comprobar la estabilidad numérica, la convergencia respecto a la malla, la sensibilidad a pequeñas variaciones geométricas, las pérdidas, la banda de operación y la reproducibilidad de los resultados. Si el coste computacional lo permite, el mejor candidato debe volver a simularse con una malla más fina o condiciones de validación independientes.

El proyecto plantea una metodología computacional: construir y validar un simulador, estudiar la influencia de los parámetros geométricos y automatizar la búsqueda de diseños. Por tanto, los resultados dependerán de las hipótesis adoptadas, del modelo material, de las condiciones de frontera, de la resolución numérica y de la función objetivo seleccionada. La optimización computacional identifica diseños prometedores dentro de ese modelo; no sustituye por sí sola la fabricación ni la validación experimental de un metamaterial físico.

#### 1.10. Referencias bibliográficas

1. Jackson, J. D. (1999). *Classical Electrodynamics* (3.ª ed.). John Wiley & Sons. Texto de referencia para campos eléctricos y magnéticos, ecuaciones de Maxwell y ondas electromagnéticas.
2. Griffiths, D. J. (2017). *Introduction to Electrodynamics* (4.ª ed.). Cambridge University Press. https://doi.org/10.1017/9781108333511
3. Veselago, V. G. (1968). The electrodynamics of substances with simultaneously negative values of $\varepsilon$ and $\mu$. *Soviet Physics Uspekhi, 10*, 509–514. https://doi.org/10.1070/PU1968v010n04ABEH003699
4. Smith, D. R., Padilla, W. J., Vier, D. C., Nemat-Nasser, S. C., & Schultz, S. (2000). Composite medium with simultaneously negative permeability and permittivity. *Physical Review Letters, 84*(18), 4184–4187. https://doi.org/10.1103/PhysRevLett.84.4184
5. Pendry, J. B. (2000). Negative refraction makes a perfect lens. *Physical Review Letters, 85*(18), 3966–3969. https://doi.org/10.1103/PhysRevLett.85.3966
6. Yee, K. S. (1966). Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media. *IEEE Transactions on Antennas and Propagation, 14*(3), 302–307. https://doi.org/10.1109/TAP.1966.1138693
7. Taflove, A., & Hagness, S. C. (2005). *Computational Electrodynamics: The Finite-Difference Time-Domain Method* (3.ª ed.). Artech House. ISBN 978-1-58053-832-9.
8. Schneider, J. B. (2010). *Understanding the Finite-Difference Time-Domain Method*. Washington State University. https://www.eecs.wsu.edu/~schneidj/ufdtd/
9. Holland, R. (1993). Pitfalls of time domain modeling of electromagnetic scattering. *IEEE Transactions on Antennas and Propagation, 41*(2), 169–173. https://doi.org/10.1109/8.210124
10. Balanis, C. A. (2012). *Advanced Engineering Electromagnetics* (2.ª ed.). John Wiley & Sons.

---

### 2. Metodología Aplicada y Código en Funcionamiento

#### 2.1. Adecuación del método frente al objetivo planteado
El método implementado combina una **arquitectura de software guiada por el dominio físico** (*Domain-Driven Design* aplicado a la física computacional) con un **desarrollo incremental por niveles**. Esta estrategia resulta idónea para alcanzar el objetivo del proyecto por tres razones fundamentales:

1. **Trazabilidad y ausencia de "cajas negras":** Cada entidad matemática y física ($\vec{r}$, $q$, $\vec{F}$, $\vec{E}$) está encapsulada en una capa de dominio pura en Python, desacoplada de la interfaz gráfica. Esto garantiza que la discretización espacial sobre grillas 2D/3D y el principio de superposición se verifiquen analítica y computacionalmente antes de acoplar el bucle temporal de las ecuaciones de Maxwell (FDTD).
2. **Eficiencia numérica y reactividad en tiempo real:** El uso de operaciones vectorizadas con **NumPy** en el backend junto con canales **WebSockets** asíncronos en **FastAPI** permite calcular mallas de campo y actualizar estados físicos en milisegundos, requisito indispensable para ejecutar cientos de simulaciones sucesivas durante la fase de optimización genética.
3. **Escalabilidad progresiva de la malla espacial:** La infraestructura de grillas de muestreo espacial desarrollada en el Nivel 2 (campos vectoriales $\vec{E}(\vec{r})$) constituye la base geométrica directa sobre la cual se instancia la celda escalonada de Yee para el solucionador FDTD en el Nivel 3.

#### 2.2. Arquitectura y código actualmente en funcionamiento
El sistema se encuentra operativo bajo una arquitectura cliente-servidor modular dividida en dos bloques principales:

* **Backend Físico y Numérico (`backend/`):**
  * **Capa de Dominio Físico Puro (`backend/domain/`):**
    * `vectors.py`: Implementación completa de álgebra vectorial en 2D y 3D (`Vector2D`, `Vector3D`), incluyendo cálculo de normas euclidianas, normalización a vectores unitarios, productos escalares y sobrecarga de operadores aritméticos.
    * `units.py`: Centralización de constantes universales en el Sistema Internacional ($\varepsilon_0$, $\mu_0$, $c$, $k_e$) y factores de conversión controlada a sub-escalas espaciales.
    * `particles.py`: Modelado formal de fuentes puntuales con posición vectorial y magnitud de carga en Coulombs ($\mathrm{C}$).
    * `coulomb.py`: Cómputo vectorial exacto de la Ley de Coulomb ($\vec{F} = k \frac{q_1 q_2}{r^2}\hat{r}$), evaluación de interacciones atractivas/repulsivas por pares, prevención de singularidades numéricas cuando $r \to 0$ y superposición de fuerzas netas.
    * `electric_field.py`: Evaluación del campo eléctrico continuo $\vec{E}(\vec{r}) = \sum \vec{E}_i(\vec{r})$ en coordenadas arbitrarias y generación vectorizada de mallas espaciales 2D y 3D.
  * **Capa de Servicios y API (`backend/api/`):**
    * **Endpoints REST (`/api/physics/*`):** Exposición estructurada y validada con **Pydantic v2** para consulta de constantes, cálculo de fuerzas interpartícula y generación de grillas vectoriales.
    * **Canal WebSocket (`/ws/physics`):** Comunicación bidireccional en tiempo real que recalcula instantáneamente el estado del campo al desplazar fuentes o modificar parámetros desde el navegador.
* **Frontend Interactivo de Visualización Científica (`frontend/`):**
  * **Sistema de Tipos Espejo (`frontend/src/types/`):** Contratos en **TypeScript** sincronizados con los modelos de Pydantic del backend (`Vector2D`, `Vector3D`, `Particle`, `FieldPoint`).
  * **Módulos Experimentales Interactivos (`frontend/src/pages/chapters/`):**
    * `Chapter1_Charges.tsx` (*Capítulo 1 — Cargas Eléctricas*): Cuantización de la carga, polaridades e interacciones elementales.
    * `Chapter2_Coulomb.tsx` (*Capítulo 2 — Ley de Coulomb 2D*): Laboratorio interactivo en el plano para variar distancias y magnitudes de carga verificando la ley del inverso del cuadrado.
    * `Chapter3_Forces3D.tsx` (*Capítulo 3 — Fuerzas en 3D*): Escena espacial tridimensional renderizada con **Three.js / React Three Fiber** con control de cámara orbital y representación de vectores de fuerza en 3D.
    * `Chapter4_ElectricField.tsx` (*Capítulo 4 — Campo Eléctrico*): Muestreo espacial continuo sobre una rejilla bidimensional, visualización de vectores de campo $\vec{E}(x,y)$ y superposición multicomponente.

#### 2.3. Stack tecnológico implementado

| Capa / Área | Tecnología | Propósito en el proyecto |
| :--- | :--- | :--- |
| **Motor Físico (Backend)** | **Python 3.12+** | Modelado matemático de leyes electromagnéticas y estructuras de datos físicas. |
| **API y Tiempo Real** | **FastAPI + Uvicorn** | Servidor asíncrono de alto rendimiento para endpoints REST y streaming por WebSockets. |
| **Cálculo Científico** | **NumPy** | Operaciones matriciales y evaluación vectorizada sobre grillas espaciales de simulación. |
| **Contratos de Datos** | **Pydantic v2** | Validación estricta de esquemas, unidades SI y serialización entre cliente y servidor. |
| **Pruebas Automatizadas** | **Pytest** | Suite de pruebas unitarias para verificar simetría vectorial, ley $1/r^2$ y manejo de singularidades. |
| **Interfaz de Usuario** | **React 19 + TypeScript + Vite** | Aplicación web modular con tipado estático estricto y actualización reactiva de estados. |
| **Visualización 3D** | **Three.js / @react-three/fiber** | Renderizado acelerado por hardware (WebGL) de partículas, grillas y campos vectoriales 3D. |
| **Gráficas y Notación** | **Recharts + KaTeX** | Trazado de curvas de respuesta física (como $T(f)$) y renderizado tipográfico de ecuaciones de Maxwell. |

---

### 3. Resultados Preliminares y Validación de Sentido Físico

Actualmente se encuentran completados y verificados los **Niveles 1 (Electrostática Discreta)** y **Nivel 2 (Campos Vectoriales y Grillas Espaciales)**. Los resultados preliminares obtenidos del código en ejecución demuestran consistencia matemática y **pleno sentido físico** bajo cinco criterios cuantitativos verificados con `pytest` y en la visualización interactiva:

1. **Cumplimiento exacto de la dependencia radial ($1/r^2$):** Al duplicar la separación espacial $r$ entre dos cargas puntuales en los módulos de los Capítulos 2 y 3, la magnitud de la fuerza eléctrica $\|\vec{F}\|$ y de la intensidad de campo $\|\vec{E}\|$ se reduce exactamente a la cuarta parte ($0{,}25\,\|\vec{E}_0\|$), reproduciendo el decaimiento teórico en el vacío con constante $k_e = \frac{1}{4\pi\varepsilon_0} \approx 8{,}98755 \times 10^9\,\mathrm{N\cdot m^2/C^2}$.
2. **Simetría direccional y Tercera Ley de Newton ($\vec{F}_{12} = -\vec{F}_{21}$):** Las pruebas unitarias confirman que para cualquier par de cargas $(q_1, q_2)$ en el espacio 2D o 3D, los vectores de fuerza calculados presentan igual magnitud, dirección colineal y sentidos exactamente opuestos, cambiando correctamente entre regímenes de repulsión ($q_1 q_2 > 0$) y atracción ($q_1 q_2 < 0$).
3. **Principio de superposición lineal sobre la rejilla ($\vec{E}_{\text{total}} = \sum \vec{E}_i$):** En la grilla espacial del Capítulo 4 (`electric_field.py`), la configuración de un dipolo eléctrico (dos cargas de igual magnitud y signo opuesto, $+q$ y $-q$) produce un campo vectorial divergente desde la fuente positiva y convergente hacia el sumidero negativo, mientras que en el punto medio de dos cargas idénticas ($+q, +q$) el campo neto se anula ($\|\vec{E}\| = 0\,\mathrm{V/m}$) tal como predice la teoría clásica.
4. **Estabilidad numérica ante singularidades ($r \to 0$):** El motor físico incorpora un umbral de regularización espacial cuando la coordenada de muestreo de la rejilla coincide con la posición exacta de una fuente puntual, evitando divisiones por cero (`NaN` / `Inf`) sin distorsionar el campo en las celdas adyacentes.
5. **Desempeño computacional de la grilla en tiempo real:** La evaluación de mallas vectoriales de campo mediante el canal WebSocket `/ws/physics` mantiene tiempos de respuesta inferiores a la tasa de refresco visual al arrastrar partículas interactivamente, validando que la arquitectura soporta el cómputo reiterado de grillas 2D requerido para el siguiente nivel.

---

### 4. Trabajo Pendiente para Completar el Proyecto

Para culminar satisfactoriamente el proyecto a partir de la plataforma física e interactiva ya construida, restan tres etapas de desarrollo claramente acotadas:

1. **Nivel 3 — Solucionador Electrodinámico FDTD 2D y Validación (Fase 2):**
   * Extender el módulo de grillas espaciales (`backend/domain/`) para implementar la celda escalonada de Yee en 2D (modo TM: $E_z, H_x, H_y$) resolviendo las ecuaciones rotacionales de Faraday y Ampère-Maxwell en el dominio del tiempo bajo la condición de estabilidad de Courant-Friedrichs-Lewy (CFL).
   * Incorporar la fuente de excitación de pulso gaussiano de banda ancha, las condiciones de frontera absorbentes (PML) en los extremos de propagación y las fronteras periódicas en la dirección transversal.
   * Implementar el cálculo de la Transformada Rápida de Fourier (FFT) sobre los monitores de campo incidente, reflejado y transmitido para obtener las curvas espectrales $R(f)$ y $T(f)$, completando la validación en los tres casos de referencia (vacío, lámina dieléctrica convencional y arreglo periódico con verificación de $R + T = 1$).
2. **Análisis Paramétrico de la Celda Unitaria y Función Objetivo (Fase 3):**
   * Parametrizar la geometría de la celda periódica en función del tamaño de celda $(a)$, la separación $(d)$ y el ancho del elemento $(w)$ alrededor de la banda de microondas ($5\text{–}15\text{ GHz}$, con frecuencia objetivo $f_0 = 10\text{ GHz}$).
   * Integrar en el frontend (`Recharts`) las gráficas espectrales de transmisión $T(f)$ y formalizar en el backend el evaluador de la función objetivo $F(\mathbf{p}) = T(f_0; \mathbf{p})$ junto con su variante multiobjetivo ponderada.
3. **Nivel 4 — Optimización Computacional Automática (Fase 4):**
   * Desarrollar el módulo generador de estructuras candidatas y ejecutar el barrido sistemático sobre el espacio discreto $(a, d, w)$ como línea base.
   * Implementar el motor de **algoritmo genético** (inicialización poblacional, evaluación FDTD de aptitud, selección por torneo/elitismo, cruce paramétrico y mutación) y conectar su progreso iterativo (generación actual, evolución de $F(\mathbf{p})$ y visualización de la geometría óptima encontrada) en la interfaz interactiva final.
