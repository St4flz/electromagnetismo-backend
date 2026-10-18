# 🔋 Plan de Desarrollo — De Partículas a Campo Eléctrico

> **Narrativa:** Contamos una historia que empieza con *"¿Qué es una carga?"* y termina con *"¿Qué es un campo eléctrico?"*, donde cada concepto se construye sobre el anterior y cada línea de código explica la física.

---

## 🧱 Paso 0 — Abstracciones Computacionales

Antes de escribir una sola página, necesitamos decidir **cómo representamos la física en código**. Estas abstracciones son el lenguaje común entre frontend y backend.

### El principio: **un tipo por cada concepto físico**

No usamos `float` sueltos ni `dict` genéricos. Cada concepto físico tiene su propia representación con nombre, unidades y operaciones.

### Backend (Python)

```
backend/
├── domain/                        # La física vive aquí
│   ├── __init__.py
│   ├── vectors.py                 # Vector2D, Vector3D — posición, fuerza, campo
│   ├── units.py                   # Sistema de unidades SI, conversiones
│   ├── particles.py               # Particle — carga puntual con posición y q
│   ├── coulomb.py                 # Ley de Coulomb: F = k·q₁·q₂/r²·r̂
│   └── electric_field.py          # Campo eléctrico: E = F/q₀ = k·q/r²·r̂
│
├── api/
│   ├── __init__.py
│   ├── routers/
│   │   └── physics.py             # Endpoints REST
│   ├── schemas/
│   │   └── physics.py             # Pydantic models (contratos API)
│   └── websockets/
│       └── realtime_physics.py    # WS para cálculos en tiempo real
│
├── main.py
└── requirements.txt
```

#### `domain/vectors.py` — El bloque fundamental

```python
"""Representación de vectores en el espacio.

Un vector es la abstracción más básica en física:
describe una magnitud con dirección. Lo usamos para
posición, fuerza, campo eléctrico, velocidad, etc.

Internamente todo está en SI (metros, Newtons, V/m).
"""
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Vector2D:
    """Vector en el plano (x, y).
    
    Ejemplo: la posición de una carga en 2D
        r = Vector2D(x=0.03, y=0.05)  # 3 cm, 5 cm
    """
    x: float
    y: float

    def __add__(self, other: "Vector2D") -> "Vector2D": ...
    def __sub__(self, other: "Vector2D") -> "Vector2D": ...
    def __mul__(self, scalar: float) -> "Vector2D": ...

    @property
    def magnitude(self) -> float:
        """||v|| = √(x² + y²)"""
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def unit(self) -> "Vector2D":
        """v̂ = v / ||v|| — vector unitario (dirección sin magnitud)"""
        mag = self.magnitude
        if mag == 0:
            raise ValueError("El vector cero no tiene dirección")
        return Vector2D(self.x / mag, self.y / mag)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}
```

#### `domain/particles.py` — Carga puntual

```python
"""Una partícula con carga eléctrica.

En electrostática, todo empieza con cargas puntuales:
objetos con carga q ubicados en una posición r.

La carga q se mide en Coulombs [C].
  - Protón:   q = +1.6 × 10⁻¹⁹ C
  - Electrón: q = -1.6 × 10⁻¹⁹ C

Para la simulación usamos valores más grandes (microcoulombs)
para que las fuerzas sean visibles a escala humana.
"""

@dataclass
class Particle:
    """Carga puntual en el espacio.
    
    Atributos:
        position: dónde está la carga [m]
        charge:   cuánta carga tiene [C] (+ atrae -, - repele -)
        label:    nombre para la UI ("q₁", "q₂", etc.)
    """
    position: Vector2D    # En metros
    charge: float         # En Coulombs (+ o -)
    label: str = ""
```

#### `domain/coulomb.py` — La primera ley

```python
"""Ley de Coulomb — la fuerza entre dos cargas.

         k · q₁ · q₂
  F₁₂ = ――――――――――――― · r̂₁₂
              r²

Donde:
  k  = 8.9875 × 10⁹ N·m²/C²  (constante de Coulomb)
  q₁, q₂ = cargas [C]
  r  = distancia entre cargas [m]
  r̂₁₂ = vector unitario de q₁ hacia q₂

Signo:
  - Cargas del mismo signo → F repulsiva (r̂ apunta hacia afuera)
  - Cargas de signo opuesto → F atractiva (r̂ apunta hacia adentro)
"""

# k = 1/(4πε₀) — la constante de Coulomb
COULOMB_CONSTANT = 8.9875517873681764e9  # [N·m²/C²]

def coulomb_force(p1: Particle, p2: Particle) -> Vector2D:
    """Calcula la fuerza que p2 ejerce sobre p1.
    
    Retorna el vector de fuerza F₁₂ en Newtons.
    
    Implementa exactamente:
        F₁₂ = k · q₁ · q₂ / |r₁₂|² · r̂₁₂
    
    donde r₁₂ = r₂ - r₁ (vector de p1 hacia p2)
    """
    # r₁₂ = r₂ - r₁ — vector de p1 hacia p2
    r12 = p2.position - p1.position
    
    # |r₁₂| — distancia entre las cargas
    distance = r12.magnitude
    
    if distance < 1e-10:
        raise ValueError("Las cargas están demasiado cerca (singularidad)")
    
    # |F| = k · |q₁| · |q₂| / r²
    force_magnitude = COULOMB_CONSTANT * p1.charge * p2.charge / (distance ** 2)
    
    # F₁₂ = |F| · r̂₁₂
    # El signo ya está incluido en q₁·q₂:
    #   mismo signo → F positiva → apunta hacia afuera (repulsión)
    #   signo opuesto → F negativa → apunta hacia adentro (atracción)
    return r12.unit * force_magnitude
```

### Frontend (TypeScript) — Espejo de la física

```
frontend/src/
├── types/
│   ├── physics.ts          # Vector2D, Vector3D, Particle, Force
│   └── api.ts              # Tipos de request/response
│
├── services/
│   ├── api.ts              # Cliente base
│   └── physicsSocket.ts    # WebSocket para cálculos en tiempo real
```

#### `types/physics.ts`

```typescript
/** Vector en 2D — la misma abstracción que en el backend.
 *  Unidades: metros (para posición), Newtons (para fuerza), V/m (para campo). */
export interface Vector2D {
  x: number;
  y: number;
}

/** Carga puntual en el espacio 2D.
 *  charge en Coulombs (positivo o negativo). */
export interface Particle {
  id: string;
  position: Vector2D;  // [m]
  charge: number;      // [C]
  label: string;       // "q₁", "q₂"
}

/** Resultado de un cálculo de fuerza de Coulomb.
 *  Incluye la fuerza resultante Y los datos intermedios para
 *  que el frontend pueda mostrar cada parte de la ecuación. */
export interface CoulombResult {
  force: Vector2D;           // F₁₂ [N] — el vector de fuerza
  forceMagnitude: number;    // |F₁₂| [N]
  distance: number;          // r [m] — distancia entre cargas
  unitVector: Vector2D;      // r̂₁₂ — dirección
}

/** Estado completo de la escena física.
 *  Esto es lo que viaja por el WebSocket. */
export interface PhysicsState {
  particles: Particle[];
  forces: Record<string, CoulombResult>;  // keyed by "p1_id→p2_id"
  timestamp: number;
}
```

### Sistema de Unidades y Coordenadas

```
┌─────────────────────────────────────────────────────┐
│  MUNDO FÍSICO (backend)     MUNDO VISUAL (frontend) │
│                                                     │
│  Posición: metros [m]  ←→  Posición: píxeles [px]   │
│  Carga: Coulombs [C]  ←→  Radio: proporcional a |q| │
│  Fuerza: Newtons [N]  ←→  Flecha: largo ∝ |F|      │
│  Campo: V/m           ←→  Color + flechas           │
│                                                     │
│  Conversión:                                        │
│    1 metro = SCALE píxeles (ej: 1m = 200px)         │
│    El usuario trabaja en cm/mm en la UI              │
│    El backend siempre calcula en SI                  │
└─────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Regla de oro:** El backend SIEMPRE trabaja en SI (metros, Coulombs, Newtons). El frontend convierte a unidades de display (cm, µC, mN) solo para mostrar. La conversión ocurre en una capa de utilidades del frontend, nunca en el backend.

---

## 📖 Narrativa — Los 4 Capítulos

### Capítulo 1: Partículas y Cargas (2D)

> *"Todo empieza con una propiedad fundamental de la materia: la carga eléctrica."*

**¿Qué enseña?**
- Qué es una carga eléctrica (positiva/negativa)
- Que la carga es una propiedad escalar (un número con signo)
- Que las partículas existen en un espacio con coordenadas (x, y)

**¿Qué muestra la UI?**
- Un canvas 2D limpio con un sistema de coordenadas visible
- Partículas que el usuario puede crear (click) y arrastrar
- Cargas positivas: esferas con gradiente rojo-naranja y símbolo "+"
- Cargas negativas: esferas con gradiente azul-cyan y símbolo "−"
- El radio visual de cada partícula es proporcional a |q|
- Panel lateral con las propiedades de cada carga (posición, valor de q)
- Tooltip al hover: "q₁ = +3.0 µC en (2.5 cm, 4.0 cm)"

**Backend involucrado:**
- Mínimo: solo valida y almacena el estado de las partículas
- Endpoint: `POST /physics/particles` → guarda configuración
- El WebSocket aún no se necesita aquí (no hay cálculo dinámico)

**Componentes clave:**
```
SimulationCanvas2D.tsx    — el canvas donde viven las partículas
ParticleRenderer.tsx      — dibuja una partícula (+ o -)
CoordinateGrid.tsx        — ejes x, y con escala
ParticleInspector.tsx     — panel lateral con propiedades
AddParticleControls.tsx   — controles para crear partículas
```

---

### Capítulo 2: Ley de Coulomb (2D)

> *"Cuando dos cargas están cerca, aparece una fuerza entre ellas. La naturaleza de esta fuerza la describe la Ley de Coulomb."*

**¿Qué enseña?**
- La fuerza entre dos cargas: F = k·q₁·q₂/r²·r̂
- Atracción (cargas opuestas) vs repulsión (mismas cargas)
- La fuerza es un VECTOR (tiene magnitud y dirección)
- La fuerza decae con el cuadrado de la distancia (ley del inverso cuadrado)

**¿Qué muestra la UI?**
- Dos (o más) partículas en el canvas
- **Flechas de fuerza** dibujadas sobre cada partícula (vector F)
  - Largo proporcional a |F|
  - Color: mismo que la partícula que EJERCE la fuerza
- Al arrastrar una partícula, las fuerzas se recalculan **en tiempo real** (WebSocket)
- Panel de ecuación: muestra la fórmula con los **valores actuales** sustituidos
  ```
  F₁₂ = (8.99 × 10⁹) × (3.0 × 10⁻⁶) × (-2.0 × 10⁻⁶) / (0.05)²
       = -21.6 N   ← atractiva (signo negativo)
  ```
- Gráfico lateral: |F| vs r (mientras arrastras, ves la curva 1/r²)
- Indicador de dirección: r̂ dibujado como flecha punteada

**Backend involucrado — ¡Aquí es donde cobra vida!**

```
WebSocket /ws/physics/coulomb
  
  → Cliente envía: { particles: [{id, position, charge}, ...] }
  ← Servidor responde: { 
       forces: {
         "q1→q2": { force: {x, y}, magnitude, distance, unit_vector },
         "q2→q1": { force: {x, y}, magnitude, distance, unit_vector }
       }
     }
```

El backend hace el cálculo con `domain/coulomb.py` → el frontend solo dibuja.

**Componentes clave:**
```
ForceArrow.tsx           — dibuja un vector de fuerza sobre una partícula
EquationPanel.tsx        — muestra F = k·q₁·q₂/r² con valores sustituidos
ForceVsDistanceChart.tsx — gráfico |F|(r) en tiempo real
CoulombExplanation.tsx   — panel educativo con la ley paso a paso
```

---

### Capítulo 3: Cargas y Fuerza en 3D

> *"La física no vive en un plano. Extendemos nuestro mundo a tres dimensiones."*

**¿Qué enseña?**
- La misma ley de Coulomb funciona en 3D (solo cambia r de 2D a 3D)
- Vectores en 3D: r = (x, y, z)
- La fuerza sigue siendo F = k·q₁·q₂/r²·r̂, pero ahora r̂ tiene 3 componentes

**¿Qué muestra la UI?**
- Escena Three.js (react-three-fiber) con:
  - Esferas 3D para las cargas (con materiales semi-transparentes y glow)
  - Ejes x, y, z visibles con etiquetas
  - Flechas 3D de fuerza (usando `<arrowHelper>`)
  - Controles orbitales (rotar, zoom, pan)
  - Sombras y luces para dar profundidad
- Panel lateral idéntico al de 2D (mismas ecuaciones, ahora con z)
- Toggle para comparar la vista 2D (proyección) vs la 3D completa

**Backend involucrado:**
- Misma lógica, pero ahora usa `Vector3D` en vez de `Vector2D`
- El WebSocket acepta ambos formatos:
  ```
  → { particles: [{id, position: {x, y, z}, charge}, ...], dimensions: 3 }
  ```

**Componentes clave:**
```
Scene3D.tsx              — escena Three.js completa
Particle3D.tsx           — esfera 3D con material y glow
ForceArrow3D.tsx         — flecha 3D con <arrowHelper>
CoordinateAxes3D.tsx     — ejes x, y, z con labels
OrbitControlsWrapper.tsx — controles de cámara
```

---

### Capítulo 4: Campo Eléctrico

> *"¿Y si en vez de preguntar 'qué fuerza siente esta carga?' preguntamos 'qué le pasaría a CUALQUIER carga que ponga aquí?'  Eso es el campo eléctrico."*

**¿Qué enseña?**
- E = F/q₀ — el campo es la fuerza por unidad de carga
- E = k·q/r²·r̂ — campo creado por una carga puntual
- El campo existe en todo el espacio (no solo donde hay cargas)
- Principio de superposición: E_total = ΣE_i

**¿Qué muestra la UI?**
- **Campo vectorial:** flechas en una grilla mostrando E(x,y) en cada punto
  - Largo ∝ |E|, dirección = dirección de E
  - Color ∝ |E| (colormap: oscuro → claro según intensidad)
- **Líneas de campo:** curvas que nacen en + y terminan en −
- **Mapa de calor:** |E(x,y)| como colormap continuo
- Toggle entre las 3 visualizaciones
- Versión 2D y 3D disponibles

**Backend involucrado:**
```
WebSocket /ws/physics/electric-field

  → Cliente envía: { 
       particles: [...],
       field_query: {
         type: "grid",        // calcula E en una grilla de puntos
         x_range: [-0.1, 0.1],
         y_range: [-0.1, 0.1],
         resolution: 30       // 30×30 puntos
       }
     }
  ← Servidor responde: {
       field: [
         { point: {x, y}, E: {x, y}, magnitude: 1234.5 },
         ...  // 900 puntos
       ]
     }
```

**Componentes clave:**
```
VectorFieldRenderer.tsx    — dibuja flechas en la grilla
FieldLinesRenderer.tsx     — traza líneas de campo numéricamente
HeatmapRenderer.tsx        — mapa de calor de |E|
FieldVisualizationToggle.tsx — switch entre modos de visualización
SuperpositionExplanation.tsx — panel educativo sobre superposición
```

---

## 🗂️ Estructura Completa de Archivos

### Backend

```
backend/
├── main.py                           # FastAPI app + CORS + routers
├── requirements.txt                  # fastapi, uvicorn, numpy, pydantic
│
├── domain/                           # 🧠 FÍSICA PURA (sin HTTP, sin API)
│   ├── __init__.py
│   ├── vectors.py                    # Vector2D, Vector3D
│   ├── units.py                      # Constantes SI, conversiones
│   ├── particles.py                  # Particle (posición + carga)
│   ├── coulomb.py                    # coulomb_force(p1, p2) → Vector
│   └── electric_field.py            # electric_field_at(point, charges) → Vector
│
├── api/
│   ├── __init__.py
│   ├── schemas/
│   │   └── physics.py                # Pydantic: ParticleSchema, ForceResult, etc.
│   ├── routers/
│   │   └── physics.py                # REST endpoints (configuración, estado)
│   └── websockets/
│       └── realtime_physics.py       # WS: cálculos en tiempo real
│
└── tests/
    ├── test_vectors.py               # Operaciones vectoriales correctas
    ├── test_coulomb.py               # F = k·q₁·q₂/r² con valores conocidos
    └── test_electric_field.py        # E = k·q/r², superposición
```

### Frontend

```
frontend/src/
├── main.tsx
├── App.tsx                           # Router principal
├── index.css                         # Design system
│
├── pages/
│   ├── HomePage.tsx                  # Landing: intro al proyecto
│   ├── chapters/
│   │   ├── Chapter1_Charges.tsx      # Partículas y cargas 2D
│   │   ├── Chapter2_Coulomb.tsx      # Ley de Coulomb 2D
│   │   ├── Chapter3_Forces3D.tsx     # Coulomb en 3D
│   │   └── Chapter4_ElectricField.tsx # Campo eléctrico
│   └── ChapterLayout.tsx            # Layout compartido con sidebar + nav
│
├── components/
│   ├── layout/
│   │   ├── Navbar.tsx
│   │   ├── Sidebar.tsx               # Navegación por capítulos
│   │   └── ChapterNavigation.tsx     # Anterior / Siguiente
│   │
│   ├── canvas2d/                     # Todo lo 2D
│   │   ├── SimulationCanvas2D.tsx    # Canvas contenedor
│   │   ├── ParticleRenderer.tsx      # Dibuja partícula (+ / -)
│   │   ├── ForceArrow.tsx            # Vector de fuerza
│   │   ├── VectorFieldRenderer.tsx   # Grilla de vectores E
│   │   ├── FieldLinesRenderer.tsx    # Líneas de campo
│   │   ├── HeatmapRenderer.tsx       # Mapa de calor |E|
│   │   └── CoordinateGrid.tsx        # Ejes + escala
│   │
│   ├── canvas3d/                     # Todo lo 3D (Three.js)
│   │   ├── Scene3D.tsx               # Escena completa r3f
│   │   ├── Particle3D.tsx            # Esfera con shader
│   │   ├── ForceArrow3D.tsx          # Flecha 3D
│   │   ├── CoordinateAxes3D.tsx      # Ejes x,y,z
│   │   └── FieldArrows3D.tsx         # Campo vectorial 3D
│   │
│   ├── controls/
│   │   ├── ParticleControls.tsx      # Crear/editar partículas
│   │   ├── ChargeSlider.tsx          # Slider para q con label "µC"
│   │   └── VisualizationToggle.tsx   # Switch entre modos de viz
│   │
│   ├── panels/
│   │   ├── EquationPanel.tsx         # Ecuación con valores sustituidos
│   │   ├── ParticleInspector.tsx     # Propiedades de partícula seleccionada
│   │   ├── ForceInspector.tsx        # Detalles de F₁₂
│   │   └── PhysicsExplanation.tsx    # Texto educativo del capítulo
│   │
│   └── charts/
│       ├── ForceVsDistanceChart.tsx  # |F| vs r
│       └── FieldIntensityChart.tsx   # |E| vs r
│
├── hooks/
│   ├── usePhysicsSocket.ts           # WebSocket para cálculos en tiempo real
│   ├── useParticles.ts               # Estado de partículas (CRUD, drag)
│   └── useFieldVisualization.ts      # Estado del modo de visualización
│
├── services/
│   ├── api.ts                        # Fetch base
│   └── physicsSocket.ts              # Clase WebSocket con reconexión
│
├── types/
│   ├── physics.ts                    # Vector2D, Particle, CoulombResult, etc.
│   └── api.ts                        # Tipos de mensajes WS
│
└── utils/
    ├── coordinates.ts                # Conversión metros ↔ píxeles
    ├── units.ts                      # Formateo: 0.000003 → "3.0 µC"
    └── colorMaps.ts                  # Mapas de color para campos
```

---

## 🔌 Protocolo WebSocket — Diseño Detallado

Un solo WebSocket maneja toda la física. El tipo de cálculo se indica en el mensaje.

```typescript
// Mensaje del frontend → backend
type PhysicsRequest = {
  type: "coulomb" | "electric_field";
  particles: ParticleData[];
  // Solo para electric_field:
  fieldQuery?: {
    type: "grid";
    xRange: [number, number];
    yRange: [number, number];
    resolution: number;
  };
  dimensions: 2 | 3;
}

// Respuesta del backend → frontend
type PhysicsResponse = {
  type: "coulomb_result" | "electric_field_result";
  
  // Para coulomb:
  forces?: Record<string, {
    force: Vector2D | Vector3D;    // F₁₂ en Newtons
    magnitude: number;              // |F₁₂|
    distance: number;               // r en metros
    unitVector: Vector2D | Vector3D; // r̂₁₂
  }>;
  
  // Para electric_field:
  field?: Array<{
    point: Vector2D | Vector3D;
    E: Vector2D | Vector3D;         // Campo E en V/m
    magnitude: number;               // |E| en V/m
  }>;
  
  computeTimeMs: number;  // Cuánto tardó el cálculo (transparencia)
}
```

**Flujo en tiempo real:**
```
Usuario arrastra q₁ a nueva posición
  → Frontend: throttle a 60fps, envía PhysicsRequest al WS
    → Backend: calcula F = k·q₁·q₂/r²·r̂ con domain/coulomb.py
      → Backend: envía PhysicsResponse
        → Frontend: actualiza flechas de fuerza en el canvas
          → Total: <16ms (1 frame)
```

---

## 📋 Tareas — Orden de Ejecución

### Sprint 0: Infraestructura (1-2 días)

| # | Tarea | Archivos |
|---|-------|----------|
| 0.1 | Setup FastAPI con CORS, estructura de carpetas | `main.py`, `requirements.txt` |
| 0.2 | Instalar react-router-dom, katex, three/r3f | `package.json` |
| 0.3 | Design system: CSS variables, tipografía, dark theme | `index.css` |
| 0.4 | Layout base: Navbar + Sidebar + ChapterLayout | `components/layout/*` |
| 0.5 | Tipos compartidos `physics.ts` en frontend | `types/physics.ts` |
| 0.6 | `domain/vectors.py` + tests | `domain/vectors.py`, `tests/test_vectors.py` |
| 0.7 | `domain/units.py` (constantes k, ε₀, μ₀) | `domain/units.py` |
| 0.8 | Schemas Pydantic base | `api/schemas/physics.py` |

### Sprint 1: Capítulo 1 — Partículas (2-3 días)

| # | Tarea | Archivos |
|---|-------|----------|
| 1.1 | `domain/particles.py` | Backend |
| 1.2 | `SimulationCanvas2D.tsx` — canvas con interacción | Frontend |
| 1.3 | `CoordinateGrid.tsx` — ejes y escala | Frontend |
| 1.4 | `ParticleRenderer.tsx` — esferas + / - | Frontend |
| 1.5 | `useParticles.ts` — crear, mover, eliminar | Frontend |
| 1.6 | `ParticleControls.tsx` + `ChargeSlider.tsx` | Frontend |
| 1.7 | `ParticleInspector.tsx` — panel de propiedades | Frontend |
| 1.8 | `Chapter1_Charges.tsx` — página completa | Frontend |
| 1.9 | `coordinates.ts` + `units.ts` — conversiones | Frontend |
| 1.10 | `HomePage.tsx` — landing page | Frontend |

### Sprint 2: Capítulo 2 — Coulomb (3-4 días)

| # | Tarea | Archivos |
|---|-------|----------|
| 2.1 | `domain/coulomb.py` + tests | Backend |
| 2.2 | WebSocket endpoint + handler | `api/websockets/realtime_physics.py` |
| 2.3 | `usePhysicsSocket.ts` — hook de conexión WS | Frontend |
| 2.4 | `ForceArrow.tsx` — vector de fuerza en canvas | Frontend |
| 2.5 | `EquationPanel.tsx` — ecuación con valores vivos | Frontend |
| 2.6 | `ForceVsDistanceChart.tsx` — gráfico |F| vs r | Frontend |
| 2.7 | `ForceInspector.tsx` — detalle de cada fuerza | Frontend |
| 2.8 | `Chapter2_Coulomb.tsx` — página completa | Frontend |
| 2.9 | Integración: arrastrar partícula → WS → fuerza → flecha | Full stack |

### Sprint 3: Capítulo 3 — 3D (3-4 días)

| # | Tarea | Archivos |
|---|-------|----------|
| 3.1 | `Vector3D` en backend + actualizar coulomb.py | Backend |
| 3.2 | Actualizar WS para soportar `dimensions: 3` | Backend |
| 3.3 | `Scene3D.tsx` — escena r3f con luces y controles | Frontend |
| 3.4 | `Particle3D.tsx` — esfera con material PBR | Frontend |
| 3.5 | `ForceArrow3D.tsx` — flechas 3D | Frontend |
| 3.6 | `CoordinateAxes3D.tsx` — ejes con labels | Frontend |
| 3.7 | `Chapter3_Forces3D.tsx` — página completa | Frontend |

### Sprint 4: Capítulo 4 — Campo Eléctrico (4-5 días)

| # | Tarea | Archivos |
|---|-------|----------|
| 4.1 | `domain/electric_field.py` + tests | Backend |
| 4.2 | Actualizar WS para `type: "electric_field"` | Backend |
| 4.3 | `VectorFieldRenderer.tsx` — flechas en grilla | Frontend |
| 4.4 | `FieldLinesRenderer.tsx` — líneas de campo | Frontend |
| 4.5 | `HeatmapRenderer.tsx` — mapa de calor | Frontend |
| 4.6 | `VisualizationToggle.tsx` — switch de modos | Frontend |
| 4.7 | `FieldIntensityChart.tsx` — |E| vs r | Frontend |
| 4.8 | `Chapter4_ElectricField.tsx` — página completa | Frontend |
| 4.9 | Versión 3D del campo (FieldArrows3D) | Frontend |

---

## 🎨 Identidad Visual

| Elemento | Decisión |
|----------|----------|
| **Fondo** | `#0a0e17` — oscuro tipo espacio |
| **Superficies** | `#111827` con borde `#1e293b` — glassmorphism sutil |
| **Carga +** | Gradiente radial `#f97316` → `#ef4444` (naranja→rojo) |
| **Carga −** | Gradiente radial `#06b6d4` → `#3b82f6` (cyan→azul) |
| **Fuerza (repulsión)** | `#f59e0b` amarillo-naranja |
| **Fuerza (atracción)** | `#8b5cf6` púrpura |
| **Campo E (alto)** | `#fbbf24` dorado brillante |
| **Campo E (bajo)** | `#1e293b` casi invisible |
| **Acento UI** | `#6366f1` indigo |
| **Éxito/válido** | `#10b981` esmeralda |
| **Tipografía UI** | Inter |
| **Tipografía math** | KaTeX |
| **Tipografía datos** | JetBrains Mono |
