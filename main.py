"""Servidor FastAPI principal para el proyecto Electromagnetismo.

Proporciona la infraestructura de cómputo para:
1. Ley de Coulomb y dinámica de partículas cargadas (2D / 3D).
2. Cálculo de campos vectoriales electrostáticos y superposición E(x, y, z).
3. Comunicación en tiempo real (WebSocket) para actualización fluida con el frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.physics import router as physics_router
from api.routers.source import router as source_router
from api.websockets.realtime_physics import ws_router

app = FastAPI(
    title="Electromagnetismo Backend API",
    description="Motor de cálculo numérico y físico para electromagnetismo computacional y metamateriales.",
    version="1.0.0",
)

# Configuración de CORS para permitir la interacción fluida con Vite / React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers REST y WebSocket
app.include_router(physics_router, prefix="/api")
app.include_router(source_router, prefix="/api")
app.include_router(ws_router)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "project": "Electromagnetismo",
        "description": "Motor de cálculo físico y matemático riguroso",
        "endpoints": {
            "constants": "/api/physics/constants",
            "coulomb_rest": "/api/physics/coulomb",
            "field_grid_rest": "/api/physics/electric-field/grid",
            "trajectory_rest": "/api/physics/trajectory",
            "realtime_ws": "/ws/physics",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}