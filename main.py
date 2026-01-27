from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dpnl_optimizer_tool import find_custom_optimum_from_counts, find_custom_optimum_from_phases

app = FastAPI(docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/optimize/counts")
def optimize_counts(dt: int, edu: float, dp: float, brdu: float, g1w: float, sw: float):
    result = find_custom_optimum_from_counts(dt, edu, dp, brdu, g1w, sw)
    return {"Optimal t_wait": int(result[0]), "SNR k2 from simulation": float(result[1])}

@app.get("/optimize/phases")
def optimize_phases(tg1: float, ts: float, tc: float, g1w: float, sw: float):
    result = find_custom_optimum_from_phases(tg1, ts, tc, g1w, sw)
    return {"Optimal t_wait": int(result[0]), "SNR k2 from simulation": float(result[1])}
