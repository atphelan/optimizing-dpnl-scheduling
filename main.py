from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dpnl_optimizer_tool import find_s_optimum_from_counts, find_s_optimum_from_phases

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/optimize/counts")
def optimize_counts(dt: int, edu: float, dp: float, brdu: float):
    result = find_s_optimum_from_counts(dt, edu, dp, brdu)
    return {"Optimal t_wait": int(result[0]), "SNR k2 from simulation": float(result[1])}

@app.get("/optimize/phases")
def optimize_phases(tg1: float, ts: float, tp: float):
    result = find_s_optimum_from_phases(tg1, ts, tp)
    return {"Optimal t_wait": int(result[0]), "SNR k2 from simulation": float(result[1])}
