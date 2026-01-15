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

@app.get("/compute")
def compute(a: float, b: float, c: float, d: float):
    result = find_s_optimum_from_counts(a, b, c, d)
    return {"result": result}
