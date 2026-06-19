import uuid
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.engine import InferenceEngine
from app.metrics import Metrics
from app.request import GenerateRequest

engine = InferenceEngine()
metrics = Metrics()

@asynccontextmanager
async def lifespan(app : FastAPI) -> AsyncGenerator[None, None]:
    '''
        我们的app 有生命周期，
        在启动时需要初始化引擎： engine.start()
        在停止时需要关闭引擎： engine.stop()
    '''
    await engine.start()
    yield
    await engine.stop()

app = FastAPI(lifespan = lifespan)

class GenerateInput(BaseModel):
    prompt : str = Field(...,min_length=1)
    max_tokens : int = Field(default=64,gt=1,le=1024)

class GenerateOuput(BaseModel):
    request_id : str
    text : str
    latency_ms : float

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "engine": engine.running,
    }

@app.post("/generate",response_model=GenerateOuput)
async def generate(inp : GenerateInput) -> GenerateOuput:
    req_id = str(uuid.uuid4())

    req = GenerateRequest(
        request_id=req_id,
        prompt= inp.prompt,
        max_tokens = inp.max_tokens,
        arrival_time=time.time(),
    )

    result = await engine.submit(req)

    metrics.record_latency(result.latency_ms)

    return GenerateOuput(
        request_id=req_id,
        text=result.text,
        latency_ms=result.latency_ms,
    )

@app.get("/metrics")
async def get_metrics() -> dict:
    snapshot = metrics.snapshot()
    engine_stats = {}
    if hasattr(engine, "stats"):
        engine_stats = engine.stats()
    return {
        **snapshot,
        "engine": engine_stats,
    }