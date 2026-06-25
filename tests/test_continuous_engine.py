import pytest

from app.continuous_engine import ContinuousBatchingEngine
from app.request import GenerateRequest


def make_request(i: int, max_tokens: int = 2) -> GenerateRequest:
    return GenerateRequest(
        request_id=f"req-{i}",
        prompt="hello model",
        max_tokens=max_tokens,
    )


@pytest.mark.asyncio
async def test_submit_without_start_raises_error():
    engine = ContinuousBatchingEngine()

    with pytest.raises(RuntimeError):
        await engine.submit(make_request(1))
    
@pytest.mark.asyncio
async def test_submit_one_request():
    engine = ContinuousBatchingEngine(max_running_requests=2, decode_step_time=0.001)
    await engine.start()

    result = await engine.submit(make_request(1, max_tokens=2))

    await engine.stop()

    assert result.request_id == "req-1"
    assert result.generated_tokens == 2
    assert result.latency_ms > 0

import asyncio


@pytest.mark.asyncio
async def test_submit_multiple_requests():
    engine = ContinuousBatchingEngine(max_running_requests=2, decode_step_time=0.001)
    await engine.start()

    tasks = [
        asyncio.create_task(engine.submit(make_request(i, max_tokens=2)))
        for i in range(5)
    ]

    results = await asyncio.gather(*tasks)

    await engine.stop()

    assert len(results) == 5
    assert {r.request_id for r in results} == {f"req-{i}" for i in range(5)}