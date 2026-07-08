import asyncio
from app.continuous_engine import (
    ContinuousBatchingEngine,
)
from app.request import GenerateRequest


async def run_request(
    engine: ContinuousBatchingEngine,
    request_id: str,
    prompt: str,
) -> None:
    before_actual = engine.total_actual_prefill_tokens
    before_saved = engine.total_saved_prefill_tokens
    result = await engine.submit(
        GenerateRequest(
            request_id=request_id,
            prompt=prompt,
            max_tokens=4,
        )
    )
    actual_delta = (
        engine.total_actual_prefill_tokens - before_actual
    )
    saved_delta = (
        engine.total_saved_prefill_tokens - before_saved
    )
    print({
        "request_id": result.request_id,
        "latency_ms": round(result.latency_ms, 2),
        "actual_prefill_tokens": actual_delta,
        "saved_prefill_tokens": saved_delta,
    })


async def main() -> None:
    engine = ContinuousBatchingEngine(
        max_running_requests=1,
        max_running_tokens=512,
        total_kv_blocks=32,
        block_size_tokens=4,
        decode_step_time=0.005,
        prefill_time_per_token=0.01,
        enable_prefix_cache=True,
        prefix_cache_max_blocks=32,
    )
    await engine.start()

    common_prefix = (
        "you are a helpful coding assistant "
        "please answer carefully "
    )

    await run_request(
        engine,
        "req-1",
        common_prefix + "explain asyncio",
    )
    await run_request(
        engine,
        "req-2",
        common_prefix + "explain future",
    )
    await run_request(
        engine,
        "req-3",
        "this prompt has completely different content",
    )

    print("\nfinal stats:")
    print(engine.stats())

    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
