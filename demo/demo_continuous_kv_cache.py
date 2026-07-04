import asyncio

from app.continuous_engine import ContinuousBatchingEngine
from app.request import GenerateRequest


async def main() -> None:
    engine = ContinuousBatchingEngine(
        max_running_requests=8,
        max_running_tokens=1024,
        total_kv_blocks=8,
        block_size_tokens=16,
        decode_step_time=0.005,
        prefill_time_per_token=0.001,
    )

    await engine.start()

    requests = [
        GenerateRequest(
            request_id=f"req-{i}",
            prompt="hello model",
            max_tokens=32,
        )
        for i in range(6)
    ]

    tasks = [
        asyncio.create_task(engine.submit(req))
        for req in requests
    ]

    while not all(task.done() for task in tasks):
        print(engine.stats())
        await asyncio.sleep(0.02)

    results = await asyncio.gather(*tasks)

    print("\nresults:")
    for result in results:
        print(
            result.request_id,
            f"{result.latency_ms:.2f}ms",
        )

    print("\nfinal stats:")
    print(engine.stats())

    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())