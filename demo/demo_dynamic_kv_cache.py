import asyncio

from app.continuous_engine  import ContinuousBatchingEngine
from app.request import GenerateRequest

async def main() -> None:
    engine = ContinuousBatchingEngine(
        max_running_requests=2,
        max_running_tokens=512,
        total_kv_blocks=8,
        block_size_tokens=4,
        decode_step_time=0.02,
        prefill_time_per_token=0.05,
    )

    await engine.start()

    requests = [
        GenerateRequest(
            request_id="req-short",
            prompt="one two three",
            max_tokens=3,
        ),
        GenerateRequest(
            request_id="req-long",
            prompt="one two three",
            max_tokens=12,
        ),
    ]

    tasks = [
        asyncio.create_task(engine.submit(req))
        for req in requests
    ]

    while not all(task.done() for task in tasks):
        running_state = [
            {
                "request_id" : item.req.request_id,
                "generated_tokens" : item.generated_tokens,
                "kv_blocks" : list(item.kv_block_ids)
            }
            for item in engine.running_requests
        ]

        print({
            "running" : running_state,
            "kv_cache" : engine.kv_allocator.stats(),
        })

        await asyncio.sleep(0.03)

    results = await asyncio.gather(*tasks,return_exceptions=True)

    print("/n results:")
    for result in results:
        print(result)

    print("\nfinal stats:")
    print(engine.stats())

    await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
