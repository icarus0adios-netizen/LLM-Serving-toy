import asyncio

from app.continuous_engine import (
    ContinuousBatchingEngine,
)
from app.request import GenerateRequest


async def main() -> None:
    engine = ContinuousBatchingEngine(
        max_running_requests=3,
        max_running_tokens=1024,
        total_kv_blocks=5,
        block_size_tokens=4,
        decode_step_time=0.02,
        prefill_time_per_token=0.005,
        max_preemptions_per_request=10,
    )

    await engine.start()

    requests = [
        GenerateRequest(
            request_id=f"req-{i}",
            prompt="one two three four",
            max_tokens=10,
        )
        for i in range(3)
    ]

    tasks = [
        asyncio.create_task(engine.submit(req))
        for req in requests
    ]

    while not all(task.done() for task in tasks):
        running = [
            {
                "id": item.req.request_id,
                "state": item.state.name,
                "generated": item.generated_tokens,
                "blocks": list(item.kv_block_ids),
                "preemptions": item.preemption_count,
            }
            for item in engine.running_requests
        ]

        print({
            "running": running,
            "waiting": engine.waiting_queue.qsize(),
            "kv": engine.kv_allocator.stats(),
            "preemptions": engine.total_preemption_count,
        })

        await asyncio.sleep(0.03)

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    print("\nresults:")
    for result in results:
        print(result)

    print("\nfinal stats:")
    print(engine.stats())

    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())