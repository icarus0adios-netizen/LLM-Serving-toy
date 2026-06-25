import asyncio
import time

from app.request import GenerateRequest
from app.continuous_engine import ContinuousBatchingEngine

def make_request(request_id : int,max_tokens : int) -> GenerateRequest:
    return GenerateRequest(
        request_id = f"req-{request_id}",
        max_tokens = max_tokens,
        prompt = "hello model inference"
    )

async def main() ->  None:
    engine = ContinuousBatchingEngine(max_running_requests=4)
    await engine.start()

    start = time.perf_counter()

    requests = [
        make_request(1, 2),
        make_request(2, 8),
        make_request(3, 2),
        make_request(4, 8),
        make_request(5, 2),
        make_request(6, 8),
    ]

    tasks = [
        asyncio.create_task(engine.submit(req)) 
        for req in requests
    ]

    results = await asyncio.gather(*tasks)

    end = time.perf_counter()

    await engine.stop()

    for res in results:
        print(res)

    print("total time:", end - start)
    print("stats:", engine.stats())
    
if __name__ == "__main__":
    asyncio.run(main())