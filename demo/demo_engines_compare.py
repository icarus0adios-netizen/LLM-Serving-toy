"""
对比不同引擎的性能:
    Static FIFO
    Static TokenBucket
    Continuous Batching

对比指标：
    1. 总耗时 elapsed
    2. 每个请求 latency 
    3. engine.stats()
    4. 短请求是否能更早完成
    5. 长请求是否拖慢 batch
"""

from typing import Any

from app.request import GenerateRequest
from app.scheduler import FIFOScheduler , TokenBucketScheduler
from app.engine import  InferenceEngine 
from app.continuous_engine import ContinuousBatchingEngine
import asyncio
import time


def make_mixed_requests(num_requests : int) -> tuple[list[GenerateRequest],dict[str,str]]:
    requests = []
    requests_types = {}

    for i in range(num_requests):
        if i % 10 < 7:
            prompt_words = 10
            max_tokens = 4
            req_type = "short"
        elif i % 10 < 9:
            prompt_words = 30
            max_tokens = 32
            req_type = "medium"
        else:
            prompt_words = 100
            max_tokens = 128
            req_type = "long"
        request_id = f"req-{i}"
        requests.append(GenerateRequest(
            request_id=request_id,
            prompt="x "*prompt_words,
            max_tokens=max_tokens,
        ))
        requests_types[request_id] = req_type
    
    return requests , requests_types

def percentile(values : list[float],ratio : float) -> float:
    """
        计算 values 中的 ratio 百分位数
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index  = int(len(sorted_values)*ratio)

    if index >= len(sorted_values):
        index = len(sorted_values)-1
    return sorted_values[index]

def summarize_results(
    name: str,
    results: list[Any],
    request_types: dict[str, str],
    elapsed: float,
) -> None:
    all_latencies = [result.latency_ms for result in results]

    print(f"\n=== {name} ===")
    print(f"elapsed: {elapsed:.3f}s")
    print(f"total_requests: {len(results)}")
    print(f"avg_latency: {sum(all_latencies) / len(all_latencies):.2f}ms")
    print(f"p50_latency: {percentile(all_latencies, 0.50):.2f}ms")
    print(f"p95_latency: {percentile(all_latencies, 0.95):.2f}ms")
    print(f"p99_latency: {percentile(all_latencies, 0.99):.2f}ms")

    for req_type in ["short", "medium", "long"]:
        latencies = [
            result.latency_ms
            for result in results
            if request_types[result.request_id] == req_type
        ]

        if not latencies:
            continue

        print(
            f"{req_type}: "
            f"count={len(latencies)}, "
            f"avg={sum(latencies) / len(latencies):.2f}ms, "
            f"p50={percentile(latencies, 0.50):.2f}ms, "
            f"p95={percentile(latencies, 0.95):.2f}ms"
        )

async def run_case(name : str,engine : Any,num_requests : int = 100):
    requests , requests_types = make_mixed_requests(num_requests)

    await engine.start()

    start = time.perf_counter()

    tasks = [
        engine.submit(req) 
        for req in requests
    ]
    
    results = await asyncio.gather(*tasks)

    time_diff = time.perf_counter() - start

    await engine.stop()

    summarize_results(name, results, requests_types, time_diff)

    if hasattr(engine, "stats"):
        print("stats:", engine.stats())

async def main() -> None:
    num_requests = 100

    cases = [
        (
            "Static FIFO bs=4",
            InferenceEngine(
                scheduler_type="fifo",
                max_batch_size=4,
            ),
        ),
        (
            "Static TokenBucket tokens=128",
            InferenceEngine(
                scheduler_type="token_bucket",
                max_batch_tokens=128,
            ),
        ),
        (
            "Static TokenBucket tokens=256",
            InferenceEngine(
                scheduler_type="token_bucket",
                max_batch_tokens=256,
            ),
        ),
        (
            "Static TokenBucket tokens=512",
            InferenceEngine(
                scheduler_type="token_bucket",
                max_batch_tokens=512,
            ),
        ),
        (
            "Continuous req=4 tokens=128",
            ContinuousBatchingEngine(
                max_running_requests=4,
                max_running_tokens=128,
            ),
        ),
        (
            "Continuous req=4 tokens=512",
            ContinuousBatchingEngine(
                max_running_requests=4,
                max_running_tokens=512,
            ),
        ),
        (
            "Continuous req=8 tokens=512",
            ContinuousBatchingEngine(
                max_running_requests=8,
                max_running_tokens=512,
            ),
        ),
    ]

    for name, engine in cases:
        await run_case(name, engine, num_requests=num_requests)

if __name__ == "__main__":
    asyncio.run(main())
