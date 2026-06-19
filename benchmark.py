import argparse
import asyncio
import time
from dataclasses import dataclass

import httpx
from typing import Optional

@dataclass
class RequestResult:
    '''
    一次http请求的返回状态
    '''
    success : bool
    latency_ms : float
    status_code : Optional[int] = None
    error  : Optional[str] = None

def percentile(values : list[float],percent : int):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values)*percent/100)
    if index >= len(sorted_values):
        index = len(sorted_values)-1
    return sorted_values[index]

async def send_one_request(
    client : httpx.AsyncClient,
    url : str,
    prompt : str,
    max_tokens : int
) -> RequestResult:
    '''
    发送一次http请求
    '''
    start = time.perf_counter()

    try:
        response = await client.post(
            url,
            json = {
                "prompt": prompt,
                "max_tokens": max_tokens,
            },
            timeout=30.0,
        )
        end = time.perf_counter()
        latency_ms = (end-start)*1000
        return RequestResult(
            success=True,
            latency_ms=latency_ms,
            status_code=response.status_code,
            error=None if response.status_code == 200 else response.text,
        )
    except Exception as e:
        end = time.perf_counter()
        latency_ms = (end-start)*1000

        return RequestResult(
            success=False,
            latency_ms=latency_ms,  
            status_code=None,
            error=str(e),
        )

async def run_benchmark(
    url : str,
    concurrency : int,
    num_requests : int,
    prompt_len : int,
    max_tokens : int,
) -> list[RequestResult]:
    '''
    运行基准测试
    '''
    prompt = "hello " * prompt_len
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def worker() -> RequestResult:
            async with semaphore:
                return await send_one_request(
                    client=client,
                    url=url,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
        tasks = [asyncio.create_task(worker()) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    return results

def print_summary(results: list[RequestResult], total_time: float, concurrency: int) -> None:
    total = len(results)
    success_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    latencies = [r.latency_ms for r in success_results]
    qps = total / total_time if total_time > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"Total requests: {total}")
    print(f"Concurrency: {concurrency}")
    print(f"Success: {len(success_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"QPS: {qps:.2f}")
    print(f"Avg latency: {avg_latency:.2f} ms")
    print(f"P50 latency: {percentile(latencies, 50):.2f} ms")
    print(f"P95 latency: {percentile(latencies, 95):.2f} ms")
    print(f"P99 latency: {percentile(latencies, 99):.2f} ms")
    if failed_results:
        print("\nFirst 5 errors:")
        for r in failed_results[:5]:
            print(f"- status={r.status_code}, error={r.error}")
            
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/generate")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    return parser.parse_args()

async def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    results = await run_benchmark(
        url=args.url,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        prompt_len=args.prompt_len,
        max_tokens=args.max_tokens,
    )
    end = time.perf_counter()
    total_time = end - start
    print_summary(results, total_time, args.concurrency)

if __name__ == "__main__":
    asyncio.run(main())