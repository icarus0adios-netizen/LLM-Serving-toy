"""
演示 InferenceEngine 的基本使用流程

运行方式：
    python3 demo_engine.py

预期输出：每个请求的结果，包含 request_id、生成文本和延时
"""

import asyncio
from app.engine import InferenceEngine
from app.request import GenerateRequest


async def demo():
    # 1. 创建引擎（最多 4 个请求组成一个 batch）
    engine = InferenceEngine(max_batch_size=4)

    # 2. 启动引擎（后台 run_loop 开始工作）
    await engine.start()
    print("✅ Engine started")

    # 3. 提交一批请求（并发等待所有结果）
    requests = [
        GenerateRequest(request_id="req-1", prompt="What is AI?", max_tokens=50),
        GenerateRequest(request_id="req-2", prompt="Tell me a joke", max_tokens=30),
        GenerateRequest(request_id="req-3", prompt="Python async", max_tokens=80),
        GenerateRequest(request_id="req-4", prompt="Hello world", max_tokens=20),
        GenerateRequest(request_id="req-5", prompt="Another request", max_tokens=60),
    ]

    print(f"📤 Submitting {len(requests)} requests...")
    tasks = [engine.submit(req) for req in requests]
    results = await asyncio.gather(*tasks)

    # 4. 打印结果
    print("\n📋 Results:")
    for r in results:
        print(f"  [{r.request_id}] {r.text}  ({r.latency_ms:.1f} ms)")

    # 5. 停止引擎
    await engine.stop()
    print("\n✅ Engine stopped")


if __name__ == "__main__":
    asyncio.run(demo())
