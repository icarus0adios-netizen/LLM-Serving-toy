# 基准测试结果

## 环境

- 项目：llm_serving_toy
- 服务端：FastAPI + InferenceEngine
- 客户端：单进程 asyncio 基准测试
- 命令运行器：uv
- 端点：POST /generate

## 结果

| concurrency | num_requests | prompt_len | max_tokens | QPS | Avg latency | P50 | P95 | P99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 32 | 8 | 21.37 | 45.63 ms | 45.49 ms | 49.02 ms | 49.02 ms |
| 8 | 100 | 64 | 16 | 48.02 | 160.99 ms | 163.33 ms | 165.74 ms | 174.31 ms |
| 32 | 300 | 128 | 32 | 24.27 | 1240.91 ms | 1295.54 ms | 1299.04 ms | 1306.86 ms |

## 观察

1. 低并发下延迟稳定。并发数为 1 时，P50/P95/P99 接近，说明几乎没有排队。
2. 中等并发提升吞吐量。并发数为 8 时，QPS 从 21.37 提升到 48.02，表明批处理有一定效果。
3. 高并发配合较长的 prompt 和 max_tokens 导致明显退化。并发数为 32 时，QPS 降至 24.27，平均延迟升至 1240.91 ms。
4. 第三次运行时 P50/P95/P99 接近，说明大多数请求经历了相似的排队和批处理执行延迟，而非少数极端异常值。

## 局限性

本基准测试是一个面向学习的单进程 asyncio 基准测试。它从客户端测量端到端的 HTTP 延迟。它不测量真实的 TTFT（首 Token 生成时间）、TPOT（每个输出 Token 的时间）、GPU 利用率、KV 缓存使用情况或 Token 级别的流式延迟。

当前引擎使用虚拟的 prefill 和 decode 延迟，因此结果仅应用于理解请求排队、批处理以及延迟/吞吐量之间的权衡。

## 引擎对比实验

### 请求混合比例

- short（短请求）: 70%, prompt_len=10, max_tokens=4
- medium（中等请求）: 20%, prompt_len=30, max_tokens=32
- long（长请求）: 10%, prompt_len=100, max_tokens=128

### 关键发现

1. Static FIFO 实现简单，但存在队首阻塞（head-of-line blocking）问题。
2. TokenBucket 对 max_batch_tokens 参数敏感。
3. 在此模拟环境中，max_batch_tokens=512 的 TokenBucket 表现优于 FIFO。
4. Continuous Batching 显著降低短请求的延迟。
5. Continuous Batching 在当前准入策略下会恶化长请求的延迟。
6. 该模拟器未模拟真实的 GPU/KV 缓存并行度。
