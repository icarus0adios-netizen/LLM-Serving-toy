# llm-serving-toy

一个用于学习 LLM 推理服务基础概念的 Toy 项目，包含请求排队、批处理调度、KV Cache 管理、Prefix Cache、连续批处理引擎和抢占调度。

## 项目结构

```
llm-serving-toy/
├── app/
│   ├── __init__.py              # 模块入口
│   ├── request.py               # 请求数据模型
│   ├── request_state.py         # 请求状态枚举（WAITING / RUNNING / PREEMPTED / FINISHED / FAILED）
│   ├── queue.py                 # 请求队列（基于 deque）
│   ├── scheduler.py             # 批处理调度策略（FIFO / Token Bucket）
│   ├── engine.py                # 静态批处理引擎（InferenceEngine）
│   ├── kv_cache.py              # KV Cache Block 分配器（BlockAllocator）
│   ├── prefix_cache.py          # 树形 Prefix Cache（内容寻址、引用计数、FIFO 淘汰）
│   ├── continuous_engine.py     # 连续批处理引擎（ContinuousBatchingEngine，含 Prefix Cache 集成）
│   ├── metrics.py               # 延迟指标收集（P50/P95/P99）
│   └── server.py                # FastAPI 服务（/health, /generate, /metrics）
├── tests/                       # 单元测试
│   ├── test_queue.py
│   ├── test_scheduler.py
│   ├── test_engine.py
│   ├── test_kv_cache.py
│   ├── test_prefix_cache.py
│   ├── test_continuous_engine.py
│   └── test_continuous_engine_prefix_cache.py
├── demo/                        # 演示脚本
│   ├── demo_request.py
│   ├── demo_queue.py
│   ├── demo_scheduler.py
│   ├── demo_engine.py
│   ├── demo_kv_cache.py
│   ├── demo_continuous_engine.py
│   ├── demo_continuous_kv_cache.py
│   ├── demo_dynamic_kv_cache.py
│   ├── demo_kv_preemption.py
│   ├── demo_predix_cache.py
│   ├── demo_continuous_prefix_cache.py
│   └── demo_engines_compare.py
├── benchmark.py                 # HTTP 基准测试客户端
├── benchmark_result.md          # 基准测试结果记录
├── main.py                      # 项目入口占位
├── pyproject.toml               # 项目配置与依赖
└── pytest.ini                   # pytest 配置
```

## 核心概念

### 两代引擎

项目包含两代推理引擎，反映了从静态批处理到连续批处理（Continuous Batching）的演进过程：

| 引擎 | 模块 | 特点 |
|------|------|------|
| **InferenceEngine** | `engine.py` | 静态批处理：固定 batch 大小，等所有请求完成后才返回 |
| **ContinuousBatchingEngine** | `continuous_engine.py` | 连续批处理：每步 decode 后立即返回完成的请求，支持抢占、KV Cache 管理、Prefix Cache |

### ContinuousBatchingEngine 请求生命周期

```
客户端 submit(GenerateRequest)
  → 进入 waiting_queue
  → _admit_new_requests() 准入检查：
      1. 执行 Prefix Cache lookup
      2. Token budget 检查
      3. KV block 可行性检查
      4. KV block 分配
  → 进入 running_requests
  → _prefill_unprefilled_requests()  模拟 prefill（Prefix Cache 命中跳过已缓存 token）
  → _decode_one_step()               模拟 decode（每步可能触发 KV block 扩容或抢占）
  → _complete_finished_requests()    完成的请求返回结果，释放资源
```

### 调度策略

- **FIFOScheduler** — 按先来先服务顺序取固定数量请求组成 batch
- **TokenBucketScheduler** — 根据请求的 tokens 总量（prompt_len + max_tokens）限制每个 batch 的大小，更贴近真实推理服务的资源约束

### KV Cache

`BlockAllocator` 管理固定数量的 KV block：

- `allocate(request_id, num_blocks)` — 分配 block
- `allocate_more(request_id, num_blocks)` — 动态扩容
- `free_by_request(request_id)` — 按请求释放
- 支持动态扩容（decode 阶段按需增长）
- 支持抢占（preemption）：选择占用 block 最多的请求释放其 KV block，重新放入 waiting queue

### Prefix Cache

`PrefixCache` 使用树形结构 + 内容寻址（SHA-256）缓存 token 块：

- `lookup(request_id, prompt)` — 查找最长匹配前缀，命中块增加引用计数
- `insert(request_id, prompt)` — 写入新块，自动淘汰最久未引用的块
- `release_request(request_id)` — 释放请求对缓存块的引用
- 集成到 ContinuousBatchingEngine 中：命中前缀可减少 prefill token 数，降低模拟 prefill 延迟

### 请求状态流转

```
WAITING → (准入) → RUNNING → (生成完成) → FINISHED
                           ↘ (抢占) → PREEMPTED → (重新准入) → RUNNING
                                                ↘ (超过最大抢占次数) → FAILED
```

### FastAPI 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/generate` | POST | 提交推理请求 |
| `/metrics` | GET | 获取服务延迟统计 |

## 快速开始

```bash
# 安装依赖
uv sync

# 启动服务
uv run uvicorn app.server:app --reload

# 健康检查
curl http://127.0.0.1:8000/health

# 推理请求
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello world", "max_tokens": 16}'

# 指标查看
curl http://127.0.0.1:8000/metrics
```

## 运行基准测试

```bash
# 先启动服务，然后在另一个终端运行：
uv run python benchmark.py \
  --concurrency 8 \
  --num-requests 100 \
  --prompt-len 64 \
  --max-tokens 16
```

可选参数见 [benchmark.py](benchmark.py) 中的 `parse_args()`。

## 运行演示

```bash
# 基础概念
uv run python demo/demo_queue.py
uv run python demo/demo_scheduler.py
uv run python demo/demo_request.py

# 静态批处理引擎
uv run python demo/demo_engine.py

# KV Cache
uv run python demo/demo_kv_cache.py
uv run python demo/demo_dynamic_kv_cache.py

# 连续批处理引擎
uv run python demo/demo_continuous_engine.py
uv run python demo/demo_continuous_kv_cache.py
uv run python demo/demo_kv_preemption.py

# Prefix Cache
uv run python demo/demo_predix_cache.py
uv run python demo/demo_continuous_prefix_cache.py

# 引擎对比
uv run python demo/demo_engines_compare.py
```

部分演示脚本需要设置 `PYTHONPATH=.`：

```bash
PYTHONPATH=. uv run python demo/demo_continuous_prefix_cache.py
```

## 运行测试

```bash
# 运行全部测试
uv run pytest

# 运行特定模块测试
uv run pytest tests/test_prefix_cache.py
uv run pytest tests/test_continuous_engine_prefix_cache.py
```

## 依赖

- Python >= 3.9
- FastAPI
- Uvicorn
- Pydantic
- httpx (dev, 基准测试用)
- pytest (dev)

## 设计目标

本项目用于学习 LLM 推理服务中的以下基础概念：

- **请求排队与批处理** — FIFO / Token Bucket 调度策略
- **连续批处理** — 每步 decode 完成立即返回结果，避免等待队列清空
- **KV Cache 管理** — Block 分配、动态扩容、释放
- **Prefix Cache** — 树形内容寻址缓存、引用计数、FIFO 淘汰、减少重复 prefill
- **抢占调度** — 根据 KV block 占用选择 victim，支持恢复时部分复用缓存
- **异步推理引擎** — 基于 asyncio 的事件循环设计
- **服务指标收集** — P50/P95/P99 延迟

当前引擎使用模拟延迟，结果仅用于理解上述机制的交互行为和性能特征。
