# llm-serving-toy

一个用于学习 LLM 推理服务基础概念的 Toy 项目，包含请求排队、批处理调度、模拟推理引擎和性能基准测试。

## 项目结构

```
llm-serving-toy/
├── app/
│   ├── __init__.py       # 模块入口
│   ├── engine.py         # 异步推理引擎（后台 batch 处理循环）
│   ├── queue.py          # 请求队列（基于 deque）
│   ├── scheduler.py      # 批处理调度策略（FIFO / Token Bucket）
│   ├── request.py        # 请求数据模型
│   ├── metrics.py        # 延迟指标收集（P50/P95/P99）
│   └── server.py         # FastAPI 服务（/health, /generate, /metrics）
├── tests/                # 单元测试
│   ├── test_engine.py
│   ├── test_queue.py
│   └── test_scheduler.py
├── demo/                 # 演示脚本
│   ├── demo_engine.py
│   ├── demo_queue.py
│   ├── demo_request.py
│   └── demo_scheduler.py
├── benchmark.py          # HTTP 基准测试客户端
├── benchmark_result.md   # 基准测试结果记录
├── main.py               # 项目入口占位
├── pyproject.toml        # 项目配置与依赖
└── pytest.ini            # pytest 配置
```

## 核心概念

### 请求生命周期

```
客户端 POST /generate
  → GenerateRequest 入队 (RequestQueue)
  → Scheduler 按策略从队列中取一批请求
  → Engine 执行 batch 推理（模拟 prefill + decode）
  → 结果通过 Future 返回给对应客户端
  → Metrics 记录延迟
```

### 调度策略

- **FIFOScheduler** — 按先来先服务顺序取固定数量请求组成 batch
- **TokenBucketScheduler** — 根据请求的 tokens 总量（prompt_len + max_tokens）限制每个 batch 的大小，更贴近真实推理服务的资源约束

### 推理引擎

`InferenceEngine` 使用异步事件循环：

- `submit(req)` — 提交请求，返回一个 awaitable 的 Future
- `run_loop()` — 后台循环，不断从队列取请求并执行 batch
- `_fake_prefill()` / `_fake_decode()` — 模拟推理延迟（基于 prompt 和 output 长度）

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
uv run python demo/demo_queue.py
uv run python demo/demo_scheduler.py
uv run python demo/demo_engine.py
```

## 运行测试

```bash
uv run pytest
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

- 请求排队与批处理
- 调度策略对延迟和吞吐量的影响
- 异步推理引擎的事件循环设计
- 服务指标收集（P50/P95/P99 延迟）

当前引擎使用模拟延迟，结果仅用于理解请求排队和批处理的性能特征。
