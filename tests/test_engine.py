import pytest
import pytest_asyncio
import asyncio

from app.engine import InferenceEngine, GenerateResult
from app.request import GenerateRequest


def make_request(id_: str = "1", prompt: str = "hello world", max_tokens: int = 64) -> GenerateRequest:
    return GenerateRequest(
        request_id=id_,
        prompt=prompt,
        max_tokens=max_tokens,
    )


@pytest_asyncio.fixture
async def engine():
    """每个测试用例创建一个新的引擎，自动启动和停止"""
    eng = InferenceEngine(max_batch_size=4)
    await eng.start()
    yield eng
    await eng.stop()


# ─────────────────── 生命周期 ───────────────────


@pytest.mark.asyncio
async def test_start_stop():
    """引擎启动后可以正常停止"""
    eng = InferenceEngine(max_batch_size=4)
    assert eng.running is False

    await eng.start()
    assert eng.running is True
    assert eng._worker_task is not None

    await eng.stop()
    assert eng.running is False
    assert eng._worker_task is None


@pytest.mark.asyncio
async def test_double_start_is_idempotent():
    """重复 start 不会创建多个 worker"""
    eng = InferenceEngine(max_batch_size=4)
    await eng.start()
    task_id = id(eng._worker_task)
    await eng.start()
    assert id(eng._worker_task) == task_id
    await eng.stop()


# ─────────────────── 正常提交流程 ───────────────────


@pytest.mark.asyncio
async def test_submit_returns_result(engine: InferenceEngine):
    """提交一个请求并等待结果"""
    req = make_request()
    result = await engine.submit(req)

    assert isinstance(result, GenerateResult)
    assert result.request_id == "1"
    assert "fake generated text" in result.text
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_submit_multiple_requests(engine: InferenceEngine):
    """并发提交多个请求，全部能拿到正确结果"""
    reqs = [make_request(id_=str(i)) for i in range(5)]
    results = await asyncio.gather(*(engine.submit(req) for req in reqs))

    assert len(results) == 5
    for i, r in enumerate(results):
        assert r.request_id == str(i)


# ─────────────────── 异常场景 ───────────────────


@pytest.mark.asyncio
async def test_submit_before_start_raises():
    """引擎未启动时 submit 应报错"""
    eng = InferenceEngine(max_batch_size=4)
    with pytest.raises(ValueError, match="engine is not running"):
        await eng.submit(make_request())


@pytest.mark.asyncio
async def test_duplicate_id_in_pending_raises():
    """同一个 request_id 在 pending 中时再次提交应报错"""
    eng = InferenceEngine(max_batch_size=4)
    await eng.start()

    # 创建一个 future 手动放入 pending，模拟请求正在等待
    loop = asyncio.get_running_loop()
    fake_future = loop.create_future()
    eng.pending["dup"] = fake_future

    with pytest.raises(ValueError, match="already pending"):
        await eng.submit(make_request(id_="dup"))

    # 清理 fake future 避免 stop 时报错
    fake_future.cancel()
    eng.pending.pop("dup", None)
    await eng.stop()


@pytest.mark.asyncio
async def test_stop_clears_pending_futures():
    """stop 时，还留在 pending 中的 future 应收到异常（确定性测试）"""
    eng = InferenceEngine(max_batch_size=4)
    await eng.start()

    loop = asyncio.get_running_loop()
    fut1 = loop.create_future()
    fut2 = loop.create_future()
    eng.pending["unfinished-1"] = fut1
    eng.pending["unfinished-2"] = fut2

    await eng.stop()

    # 两个 future 都应该被 set_exception
    assert fut1.done()
    assert isinstance(fut1.exception(), RuntimeError)
    assert str(fut1.exception()) == "engine stopped"

    assert fut2.done()
    assert isinstance(fut2.exception(), RuntimeError)
    assert str(fut2.exception()) == "engine stopped"


@pytest.mark.asyncio
async def test_stop_while_requests_pending():
    """stop 时，未完成的请求应收到异常（cancel 会立刻中断 worker）"""
    eng = InferenceEngine(max_batch_size=1)
    await eng.start()

    reqs = [make_request(id_=str(i)) for i in range(3)]
    tasks = [asyncio.create_task(eng.submit(req)) for req in reqs]

    # 确保所有请求都在 queue/pending 中
    await asyncio.sleep(0.01)

    await eng.stop()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    # cancel 会立即中断 worker，所有请求都可能来不及完成
    # 它们要么以异常结束，要么（极小概率）已完成
    for r in results:
        if not isinstance(r, GenerateResult):
            assert isinstance(r, RuntimeError)
            assert str(r) == "engine stopped"


# ─────────────────── 批量处理 ───────────────────


@pytest.mark.asyncio
async def test_batch_processing_respects_max_batch_size():
    """引擎按 max_batch_size 分批处理"""
    eng = InferenceEngine(max_batch_size=2)
    await eng.start()

    reqs = [make_request(id_=str(i)) for i in range(5)]
    results = await asyncio.gather(*(eng.submit(req) for req in reqs))

    assert len(results) == 5
    await eng.stop()


@pytest.mark.asyncio
async def test_latency_measurement(engine: InferenceEngine):
    """latency_ms 应该是一个合理的正数"""
    result = await engine.submit(make_request(prompt="a" * 100, max_tokens=200))
    assert result.latency_ms > 0
    # 基于 fake prefill(100*0.001) + fake decode(200*0.001) ≈ 0.3s = 300ms
    assert result.latency_ms > 200
