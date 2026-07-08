import pytest
from app.continuous_engine import (
    ContinuousBatchingEngine,
)
from app.request import GenerateRequest


@pytest.mark.asyncio
async def test_second_request_hits_prefix_cache():
    """第二个请求应该命中 prefix cache。"""
    engine = ContinuousBatchingEngine(
        max_running_requests=1,
        max_running_tokens=512,
        total_kv_blocks=32,
        block_size_tokens=4,
        decode_step_time=0.001,
        prefill_time_per_token=0.001,
        enable_prefix_cache=True,
        prefix_cache_max_blocks=32,
    )
    await engine.start()

    req1 = GenerateRequest(
        request_id="req-1",
        prompt="a b c d e f g h question one",
        max_tokens=2,
    )
    await engine.submit(req1)

    req2 = GenerateRequest(
        request_id="req-2",
        prompt="a b c d e f g h question two",
        max_tokens=2,
    )
    await engine.submit(req2)

    assert engine.total_prefix_cache_misses == 1
    assert engine.total_prefix_cache_hits == 1
    assert engine.total_prefix_matched_tokens == 8

    await engine.stop()


@pytest.mark.asyncio
async def test_full_prefix_hit_skips_prefill_tokens():
    """完整命中应该跳过所有 prefill token。"""
    engine = ContinuousBatchingEngine(
        max_running_requests=1,
        max_running_tokens=512,
        total_kv_blocks=32,
        block_size_tokens=4,
        decode_step_time=0.001,
        prefill_time_per_token=0.001,
        enable_prefix_cache=True,
        prefix_cache_max_blocks=32,
    )
    await engine.start()

    prompt = "a b c d e f g h"
    await engine.submit(
        GenerateRequest(
            request_id="req-1",
            prompt=prompt,
            max_tokens=1,
        )
    )

    actual_after_first = engine.total_actual_prefill_tokens

    await engine.submit(
        GenerateRequest(
            request_id="req-2",
            prompt=prompt,
            max_tokens=1,
        )
    )

    second_actual_prefill = (
        engine.total_actual_prefill_tokens - actual_after_first
    )
    assert actual_after_first == 8
    assert second_actual_prefill == 0
    assert engine.total_saved_prefill_tokens >= 8

    await engine.stop()


@pytest.mark.asyncio
async def test_finished_request_leaves_no_prefix_reference():
    """完成后所有 prefix cache 引用归零。"""
    engine = ContinuousBatchingEngine(
        max_running_requests=1,
        max_running_tokens=512,
        total_kv_blocks=32,
        block_size_tokens=4,
        decode_step_time=0.001,
        prefill_time_per_token=0.001,
        enable_prefix_cache=True,
        prefix_cache_max_blocks=32,
    )
    await engine.start()

    await engine.submit(
        GenerateRequest(
            request_id="req-1",
            prompt="a b c d e f g h",
            max_tokens=1,
        )
    )

    entries = list(engine.prefix_cache.key_to_entry.values())
    assert len(entries) == 2
    assert all(entry.ref_count == 0 for entry in entries)
    assert all(entry.request_ids == set() for entry in entries)

    await engine.stop()


@pytest.mark.asyncio
async def test_prefix_cache_can_be_disabled():
    """禁用 prefix cache 后不记录任何命中/未命中统计。"""
    engine = ContinuousBatchingEngine(
        max_running_requests=1,
        max_running_tokens=512,
        total_kv_blocks=32,
        block_size_tokens=4,
        decode_step_time=0.001,
        prefill_time_per_token=0.001,
        enable_prefix_cache=False,
    )
    await engine.start()

    prompt = "a b c d e f g h"
    await engine.submit(
        GenerateRequest(
            request_id="req-1",
            prompt=prompt,
            max_tokens=1,
        )
    )

    await engine.submit(
        GenerateRequest(
            request_id="req-2",
            prompt=prompt,
            max_tokens=1,
        )
    )

    assert engine.total_prefix_cache_hits == 0
    assert engine.total_prefix_cache_misses == 0
    assert engine.total_actual_prefill_tokens == 16
    assert engine.total_saved_prefill_tokens == 0

    await engine.stop()
