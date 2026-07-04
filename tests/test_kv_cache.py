import pytest

from app.kv_cache import BlockAllocator, OutOfKVCacheError


def test_allocator_init():
    allocator = BlockAllocator(total_blocks=4, block_size_tokens=16)

    assert allocator.total_blocks == 4
    assert allocator.block_size_tokens == 16
    assert allocator.free_block_count() == 4
    assert allocator.used_block_count() == 0


def test_allocate_blocks():
    allocator = BlockAllocator(total_blocks=4, block_size_tokens=16)

    block_ids = allocator.allocate("req-1", 2)

    assert block_ids == [0, 1]
    assert allocator.free_block_count() == 2
    assert allocator.used_block_count() == 2
    assert allocator.request_to_blocks["req-1"] == [0, 1]


def test_allocate_for_tokens_round_up():
    allocator = BlockAllocator(total_blocks=8, block_size_tokens=16)

    assert allocator.num_blocks_for_tokens(1) == 1
    assert allocator.num_blocks_for_tokens(16) == 1
    assert allocator.num_blocks_for_tokens(17) == 2
    assert allocator.num_blocks_for_tokens(32) == 2
    assert allocator.num_blocks_for_tokens(33) == 3


def test_allocate_for_tokens():
    allocator = BlockAllocator(total_blocks=8, block_size_tokens=16)

    block_ids = allocator.allocate_for_tokens("req-1", 33)

    assert block_ids == [0, 1, 2]
    assert allocator.used_block_count() == 3
    assert allocator.free_block_count() == 5


def test_free_blocks():
    allocator = BlockAllocator(total_blocks=4, block_size_tokens=16)

    block_ids = allocator.allocate("req-1", 2)
    allocator.free(block_ids)

    assert allocator.free_block_count() == 4
    assert allocator.used_block_count() == 0
    assert "req-1" not in allocator.request_to_blocks


def test_free_by_request():
    allocator = BlockAllocator(total_blocks=4, block_size_tokens=16)

    allocator.allocate("req-1", 2)
    allocator.free_by_request("req-1")

    assert allocator.free_block_count() == 4
    assert allocator.used_block_count() == 0
    assert "req-1" not in allocator.request_to_blocks


def test_allocate_not_enough_blocks():
    allocator = BlockAllocator(total_blocks=2, block_size_tokens=16)

    allocator.allocate("req-1", 2)

    with pytest.raises(OutOfKVCacheError):
        allocator.allocate("req-2", 1)


def test_duplicate_request_allocate_raises_error():
    allocator = BlockAllocator(total_blocks=4, block_size_tokens=16)

    allocator.allocate("req-1", 1)

    with pytest.raises(ValueError):
        allocator.allocate("req-1", 1)


def test_free_invalid_block_id():
    allocator = BlockAllocator(total_blocks=4, block_size_tokens=16)

    with pytest.raises(ValueError):
        allocator.free([999])


def test_free_unknown_request():
    allocator = BlockAllocator(total_blocks=4, block_size_tokens=16)

    with pytest.raises(KeyError):
        allocator.free_by_request("unknown")


def test_stats():
    allocator = BlockAllocator(total_blocks=4, block_size_tokens=16)

    allocator.allocate("req-1", 2)

    assert allocator.stats() == {
        "total_blocks": 4,
        "block_size_tokens": 16,
        "used_blocks": 2,
        "free_blocks": 2,
        "num_requests": 1,
    }