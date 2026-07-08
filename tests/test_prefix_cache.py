from app.prefix_cache import PrefixCache


def test_lookup_miss():
    cache = PrefixCache(
        block_size_tokens=4,
        max_cached_blocks=8,
    )

    match = cache.lookup(
        "req-1",
        "a b c d e f g h",
    )

    assert match.matched_blocks == 0
    assert match.matched_tokens == 0
    assert match.block_ids == []

def test_full_prefix_hit():
    cache = PrefixCache(
        block_size_tokens=4,
        max_cached_blocks=8,
    )

    cache.insert(
        "req-1",
        "a b c d e f g h",
    )
    cache.release_request("req-1")

    match = cache.lookup(
        "req-2",
        "a b c d e f g h i j",
    )

    assert match.matched_blocks == 2
    assert match.matched_tokens == 8

def test_partial_prefix_hit():
    cache = PrefixCache(
        block_size_tokens=4,
        max_cached_blocks=8,
    )

    cache.insert(
        "req-1",
        "a b c d e f g h",
    )
    cache.release_request("req-1")

    match = cache.lookup(
        "req-2",
        "a b c d x y z w",
    )

    assert match.matched_blocks == 1
    assert match.matched_tokens == 4

def test_cannot_skip_mismatched_block():
    cache = PrefixCache(
        block_size_tokens=4,
        max_cached_blocks=8,
    )

    cache.insert(
        "req-1",
        "a b c d e f g h i j k l",
    )
    cache.release_request("req-1")

    match = cache.lookup(
        "req-2",
        "a b c d x x x x i j k l",
    )

    assert match.matched_blocks == 1

def test_shared_block_reference_count():
    cache = PrefixCache(
        block_size_tokens=4,
        max_cached_blocks=8,
    )

    block_ids = cache.insert(
        "req-1",
        "a b c d",
    )

    block_id = block_ids[0]
    cache_key = cache.block_id_to_key[block_id]

    cache.lookup(
        "req-2",
        "a b c d extra",
    )

    entry = cache.key_to_entry[cache_key]
    assert entry.ref_count == 2

    cache.release_request("req-1")
    assert entry.ref_count == 1

    cache.release_request("req-2")
    assert entry.ref_count == 0

import pytest


def test_cannot_evict_referenced_block():
    cache = PrefixCache(
        block_size_tokens=4,
        max_cached_blocks=1,
    )

    cache.insert(
        "req-1",
        "a b c d",
    )

    with pytest.raises(RuntimeError):
        cache.insert(
            "req-2",
            "x y z w",
        )

def test_evict_unreferenced_block():
    cache = PrefixCache(
        block_size_tokens=4,
        max_cached_blocks=1,
    )

    first_ids = cache.insert(
        "req-1",
        "a b c d",
    )
    cache.release_request("req-1")

    second_ids = cache.insert(
        "req-2",
        "x y z w",
    )

    assert first_ids != second_ids
    assert cache.total_evictions == 1