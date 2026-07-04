from app.kv_cache import BlockAllocator, KVCacheBlock, OutOfKVCacheError

def main():
    allocator = BlockAllocator(total_blocks=8, block_size_tokens=16)

    print("initial block allocator:", allocator.stats())

    req1_blocks = allocator.allocate_for_tokens("req-1",30)
    print("req-1 allocated blocks:", req1_blocks)
    print("block allocator stats:", allocator.stats())

    req2_blocks = allocator.allocate_for_tokens("req-2",64)
    print("req-2 allocated blocks:", req2_blocks)
    print("block allocator stats:", allocator.stats())

    allocator.free_by_request("req-1")
    print("after free req-1:", allocator.stats())

    req3_blocks = allocator.allocate_for_tokens("req-3",32)
    print("req-3 allocated blocks:", req3_blocks)
    print("block allocator stats:", allocator.stats())


    try:
        allocator.allocate_for_tokens("req-4", 128)
    except OutOfKVCacheError as e:
        print("allocate req-4 failed:", e)
    
if __name__ == "__main__":
    main()