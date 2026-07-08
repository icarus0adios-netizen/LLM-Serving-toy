from app.prefix_cache import PrefixCache


def main() -> None:
    cache = PrefixCache(
        block_size_tokens=4,
        max_cached_blocks=8,
    )

    prompt_a = (
        "you are a helpful coding assistant "
        "please answer the following question "
        "how does asyncio work"
    )

    prompt_b = (
        "you are a helpful coding assistant "
        "please answer the following question "
        "what is a future"
    )

    print("initial:")
    print(cache.stats())

    blocks_a = cache.insert("req-A", prompt_a)
    print("\nreq-A inserted:")
    print("blocks:", blocks_a)
    print(cache.stats())

    cache.release_request("req-A")

    match_b = cache.lookup("req-B", prompt_b)
    print("\nreq-B lookup:")
    print(match_b)
    print(cache.stats())

    blocks_b = cache.insert("req-B", prompt_b)
    print("\nreq-B inserted:")
    print("blocks:", blocks_b)
    print(cache.stats())

    cache.release_request("req-B")

    print("\nfinal:")
    print(cache.stats())


if __name__ == "__main__":
    main()