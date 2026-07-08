from dataclasses import dataclass, field
from hashlib import sha256
from typing import Sequence

@dataclass 
class PrefixCacheEntry:
    """A cached token block entry.

    Attributes:
        cache_key: SHA-256 hash key derived from parent key and token block.
        block_id: Unique identifier for this cached block.
        token_block: The token sequence stored in this block.
        ref_count: Number of active references holding this block.
        insertion_order: Monotonically increasing counter used for eviction (FIFO).
        request_ids: Set of request IDs currently referencing this block.
    """
    cache_key : str
    block_id : int
    token_block : tuple[str,...]
    ref_count  : int =0
    insertion_order : int =0
    request_ids : set[str] = field(default_factory=set)

@dataclass
class PrefixMatch:
    """Result of a prefix lookup operation.

    Attributes:
        block_ids: Ordered list of matched block IDs from root to deepest match.
        matched_tokens: Total number of tokens matched across all blocks.
        matched_blocks: Number of complete blocks matched.
    """
    block_ids : list[int]
    matched_tokens : int 
    matched_blocks : int

class PrefixCache:
    """A tree-structured prefix cache for token sequences.

    Organises tokenised prompts into a chain of fixed-size blocks, each
    identified by a content-addressed hash key.  Lookups walk the chain
    from the root until a miss occurs, returning the longest matching
    prefix.  Eviction uses a FIFO policy on blocks with zero reference
    count.

    Attributes:
        block_size_tokens: Number of tokens per block.
        max_cached_blocks: Maximum number of blocks allowed in the cache.
    """

    def __init__(
        self,
        block_size_tokens : int,
        max_cached_blocks : int,
     ) -> None:
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be greater than 0")
        if max_cached_blocks <= 0:
            raise ValueError("max_cached_blocks must be greater than 0")
        
        self.block_size_tokens = block_size_tokens
        self.max_cached_blocks = max_cached_blocks

        self.key_to_entry : dict[str,PrefixCacheEntry] = {}
        self.block_id_to_key : dict[int,str] = {}

        self._next_block_id = 0
        self._next_insertion_counter = 0

        self.total_lookups = 0
        self.total_hits = 0
        self.total_matched_tokens = 0
        self.total_evictions = 0
    
    def _compute_block_key(
        self,
        parent_key : str,
        token_block : Sequence[str],
    ) -> str:
        """Compute a content-addressed hash key for a token block.

        The key is derived from the parent key and the tokens joined by
        the unit separator character, producing a unique identifier that
        captures the block's position in the prefix tree.

        Args:
            parent_key: The cache key of the preceding block (or "ROOT").
            token_block: The token sequence forming this block.

        Returns:
            SHA-256 hex digest as the block's cache key.
        """
        payload = parent_key + "|" + "\x1f".join(token_block)
        return sha256(payload.encode("utf-8")).hexdigest()
    
    def _full_token_block(
        self,   
        prompt : str,
    ) -> list[tuple[str,...]]:
        """Split a prompt into complete token blocks.

        The prompt is split on whitespace and grouped into blocks of
        ``block_size_tokens``.  Trailing tokens that do not form a
        complete block are discarded.

        Args:
            prompt: The raw input string to tokenise.

        Returns:
            A list of token tuples, each representing one block.
        """
        
        tokens = prompt.split()

        full_block_count = len(tokens) // self.block_size_tokens

        blocks : list[tuple[str,...]] = []

        for index in range(full_block_count):
            start = index*self.block_size_tokens
            end = start + self.block_size_tokens
            blocks.append(tuple(tokens[start:end]))

        return blocks

    def lookup(
        self,
        request_id : str,
        prompt : str,
    ) -> PrefixMatch:
        """Find the longest cached prefix for a given prompt.

        Walks the prefix tree block by block from the root.  As soon as
        a block is missing from the cache the walk stops and the longest
        prefix found so far is returned.  Hit-rate and matched-token
        statistics are updated internally.

        Args:
            request_id: Unique identifier for the requesting session.
            prompt: The input string to look up.

        Returns:
            A PrefixMatch describing the longest cached prefix found.

        Raises:
            ValueError: If request_id is empty.
        """

        if not request_id:
            raise ValueError("request_id must be non-empty")
        
        self.total_lookups += 1

        token_blocks = self._full_token_block(prompt)

        parent_key= "ROOT"
        matched_block_ids : list[int] = []

        for token_block in token_blocks:
            cache_key = self._compute_block_key(
                parent_key,
                token_block,
            )

            entry = self.key_to_entry.get(cache_key)

            if entry is None:
                break
            
            entry.ref_count += 1
            entry.request_ids.add(request_id)

            matched_block_ids.append(entry.block_id)
            parent_key = cache_key

        matched_blocks = len(matched_block_ids)
        matched_tokens = self.block_size_tokens * matched_blocks

        if matched_blocks > 0:
            self.total_hits += 1
            self.total_matched_tokens += matched_tokens

        return PrefixMatch(
            block_ids=matched_block_ids,
            matched_tokens=matched_tokens,
            matched_blocks=matched_blocks,
        )

    def insert(
        self,
        request_id : str,
        prompt : str,
    ) -> list[int]:
        """Insert a prompt's token blocks into the cache.

        Walks through the prompt's blocks and creates new cache entries
        for any block not yet present.  Already-cached blocks are
        re-used and their reference count is bumped.  If the cache is
        full, the oldest unreferenced block is evicted first.

        Args:
            request_id: Unique identifier for the requesting session.
            prompt: The input string whose blocks should be cached.

        Returns:
            A list of block IDs for every block in the prompt (existing
            and newly created alike), in tree order.

        Raises:
            ValueError: If request_id is empty.
            RuntimeError: If the cache is full and no block can be evicted.
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        
        token_blocks = self._full_token_block(prompt)

        parent_key = "ROOT"
        block_ids : list[int] = []

        for token_block in token_blocks:
            cache_key = self._compute_block_key(
                parent_key,
                token_block,
            )

            existing = self.key_to_entry.get(cache_key)

            if existing is not None:
                if request_id not in existing.request_ids:
                    existing.request_ids.add(request_id)
                    existing.ref_count += 1
                
                block_ids.append(existing.block_id)
                parent_key = cache_key
                continue
            
            self._evict_if_needed()

            block_id = self._next_block_id
            self._next_block_id += 1
            self._next_insertion_counter += 1

            entry = PrefixCacheEntry(
                cache_key = cache_key,
                block_id = block_id,
                token_block = token_block,
                ref_count= 1,
                insertion_order= self._next_insertion_counter,
                request_ids= {request_id},
            )

            self.key_to_entry[cache_key] = entry
            self.block_id_to_key[block_id] = cache_key

            block_ids.append(block_id)
            parent_key = cache_key

        return block_ids
    
    def release_request(
        self,
        request_id : str,
    ) -> None:
        """Release all blocks held by a given request.

        Decrements the reference count on every cached block that the
        request was referencing and removes the request ID from the
        block's tracking set.

        Args:
            request_id: The identifier of the request to release.

        Raises:
            RuntimeError: If a block's reference count drops below zero,
                indicating a mismatched release.
        """
        for entry in self.key_to_entry.values():
            if request_id not in entry.request_ids:
                continue

            entry.request_ids.remove(request_id)
            entry.ref_count -= 1

            if entry.ref_count < 0:
                raise RuntimeError(
                    f"negative ref_count for block {entry.block_id}"
                )

    def _evict_if_needed(self):
        """Evict the oldest unreferenced block if the cache is full.

        Checks whether the cache has reached its capacity.  If so,
        selects the block with the smallest insertion order among those
        with a zero reference count and removes it.

        Raises:
            RuntimeError: If every cached block is still referenced and
                no eviction candidate exists.
        """
        if len(self.key_to_entry) < self.max_cached_blocks:
            return
            
        candidates = [
            entry
            for entry in self.key_to_entry.values()
            if entry.ref_count == 0
        ]

        if not candidates:
            raise RuntimeError(
                "prefix cache is full and no block can be evicted"
            )
        
        vicmin = min(
            candidates,
            key=lambda x: x.insertion_order,
        )

        del self.key_to_entry[vicmin.cache_key]
        del self.block_id_to_key[vicmin.block_id]

        self.total_evictions += 1

    def stats(self) -> dict:
        """Return cache performance statistics.

        Returns:
            A dictionary with the following keys:

            - ``cached_blocks``: Number of blocks currently in the cache.
            - ``max_cached_blocks``: Maximum allowed blocks.
            - ``referenced_blocks``: Blocks with a positive reference count.
            - ``total_lookups``: Total lookup calls made.
            - ``total_hits``: Lookup calls that matched at least one block.
            - ``hit_rate``: Ratio of hits to lookups (0.0 -- 1.0).
            - ``total_matched_tokens``: Cumulative tokens matched across all
              lookups.
            - ``total_evictions``: Number of blocks evicted.
        """
        cache_blocks = len(self.key_to_entry)
        referenced_blocks = sum(
            1
            for entry in self.key_to_entry.values()
            if entry.ref_count > 0
        )

        hit_rate = 0.0
        if self.total_lookups > 0:
            hit_rate = self.total_hits / self.total_lookups

        return {
            "cached_blocks": cache_blocks,
            "max_cached_blocks": self.max_cached_blocks,
            "referenced_blocks": referenced_blocks,
            "total_lookups": self.total_lookups,
            "total_hits": self.total_hits,
            "hit_rate": hit_rate,
            "total_matched_tokens": self.total_matched_tokens,
            "total_evictions": self.total_evictions,
        }