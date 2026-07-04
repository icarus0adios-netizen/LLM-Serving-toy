from dataclasses import dataclass

from app import request

@dataclass
class KVCacheBlock:
    block_id : int           # id 号
    request_id : str = ""    # 代表当前 block 被哪个请求占用
    is_free : bool = True    # 是否已经被占用

class OutOfKVCacheError(RuntimeError):
    pass

class BlockAllocator:
    def __init__(self,total_blocks : int,block_size_tokens : int) -> None:
        if total_blocks <= 0 :
            raise ValueError("total_blocks must be greater than 0")
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be greater than 0")

        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens

        self.blocks = [
            KVCacheBlock(block_id=i) 
            for i in range(self.total_blocks)
        ]

        self.request_to_blocks : dict[str,list[int]]= {}   # 请求 -> block_id 列表  

    def allocate(self,request_id : str,num_blocks : int) -> list[int]:
        """ 为请求分配 block """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if request_id in self.request_to_blocks:
            raise ValueError("request_id already allocated")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than 0")

        free_blocks = [
            block for block in self.blocks if block.is_free
        ]

        if len(free_blocks) < num_blocks:
            raise OutOfKVCacheError(
                f"not enough free blocks: required={num_blocks}, available={len(free_blocks)}"
            )
            
        selected_blocks = free_blocks[:num_blocks]
        block_ids : list[int] = []

        for block in selected_blocks:
            block.is_free = False
            block.request_id = request_id       
            block_ids.append(block.block_id)
        
        self.request_to_blocks[request_id] = block_ids
            
        return block_ids

    def allocate_for_tokens(self,request_id : str,num_tokens : int) -> list[int]:
        """ 为请求分配 token 数对应的 block """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if request_id in self.request_to_blocks:
            raise ValueError("request_id already allocated")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be greater than 0")
        
        block_size = self.num_blocks_for_tokens(num_tokens)
        return self.allocate(request_id,block_size)

    def free(self,block_ids : list[int]) -> None:
        """ 按照 block_id 列表 释放 block """
        for block_id in block_ids:
            if block_id < 0 or block_id >= self.total_blocks:
                raise ValueError(f"block_id {block_id} is out of range")
            
            block = self.blocks[block_id]
            block.is_free = True
            block.request_id = ""
        
        self._rebuild_request_index()   #部分 block 被释放。 所以需要：重新构建 request_to_blocks 

    def free_by_request(self,request_id : str) -> None:
        """ 按照 request_id 释放 block """
        if request_id not in self.request_to_blocks:
            raise KeyError(f"request_id {request_id} not allocated")
        
        block_ids = self.request_to_blocks.pop(request_id,None)     
        if block_ids is None:
            raise KeyError(f"request_id: {request_id} not found")
        
        self.free(block_ids)

    def num_blocks_for_tokens(self,num_tokens : int) -> int:
        """ 计算 token 数对应的 block 数 """
        if num_tokens <= 0:
            raise ValueError("num_tokens must be greater than 0")
        return (num_tokens + self.block_size_tokens - 1) // self.block_size_tokens   # 注意计算方式 ： 向上取整

    def free_block_count(self) -> int:
        return sum(1 for block in self.blocks if block.is_free)
    
    def used_block_count(self) -> int:
        return self.total_blocks - self.free_block_count()
    
    def stats(self) -> dict:
        return {
            "total_blocks": self.total_blocks,
            "block_size_tokens": self.block_size_tokens,
            "used_blocks": self.used_block_count(),
            "free_blocks": self.free_block_count(),
            "num_requests": len(self.request_to_blocks),
        }

    def _rebuild_request_index(self) -> None:
        """ 重新构建 request_to_blocks 索引 """
        request_to_blocks : dict[str,list[int]] =  {}
        for block in self.blocks:
            if block.is_free or not block.request_id:
                continue

            if block.request_id not in request_to_blocks:
                request_to_blocks[block.request_id] = []
            request_to_blocks[block.request_id].append(block.block_id)
        
        self.request_to_blocks = request_to_blocks