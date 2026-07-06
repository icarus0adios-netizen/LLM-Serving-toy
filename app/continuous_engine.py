from app.request import GenerateRequest
from app.kv_cache import BlockAllocator,OutOfKVCacheError
import time
import asyncio

from dataclasses import dataclass , field

@dataclass
class RunningRequest:
    req : GenerateRequest       #原始请求
    generated_tokens : int =0   # decoding 阶段生成的 token 数
    prefilled : bool = False    # 该请求是否 完成 prefill 阶段

    kv_block_ids : list[int]  = field(default_factory=list) # 记录请求申请的 kv cache 的 block id


@dataclass
class GenerateResult:
    request_id : str     #请求 - ID 
    text : str           #推理返回的生成文本
    latency_ms : float   #推理耗时 - 单位：毫秒
    generated_tokens : int =0   # 最终输出的 token 数

class ContinuousBatchingEngine:
    """Continuous Batching Engine (连续批量引擎)."""
    def __init__(
        self,
        max_running_requests : int = 4,
        max_running_tokens : int = 512,
        total_kv_blocks : int = 64,
        block_size_tokens : int = 16,
        decode_step_time : float = 0.002,
        prefill_time_per_token : float = 0.001,
    ) -> None:
        """Initialize the continuous batching engine.

        Args:
            max_running_requests: Maximum number of requests that can run concurrently.
            max_running_tokens: Maximum total tokens allowed across all running requests.
            total_kv_blocks: Total number of KV cache blocks available.
            block_size_tokens: Number of tokens per KV cache block.
            decode_step_time: Simulated time per decode step (seconds).
            prefill_time_per_token: Simulated time per prefill token (seconds).
        """
        self.waiting_queue : asyncio.Queue[GenerateRequest]  = asyncio.Queue()
        self.running_requests : list[RunningRequest] = []
        self.pending  : dict[str,asyncio.Future[GenerateResult]] = {}

        self.max_running_requests = max_running_requests
        self.max_running_tokens = max_running_tokens
        self.decode_step_time = decode_step_time
        self.prefill_time_per_token = prefill_time_per_token

        self.kv_allocator = BlockAllocator(total_blocks=total_kv_blocks,block_size_tokens=block_size_tokens)

        self.running : bool  = False
        self._worker_task : asyncio.Task | None = None

        self.total_decode_steps : int =0        # 总的 decoding 步骤数
        self.total_completed_requests : int =0  # 已完成的请求数量
        self.total_kv_waits : int =0            # 因为暂时没有足够 KV block 而等待的次数
        self.total_rejected_requests : int =0   # 请求的总容量超过整个 allocator，永远不可能运行，被直接拒绝的数量。
        self.total_kv_growth_events : int =0
        self.total_kv_blocks_grown : int  =0
        self.total_decode_kv_failures : int =0  # 因为没有足够 KV block 而失败的 decoding 次数
    
    def stats(self) -> dict:
        """Return a snapshot of current engine statistics."""
        return {
            "waiting_queue_size" : self.waiting_queue.qsize(),
            "running_requests" : len(self.running_requests),
            "current_running_tokens" : self.current_running_tokens(),
            "max_running_requests" : self.max_running_requests,
            "max_running_tokens" : self.max_running_tokens,
            "pending" : len(self.pending),
            "total_decode_steps" : self.total_decode_steps,
            "total_completed_requests" : self.total_completed_requests,
            "total_kv_waits" : self.total_kv_waits,
            "total_rejected_requests" : self.total_rejected_requests,
            "total_kv_growth_events" : self.total_kv_growth_events,
            "total_kv_blocks_grown" : self.total_kv_blocks_grown,
            "total_decode_kv_failures" : self.total_decode_kv_failures,
            "kv_cache" : self.kv_allocator.stats(),
        }

    async def start(self):
        """Start the engine's background worker loop."""
        if self.running:
            return
        self.running = True
        self._worker_task = asyncio.create_task(self.run_loop())
    
    async def stop(self) -> None:
        """Stop the engine.

        Set all pending request futures to an exception state and release
        all KV blocks held by those requests.
        """
        self.running= False

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        self._fail_all_pending(RuntimeError("Engine Stopped"))

        while not self.waiting_queue.empty():
            try:
                self.waiting_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def submit(self, req: GenerateRequest) -> GenerateResult:
        """Submit a request to the engine and return a Future for the result.

        The request is placed into the waiting queue. A Future is returned
        that will resolve to the generation result once processed.

        Raises:
            RuntimeError: If the engine is not running.
            ValueError: If the request ID already exists in pending.
        """
        if not self.running:
            raise RuntimeError("engine is not running")
        if req.request_id in self.pending:
            raise ValueError(f"request_id {req.request_id} is already pending")
        
        loop = asyncio.get_running_loop()
        future : asyncio.Future[GenerateResult]= loop.create_future()
        self.pending[req.request_id] = future

        await self.waiting_queue.put(req)
        return await future

    async def run_loop(self) -> None:
        """Main engine loop.

        Each iteration:
        1. Admits new requests from the waiting queue.
        2. Runs prefill for unprefilled requests.
        3. Decodes one token for all running requests.
        4. Returns completed requests immediately.
        """
        try:
            while self.running:
                await self._admit_new_requests()    # 尝试加入新请求

                if not self.running_requests:
                    await asyncio.sleep(0.001)
                    continue

                await self._prefill_unprefilled_requests()    # 对所有未完成 prefill 的请求做 prefill
                await self._decode_one_step()    # 所有 running 请求 decode 1 token (一个 step)
                self._complete_finished_requests()    # 完成的请求立即返回
        except asyncio.CancelledError:
            raise

        except Exception as e:
            self.running = False
            self._fail_all_pending(e)
            raise


    async def _admit_new_requests(self) -> None:
        """Admit new requests from the waiting queue into running.

        Admission follows three layers of checks:
        1. Token budget: reject if current + request cost exceeds max_running_tokens,
           unless the queue is empty (prevents starvation).
        2. Feasibility: reject if the request requires more KV blocks than the entire cache.
        3. Availability: if no KV blocks are available, put the request back and wait.
        """
        while (
            len(self.running_requests) < self.max_running_requests
            and not self.waiting_queue.empty()
        ):
            req = await self.waiting_queue.get()
            req_cost = GenerateRequest.request_cost(req)
            current_tokens = self.current_running_tokens()
            
            # 第一层：逻辑 token budget 限制
            if(
                self.running_requests
                and current_tokens + req_cost > self.max_running_tokens
            ):
                """
                    因为如果队列里第一个请求自己就超过 token budget (即在 running_requests 中还没有请求,我们的cost 计算就超出了)
                    我们仍然要允许它进入,否则会 starvation
                """
                # NOTE: putting the request back may change strict FIFO order.
                await self.waiting_queue.put(req)
                break

            # 第二层： 永远无法运行！
            maximum_blocks = self._maximum_kv_blocks(req)

            if maximum_blocks > self.kv_allocator.total_blocks:
                self._reject_request(
                    req,
                    RuntimeError(
                        "request requires more KV blocks than the "
                        f"entire cache: required={maximum_blocks}, "
                        f"total={self.kv_allocator.total_blocks}"
                    )
                )
                self.total_rejected_requests += 1
                continue
            
            # 第三层:当前暂时没有足够 block

            required_blocks = self._prompt_kv_blocks(req)   # 只为 prompt（也就是prefill）阶段分配 block ,后续的decoding 再动态扩容

            try:
                block_ids = self.kv_allocator.allocate(
                    request_id=req.request_id,
                    num_blocks=required_blocks,
                )
            except OutOfKVCacheError:
                self.total_kv_waits +=1 
                await self.waiting_queue.put(req)
                break

            # 前面三层通过后 构建 Running request 请求，加入队列
            # NOTE: 必须 copy block_ids，否则 kv_block_ids 与 allocator 内部
            #       request_to_blocks 共享同一列表，导致 allocate_more 重复追加。
            self.running_requests.append(
                RunningRequest(
                    req=req,
                    kv_block_ids=list(block_ids),
                    )
                )

    def current_running_tokens(self) -> int:
        """Return the total token cost of all currently running requests."""
        total =0 
        for item in self.running_requests:
            total += GenerateRequest.request_cost(item.req)
        return total

    async def _prefill_unprefilled_requests(self) -> None:
        """Prefill all requests that have not yet completed their prefill phase.

        Simulates the prefill time based on the longest prompt length.
        """
        unprefilled = [item for item in self.running_requests if not item.prefilled]

        if not unprefilled:
            return

        # 模拟正在进行 prefill 花费的时间
        max_prompt_len = max(item.req.prompt_len for item in unprefilled )
        await asyncio.sleep(max_prompt_len * self.prefill_time_per_token)
        
        # 标记所有请求为已 prefill
        for item in unprefilled:
            item.prefilled = True

    async def _decode_one_step(self) -> None:
        """Decode one token for all running requests (one step).

        Simulates the decode time with a fixed sleep duration.
        """
        
        # 模拟正在进行 decode 花费的时间
        await asyncio.sleep(self.decode_step_time)

        suvivors : list[RunningRequest] = []

        for item in self.running_requests:
            try:
                self._grow_blocks_if_needed(item)
            except OutOfKVCacheError as error:
                self.total_decode_kv_failures +=1
                self._fail_running_request(item, error)
                continue

            suvivors.append(item)
            item.generated_tokens += 1
    
        self.running_requests = suvivors
        self.total_decode_steps +=1

    def _required_tokens_next_decode(self,item : RunningRequest) -> int:
        return (
            item.generated_tokens 
            + item.req.prompt_len
            + 1
        )
    
    def _additional_blocks_needed(self,item : RunningRequest) -> int:
        tokens = self._required_tokens_next_decode(item)
        needed_blocks = self.kv_allocator.num_blocks_for_tokens(tokens)
        current_blocks = len(item.kv_block_ids)

        return needed_blocks - current_blocks

    def _grow_blocks_if_needed(self,item : RunningRequest) -> None:
        additional_blocks = self._additional_blocks_needed(item)

        if additional_blocks > 0:
            block_ids = self.kv_allocator.allocate_more(
                request_id=item.req.request_id,
                num_blocks=additional_blocks,
            )
            self.total_kv_growth_events += 1   # 记录扩容事件
            self.total_kv_blocks_grown += len(block_ids)   # 记录扩容的 block 数量
 
            item.kv_block_ids.extend(block_ids)   # 注意 ！！！ 不要忘记更新 RunningRequest 中维护的 kv_block_ids

    def _fail_running_request(self,item : RunningRequest, error : Exception) -> None:
        """ 失败请求不再放进 survivors
            其 KV blocks 会立即释放；
            对应 submit() 会抛异常，而不是永久等待。
        """
        request_id = item.req.request_id

        self._release_kv_block(request_id)

        future = self.pending.pop(request_id,None)

        if future is not None and not future.done():
            future.set_exception(
                RuntimeError(
                "KV cache expansion failed during decode: "
                f"request_id={request_id}, error={error}"   
                )
            )

    def _complete_finished_requests(self) -> None:
        """Return finished requests immediately.

        Core of continuous batching: completed requests are returned right
        away instead of waiting for all requests to finish.

        Responsibilities:
        1. Free KV blocks for finished requests.
        2. Set pending futures to the completed state.
        """
        still_running : list[RunningRequest] = []
        finish_time = time.time()

        for item in self.running_requests:
            req  = item.req

            # 判断 请求是否已经完成 ！
            if item.generated_tokens < req.max_tokens:
                still_running.append(item)
                continue

            # 请求完成：
            # 1. 释放 KV block
            # 2. pending 结果置入已完成状态

            if req.request_id in self.kv_allocator.request_to_blocks:
                self.kv_allocator.free_by_request(req.request_id)

            future = self.pending.pop(req.request_id,None)

            if future is not None and not future.done():
                future.set_result(GenerateResult(
                    request_id = req.request_id,
                    text = f"fake generated text for {req.request_id}",
                    latency_ms = (finish_time - req.arrival_time) * 1000,
                    generated_tokens=item.generated_tokens,
                ))

            self.total_completed_requests +=1
            
        self.running_requests = still_running

    def _required_kv_tokens(self, req: GenerateRequest) -> int:
        """Return the maximum number of kv tokens needed: prompt_len + max_tokens."""
        return GenerateRequest.request_cost(req)   # 直接 复用 GenerateRequest.request_cost 计算 token 数

    def _required_kv_blocks(self, req: GenerateRequest) -> int:
        """Return the number of KV blocks required for the given request."""
        tokens = self._required_kv_tokens(req)
        return self.kv_allocator.num_blocks_for_tokens(tokens)
    
    def _prompt_kv_blocks(self,req : GenerateRequest) -> int:
        return self.kv_allocator.num_blocks_for_tokens(req.prompt_len)
    
    def _maximum_kv_blocks(self,req : GenerateRequest) -> int:
        return self.kv_allocator.num_blocks_for_tokens(req.max_tokens+req.prompt_len)

    def _reject_request(
        self,
        req: GenerateRequest,
        error: Exception,
    ):
        """Reject a request by setting its pending future to the given error."""
        future = self.pending.pop(req.request_id,None)

        if future is not None and not future.done():
            future.set_exception(error)

    def _fail_all_pending(self, error: Exception) -> None:
        """Fail all pending requests with the given error.

        All pending futures are set to the exception state, and all
        associated KV blocks are released.
        """
        for request_id, future in list(self.pending.items()):
            self._release_kv_block(request_id)

            if not future.done():
                future.set_exception(error)

        self.pending.clear()
        self.running_requests.clear()

    def _release_kv_block(self, req_id: str):
        """Release KV blocks associated with the given request ID."""
        if req_id in self.kv_allocator.request_to_blocks:
            self.kv_allocator.free_by_request(req_id)

    
