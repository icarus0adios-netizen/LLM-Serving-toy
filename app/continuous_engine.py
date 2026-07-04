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
    """ 
        Continuous Batching Engine(连续批量引擎) 
    """
    def __init__(
        self,
        max_running_requests : int = 4,
        max_running_tokens : int = 512,
        total_kv_blocks : int = 64,
        block_size_tokens : int = 16,
        decode_step_time : float = 0.002,
        prefill_time_per_token : float = 0.001,
    ) -> None:
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
    
    def stats(self) -> dict:
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
            "kv_cache" : self.kv_allocator.stats(),
        }

    async def start(self):
        if self.running:
            return
        self.running = True
        self._worker_task = asyncio.create_task(self.run_loop())
    
    async def stop(self) -> None:
        """
            停止引擎： 将所有请求的 Future 设置为异常状态 
            同时 释放所有请求的 KV block
        """
        self.running= False
        for request_id , future in list(self.pending.items()):
            if not future.done():
                future.set_exception(RuntimeError("engine stopped"))
            # 释放请求的 KV block
            self._release_kv_block(request_id)
            
        self.pending.clear()
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def submit(self,req : GenerateRequest) -> GenerateResult:
        """
            提交请求： 将请求加入 waiting_queue 并返回一个 Future
            Future 用于接收推理结果
            如果引擎未运行，抛出 RuntimeError
            如果请求 ID 已存在于 pending 中，抛出 ValueError
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
        """ 在每一个 loop 中执行以下操作：
            1. 尝试加入新请求
            2. 对新请求做 prefill
            3. 所有 running 请求 decode 1 token
            4. 完成的请求立即返回
            5. 下一轮继续加入新请求
        """
        while self.running:
            await self._admit_new_requests()    # 尝试加入新请求

            if not self.running_requests:
                await asyncio.sleep(0.001)
                continue

            await self._prefill_unprefilled_requests()    # 对所有未完成 prefill 的请求做 prefill
            await self._decode_one_step()    # 所有 running 请求 decode 1 token (一个 step)
            self._complete_finished_requests()    # 完成的请求立即返回


    async def _admit_new_requests(self) -> None:
        """ 尝试加入新请求 :
            如果 running_requests 中的 token 数超过 max_running_tokens,则拒绝新请求
            如果 队列中第一个请求就超过了 token budget , 仍然需要允许它进入,否则会 starvation !!!
        """
        while (
            len(self.running_requests) < self.max_running_requests
            and not self.waiting_queue.empty()
        ):
            req = await self.waiting_queue.get()
            req_cost = GenerateRequest.request_cost(req)
            current_tokens = self.current_running_tokens()
            required_blocks = self._required_kv_blocks(req)

            
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
            if required_blocks > self.kv_allocator.total_blocks:
                self._reject_request(
                    req,
                    RuntimeError(
                        "request requires more KV blocks than the "
                        f"entire cache: required={required_blocks}, "
                        f"total={self.kv_allocator.total_blocks}"
                    )
                )
                self.total_rejected_requests += 1
                continue
            
            # 第三层:当前暂时没有足够 block
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
            self.running_requests.append(
                RunningRequest(
                    req=req,
                    kv_block_ids=block_ids,
                    )
                )

    def current_running_tokens(self) -> int:
        """ 当前 running 请求的 token 数 """
        total =0 
        for item in self.running_requests:
            total += GenerateRequest.request_cost(item.req)
        return total

    async def _prefill_unprefilled_requests(self) -> None:
        """ 对所有未完成 prefill 的请求做 prefill """
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
        """ 所有 running 请求 decode 1 token (一个 step) """
        
        # 模拟正在进行 decode 花费的时间
        await asyncio.sleep(self.decode_step_time)

        for item in self.running_requests:
            item.generated_tokens +=1
    
        self.total_decode_steps +=1

    def _complete_finished_requests(self) -> None:
        """ 完成的请求立即返回
            Continuous Batching 的核心： 立即返回已完成的请求，而不是等待所有请求都完成

            任务:
                1. 释放 KV block
                2. pending 结果置入已完成状态
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

    def _required_kv_tokens(self,req : GenerateRequest) -> int:
        """ 计算理论上最大token数 : prompt_len + max_tokens  """
        return GenerateRequest.request_cost(req)   # 直接 复用 GenerateRequest.request_cost 计算 token 数

    def _required_kv_blocks(self,req : GenerateRequest) -> int :
        """ 根据请求计算 需要的 KV block 数 """
        tokens = self._required_kv_tokens(req)
        return self.kv_allocator.num_blocks_for_tokens(tokens)

    def _reject_request(
        self,
        req : GenerateRequest,
        error : Exception,
    ):
        """ 拒绝请求 """
        future = self.pending.pop(req.request_id,None)

        if future is not None and not future.done():
            future.set_exception(error)

    def _fail_all_pending(self, error: Exception) -> None:
        """
            处理异常： 将所有请求的 Future 设置为异常状态 
            同时 释放所有请求的 KV block
        """
        for request_id, future in list(self.pending.items()):
            self._release_request_kv_cache(request_id)

            if not future.done():
                future.set_exception(error)

        self.pending.clear()
        self.running_requests.clear()

    def _release_kv_block(self,req_id : str):
        """根据 req-id 释放 KV block """
        if req_id in self.kv_allocator.request_to_blocks:
            self.kv_allocator.free_by_request(req_id)

    
