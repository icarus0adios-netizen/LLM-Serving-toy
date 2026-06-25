from app.request import GenerateRequest
import time
import asyncio

from dataclasses import dataclass

@dataclass
class RunningRequest:
    req : GenerateRequest       #原始请求
    generated_tokens : int =0   # decoding 阶段生成的 token 数
    prefilled : bool = False    # 该请求是否 完成 prefill 阶段


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
        decode_step_time : float = 0.002,
        prefill_time_per_token : float = 0.001,
    ) -> None:
        self.waiting_queue : asyncio.Queue[GenerateRequest]  = asyncio.Queue()
        self.running_requests : list[RunningRequest] = []
        self.pending  : dict[str,GenerateRequest] = {}

        self.max_running_requests = max_running_requests
        self.decode_step_time = decode_step_time
        self.prefill_time_per_token = prefill_time_per_token

        self.running : bool  = False
        self._worker_task : asyncio.Task | None = None

        self.total_decode_steps : int =0
        self.total_completed_requests : int =0
    
    def stats(self) -> dict:
        return {
            "waiting_queue_size" : self.waiting_queue.qsize(),
            "running_requests" : len(self.running_requests),
            "pending" : len(self.pending),
            "total_decode_steps" : self.total_decode_steps,
            "total_completed_requests" : self.total_completed_requests,
        }

    async def start(self):
        if self.running:
            return
        self.running = True
        self._worker_task = asyncio.create_task(self.run_loop())
    
    async def stop(self) -> None:
        self.running= False
        for request_id , future in list(self.pending.items()):
            if not future.done():
                future.set_exception(RuntimeError("engine stopped"))
        self.pending.clear()
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def submit(self,req : GenerateRequest) -> GenerateResult:
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
        """ 尝试加入新请求 """
        while (
            len(self.running_requests) < self.max_running_requests
            and not self.waiting_queue.empty()
        ):
            req = await self.waiting_queue.get()
            self.running_requests.append(RunningRequest(req))

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
        """
        still_running : list[RunningRequest] = []
        finish_time = time.time()

        for item in self.running_requests:
            req  = item.req

            if req.max_tokens <= item.generated_tokens:
                future = self.pending.pop(req.request_id)
                future.set_result(GenerateResult(
                    request_id = req.request_id,
                    text = f"fake generated text for {req.request_id}",
                    latency_ms = (finish_time - req.arrival_time) * 1000,
                    generated_tokens=item.generated_tokens,
                ))
                self.total_completed_requests +=1
            else:
                still_running.append(item)
            
        self.running_requests = still_running
