import asyncio
import time
from dataclasses import dataclass

from app.queue import RequestQueue
from app.request import GenerateRequest
from app.scheduler import FIFOScheduler , TokenBucketScheduler

@dataclass
class GenerateResult:
    request_id : str     #请求 - ID 
    text : str           #推理返回的生成文本
    latency_ms : float   #推理耗时 - 单位：毫秒

class InferenceEngine:
    """异步推理引擎

    通过 request_queue + scheduler + pending 三部分协作：
      - request_queue: 存放等待处理的请求
      - scheduler: 决定每次取哪些请求组成 batch
      - pending: request_id -> Future,用于把结果还给 submit
      - running / _worker_task: 控制后台事件循环的生命周期
    """
    def __init__(self,max_batch_size : int = 4):
        self.request_queue = RequestQueue()
        self.scheduler = FIFOScheduler(self.request_queue,max_batch_size)

        self.pending : dict[str,asyncio.Future[GenerateResult]] = {}

        self.running = False
        self._worker_task : asyncio.Task | None = None

    async def start(self):
        if self.running:
            return
        self.running = True
        self._worker_task = asyncio.create_task(self.run_loop())

    async def stop(self) -> None:
        self.running = False
        for request_id, future in list(self.pending.items()):
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
            raise ValueError("engine is not running")
        
        # 检查 request_id 是否已存在 pending 中 : 确保 req_id 唯一
        if req.request_id in self.pending:
            raise ValueError(f"request_id {req.request_id} is already pending")
        
        loop = asyncio.get_running_loop()
        future : asyncio.Future[GenerateResult] = loop.create_future()
        self.pending[req.request_id] = future
        self.request_queue.push(req)

        return await future
    
    async def run_loop(self):
        while self.running:
            if self.request_queue.empty():
                await asyncio.sleep(0.001)
                continue

            batch = self.scheduler.schedule()
            
            if not batch:
                await asyncio.sleep(0.001)
                continue

            await self._run_batch(batch)

    async def _run_batch(self,batch : list[GenerateRequest]) -> None:
        try:
            await self._fake_prefill(batch)
            await self._fake_decode(batch)
        except Exception as e:
            print(f"Error in _run_batch: {e}")
            raise e

        finish_time = time.time()

        for req in batch:
            future  = self.pending.pop(req.request_id,None)
            if future == None:
                continue
            latency = (finish_time - req.arrival_time) * 1000
            result = GenerateResult(
                request_id = req.request_id,
                text = f"fake generated text for {req.request_id}",
                latency_ms = latency,
            )
            future.set_result(result)

    async def _fake_prefill(self,batch: list[GenerateRequest]) -> None:
        max_prompt_len = max(req.prompt_len for req in batch)
        await asyncio.sleep(max_prompt_len*0.001)


    async def _fake_decode(self,batch: list[GenerateRequest]) -> None:
        max_output_len = max(req.max_tokens for req in batch)
        await asyncio.sleep(max_output_len*0.001)
