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

# Static Batch !!!
class InferenceEngine:
    """异步推理引擎

    通过 request_queue + scheduler + pending 三部分协作：
      - request_queue: 存放等待处理的请求
      - scheduler: 决定每次取哪些请求组成 batch
      - pending: request_id -> Future,用于把结果还给 submit
      - running / _worker_task: 控制后台事件循环的生命周期
    """
    def __init__(self,scheduler_type : str = "fifo",max_batch_size : int = 4,max_batch_tokens : int = 512):
        self.request_queue = RequestQueue()

        # 选择 scheduler 类型 · fifo 或 token_bucket
        # fifo: 先进先出，每个 batch 都是按请求到达顺序处理的
        # token_bucket: 每个 batch 都是按请求 max_tokens 限制处理的
        if scheduler_type == "fifo":
            self.scheduler = FIFOScheduler(self.request_queue,max_batch_size)
        elif scheduler_type == "token_bucket":
            self.scheduler = TokenBucketScheduler(self.request_queue,max_batch_tokens)
        else:
            raise ValueError(f"invalid scheduler_type: {scheduler_type}")

        self.pending : dict[str,asyncio.Future[GenerateResult]] = {}
        self.total_batches =0
        self.batch_sizes : list[int] = []

        self.running = False
        self._worker_task : asyncio.Task | None = None

    async def start(self):
        """
            启动引擎： 启动一个  run_loop  这个常驻的函数进程
        """
        if self.running:
            return
        self.running = True
        self._worker_task = asyncio.create_task(self.run_loop())

    async def stop(self) -> None:
        """
            停止引擎： 取消后台事件循环，等待所有请求完成
            并将所有 pending 中的 Future 设置为异常状态
        """
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
        """
            提交请求： 将请求加入 request_queue 并返回一个 Future
            Future 用于接收推理结果
            如果引擎未运行，抛出 ValueError
        """
        if not self.running:
            raise RuntimeError("engine is not running")
        
        # 检查 request_id 是否已存在 pending 中 : 确保 req_id 唯一
        if req.request_id in self.pending:
            raise ValueError(f"request_id {req.request_id} is already pending")
        
        loop = asyncio.get_running_loop()
        future : asyncio.Future[GenerateResult] = loop.create_future()
        self.pending[req.request_id] = future
        self.request_queue.push(req)

        return await future

    def stat(self) -> dict:
        """
            获取引擎的统计信息
            包含 scheduler 类型,总 batch 数,平均 batch 大小,当前 pending 数,队列大小
        """
        avg_batch_sizes = 0.0
        if self.batch_sizes:
            avg_batch_sizes = sum(self.batch_sizes) / len(self.batch_sizes)
        
        return {
            "scheduler": self.scheduler.type,
            "total_batches": self.total_batches,
            "avg_batch_size": avg_batch_sizes,
            "pending": len(self.pending),
            "queue_size": len(self.request_queue),
        }
    
    async def run_loop(self):
        """
            引擎的核心事件循环： 不断从 request_queue 取请求，调度 batch,运行 batch,返回结果
        """
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
            """
                处理异常： 将所有请求的 Future 设置为异常状态
            """
            for req in batch:
                future = self.pending.pop(req.request_id,None)
                if future is not None and not future.done():
                    future.set_exception(e)
            return

        finish_time = time.time()

        # 统计 batch 信息 + 每个 batch 大小
        self.total_batches +=1
        self.batch_sizes.append(len(batch))

        for req in batch:
            future  = self.pending.pop(req.request_id,None)
            if future is None or future.done():
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
