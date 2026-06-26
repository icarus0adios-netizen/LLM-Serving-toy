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
        max_running_tokens : int = 512,
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

        self.running : bool  = False
        self._worker_task : asyncio.Task | None = None

        self.total_decode_steps : int =0
        self.total_completed_requests : int =0
    
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
            如果 running_requests 中的 token 数不超过 max_running_tokens,则加入新请求
        """
        while (
            len(self.running_requests) < self.max_running_requests
            and not self.waiting_queue.empty()
        ):
            req = await self.waiting_queue.get()
            req_cost = GenerateRequest.request_cost(req)
            current_tokens = self.current_running_tokens()
            
            
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

            self.running_requests.append(RunningRequest(req))

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
        """
        still_running : list[RunningRequest] = []
        finish_time = time.time()

        for item in self.running_requests:
            req  = item.req

            if req.max_tokens <= item.generated_tokens:
                future = self.pending.pop(req.request_id,None)

                if future is not None and not future.done():
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

    
