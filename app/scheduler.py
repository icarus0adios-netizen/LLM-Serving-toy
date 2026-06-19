from app.queue import RequestQueue
from app.request import GenerateRequest

class FIFOScheduler:
    def __init__(self,queue: RequestQueue,max_batch_size:int) -> None:
        if max_batch_size < 0 :
            raise ValueError("max_batch_size must be greater than 0")
        self.queue = queue
        self.max_batch_size = max_batch_size

    def schedule(self) -> list[GenerateRequest]:
        return self.queue.pop_batch(self.max_batch_size)


class TokenBucketScheduler:
    def __init__(self,queue : RequestQueue , max_batch_tokens : int) -> None:
        if max_batch_tokens < 0 :
            raise ValueError("max_batch_tokens must be greater than 0")
        self.queue = queue
        self.max_batch_tokens  = max_batch_tokens
    
    def schedule(self) -> list[GenerateRequest]:
        items = self.queue.peek_all()
        if not items:
            return []
        
        usec_tokens = 0
        selected_count = 0

        for req in items:
            cost  = GenerateRequest.request_cost(req)
            if selected_count ==0 and cost >= self.max_batch_tokens:
                selected_count += 1
                break
            if usec_tokens + cost <= self.max_batch_tokens:
                selected_count += 1
                usec_tokens += cost
            else:
                break

        return self.queue.pop_batch(selected_count)                
