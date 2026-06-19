from collections import deque
from app.request import GenerateRequest

class RequestQueue:
    def __init__(self) ->None:
        self._queue : deque[GenerateRequest] = deque()

    def __len__(self) -> int:
        return len(self._queue)

    def push(self,req : GenerateRequest) -> None:
        self._queue.append(req)

    def pop_batch(self,batch_size : int) -> list[GenerateRequest]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        batch_size = min(batch_size, len(self._queue))
        return [self._queue.popleft() for _ in range(batch_size)]

    def empty(self) -> bool:
        return len(self._queue) == 0

    def peek_all(self) -> list[GenerateRequest]:
        return list(self._queue)
