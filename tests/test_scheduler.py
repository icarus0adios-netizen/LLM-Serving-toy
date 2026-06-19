from app.scheduler import TokenBucketScheduler , FIFOScheduler
from app.request import GenerateRequest
from app.queue import RequestQueue

import pytest

def make_request(id : int) -> GenerateRequest:
    return GenerateRequest(
        request_id= f"req-{id}",
        prompt= f"prompt-{id}",
        max_tokens=64,
    )

# 1. FIFO 正常取出 batch
def test_FIFO_batch():
    queue  = RequestQueue()
    queue.push(make_request(1))
    queue.push(make_request(2))
    queue.push(make_request(3))

    scheduler = FIFOScheduler(queue,2)

    batch = scheduler.schedule()
    assert len(batch) == 2
    assert batch[0].request_id == "req-1"
    assert batch[1].request_id == "req-2"

    assert len(queue.peek_all()) == 1

# 2 FIFO 空队列返回空
def test_batch_from_empty_queue():
    queue  = RequestQueue()
    scheduler = FIFOScheduler(queue,2)

    batch = scheduler.schedule()
    assert len(batch) == 0

#3 FIFO 非法 batch_size
def test_FIFO_invalid_batch_size():
    with pytest.raises(ValueError):
        scheduler = FIFOScheduler(RequestQueue(),-1)

#4 TokenBucket 正常取出 batch
def test_token_bucket_batch():
    queue  = RequestQueue()
    queue.push(make_request(1))
    queue.push(make_request(2))
    queue.push(make_request(3))

    scheduler = TokenBucketScheduler(queue,195)

    batch = scheduler.schedule()
    assert len(batch) == 3
    assert batch[0].request_id == "req-1"
    assert batch[1].request_id == "req-2"
    assert batch[2].request_id == "req-3"

    assert len(queue.peek_all()) == 0

# 5 TokenBucket 第一个请求超出预算
def test_token_bucket_empty_queue():
    queue  = RequestQueue()
    queue.push(make_request(1))
    queue.push(make_request(2))
    queue.push(make_request(3))

    scheduler = TokenBucketScheduler(queue,1)

    batch = scheduler.schedule()
    assert len(batch) == 1

# 7 TokenBucket 非法batch size
def test_token_bucket_invalid_batch_size():
    with pytest.raises(ValueError):
        scheduler = TokenBucketScheduler(RequestQueue(),-1)