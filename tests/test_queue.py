from codecs import ascii_encode

import pytest

from app.request import GenerateRequest
from app.queue import RequestQueue

def make_request(id : int) -> GenerateRequest:
    return GenerateRequest(
        request_id= f"req-{id}",
        prompt= f"prompt-{id}",
        max_tokens=64,
    )


# 测试新队列为空
def test_new_queue_is_empty():
    queue  = RequestQueue()
    assert len(queue) == 0
    assert queue.empty() is True

# 测试 Push 一个请求
def test_push_function():
    queue = RequestQueue()
    req  = make_request(1)
    queue.push(req)

    assert len(queue) == 1
    assert queue.empty() is False


# 空队列 pop_batch 返回空列表
def test_pop_batch_from_empty_queue():
    queue = RequestQueue()

    batch  = queue.pop_batch(4)

    assert batch == []
    assert len(queue) ==0


# 队列3个req ， pop_batch 请求2个
def test_pop_batch_from_queue():
    queue  = RequestQueue()
    queue.push(make_request(1))
    queue.push(make_request(2))
    queue.push(make_request(3))

    batch = queue.pop_batch(2)
    assert len(batch) ==2
    assert batch[0].request_id == "req-1"
    assert batch[1].request_id == "req-2"
    assert len(queue) ==1

# 请求数量超过 queue 中的数量
def test_pop_batch_from_queue_exceed():
    queue  = RequestQueue()
    queue.push(make_request(1))
    queue.push(make_request(2))
    queue.push(make_request(3))

    batch = queue.pop_batch(4)
    assert len(batch) == 3
    assert batch[0].request_id == "req-1"
    assert batch[1].request_id == "req-2"
    assert batch[2].request_id == "req-3"
    assert len(queue) ==0

# 非法 batch size
def test_invalid_batch_size():
    queue  = RequestQueue()

    with pytest.raises(ValueError):
        queue.pop_batch(0)
    with pytest.raises(ValueError):
        queue.pop_batch(-1)

# 测试 peek_all
def test_peek_all():
    queue  = RequestQueue()
    queue.push(make_request(1))
    queue.push(make_request(2))
    queue.push(make_request(3))

    items = queue.peek_all()

    assert len(items) == 3
    assert items[0].request_id == "req-1"
    assert items[1].request_id == "req-2"
    assert items[2].request_id == "req-3"