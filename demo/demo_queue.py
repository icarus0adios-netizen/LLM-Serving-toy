from app.queue import RequestQueue
from app.request import GenerateRequest
from app.request import validate_request

def main():
    queue = RequestQueue()
    req1 = GenerateRequest(request_id="req_001", prompt="hello inference serving", max_tokens=1024)
    req2 = GenerateRequest(request_id="req_002", prompt="hello inference serving", max_tokens=1024)
    req3 = GenerateRequest(request_id="req_003", prompt="hello inference serving", max_tokens=1024)
    queue.push(req1)
    queue.push(req2)
    queue.push(req3)

    print("queue length", len(queue))
    print("queue empty", queue.empty())

    batch  = queue.pop_batch(2)
    print("pop batch", batch)
    print("queue length", len(queue))
    print("queue empty", queue.empty())

    last_batch = queue.pop_batch(10)
    print("pop last batch", last_batch)
    print("queue length", len(queue))
    print("queue empty", queue.empty())

if __name__ == "__main__":
    main()