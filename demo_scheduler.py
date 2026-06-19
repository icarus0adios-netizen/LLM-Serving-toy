from app.queue import RequestQueue
from app.request import GenerateRequest
from app.scheduler import TokenBucketScheduler
from app.scheduler import FIFOScheduler

def make_request(id : str , prompt : str , max_tokens : int) -> GenerateRequest :
    return GenerateRequest(
        request_id = id,
        prompt = prompt,
        max_tokens = max_tokens,
    ) 

def demo_fifo() ->None:
    queue = RequestQueue()

    queue.push(make_request("req-1", "hello model", 64))
    queue.push(make_request("req-2", "explain kv cache", 64))
    queue.push(make_request("req-3", "what is paged attention", 64))

    scheduler = FIFOScheduler(queue,2)
    batch  = scheduler.schedule()
    print("fifo batch-------------")
    print(batch)
    print("remaining queue-------------")
    print(queue.peek_all())


def demo_token_bucket() ->None:
    queue = RequestQueue()

    queue.push(make_request("req-1", "hello model", 64))
    queue.push(make_request("req-2", "explain kv cache", 64))
    queue.push(make_request("req-3", "what is paged attention", 64))

    scheduler = TokenBucketScheduler(queue,100)
    batch  = scheduler.schedule()
    print("token bucket batch-------------")
    print(batch)
    print("remaining queue-------------") 
    print(queue.peek_all())

def main() -> None:
    demo_fifo()
    demo_token_bucket()

if __name__ == "__main__":
    main()