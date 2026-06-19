from app.request import GenerateRequest, validate_request

def main() -> None:
    req = GenerateRequest(request_id="req_001", prompt="hello inference serving", max_tokens=1024)
    validate_request(req)

    print("req id", req.request_id)
    print("prompt", req.prompt)
    print("max_tokens", req.max_tokens)
    print("priority", req.priority)
    print("arrival_time", req.arrival_time)
    print("prompt_len", req.prompt_len)

if __name__ == "__main__":
    main()