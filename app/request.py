from __future__ import annotations

from dataclasses import dataclass , field
import time

@dataclass
class GenerateRequest:
    request_id : str
    prompt : str
    max_tokens : int
    priority : int =0
    arrival_time : float = field(default_factory=lambda: time.time())

    @property
    def prompt_len(self) -> int:
        return len(self.prompt.split())

    @staticmethod
    def request_cost(req : GenerateRequest) -> int:
        return req.prompt_len + req.max_tokens

def validate_request(req : GenerateRequest) -> None:
    if req.request_id is None:
        raise ValueError("request_id is None")
    if req.prompt is None:
        raise ValueError("prompt is None")
    if req.priority<0:
        raise ValueError("priority is negative")
    if req.max_tokens<=0:
        raise ValueError("max_tokens is negative")
    if req.max_tokens>=2046:
        raise ValueError("max_tokens is too long")
