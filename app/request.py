from __future__ import annotations

from dataclasses import dataclass , field
import time

@dataclass
class GenerateRequest:
    request_id : str
    prompt : str
    max_tokens : int
    priority : int =0
    arrival_time : float = field(default_factory=lambda: time.time()) # 这里不能写 "arrival_time: float = time.time()"
                                                                      # 类定义时 执行一次，所有实例共享同一个时间值
                                                                      # 每次实例化时 执行，每个实例得到自己的时间值
    @property      # 使成员函数 成为属性，直接调用即可，无需加括号
    def prompt_len(self) -> int:
        return len(self.prompt.split())

    @staticmethod      # 静态方法 成为类方法，直接调用即可，无需实例化
    def request_cost(req : GenerateRequest) -> int:
        return req.prompt_len + req.max_tokens

def validate_request(req : GenerateRequest) -> None:
    if not req.request_id:
        raise ValueError("request_id is empty")
    if not req.prompt:
        raise ValueError("prompt is empty")
    if req.priority<0:
        raise ValueError("priority is negative")
    if req.max_tokens<=0:
        raise ValueError("max_tokens is negative")
    if req.max_tokens>=2046:
        raise ValueError("max_tokens is too long")
