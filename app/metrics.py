class Metrics:
    def __init__(self):
        self.total_request : int =0
        self.latencies_ms : list[float] = []
        
    def record_latency(self,latency : float) -> None:
        try:
            self.total_request += 1
            self.latencies_ms.append(latency)
        except:
            pass

    def avg_latency(self) -> float:
        if self.latencies_ms is None:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_values = sorted(self.latencies_ms)
        index = int(len(sorted_values) * 0.95)
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]

    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_values = sorted(self.latencies_ms)
        index = int(len(sorted_values)*0.99)
        if index >= len(sorted_values):
            index = len(sorted_values)-1
        return sorted_values[index]

    def snapshot(self) -> dict:
        return {
            "total_request": self.total_request,
            "avg_latency_ms": self.avg_latency(),
            "p95_latency_ms": self.p95_latency_ms(),
            "p99_latency_ms": self.p99_latency_ms(),
        }
