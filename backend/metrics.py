"""Small dependency-free Prometheus metrics registry for the API process."""
from __future__ import annotations

from collections import defaultdict
from threading import Lock

_lock = Lock()
_requests: dict[tuple[str, str], int] = defaultdict(int)
_latency_sum: dict[str, float] = defaultdict(float)
_latency_count: dict[str, int] = defaultdict(int)
_latency_buckets: dict[tuple[str, float], int] = defaultdict(int)
_cache_events: dict[str, int] = defaultdict(int)
_task_events: dict[str, int] = defaultdict(int)

# Buckets follow Prometheus' conventional HTTP latency boundaries.  Keeping
# them in-process avoids a hard dependency on prometheus_client while still
# allowing operators to calculate histogram_quantile (P50/P95) in PromQL.
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf"))


def record_request(method: str, path: str, status_code: int, latency_seconds: float) -> None:
    route = path.split("?", 1)[0]
    with _lock:
        _requests[(method.upper(), route, str(status_code))] += 1
        _latency_sum[route] += max(0.0, latency_seconds)
        _latency_count[route] += 1
        elapsed = max(0.0, latency_seconds)
        for bucket in _BUCKETS:
            if elapsed <= bucket:
                _latency_buckets[(route, bucket)] += 1


def record_cache_event(result: str) -> None:
    with _lock:
        _cache_events[result] += 1


def record_task_event(result: str) -> None:
    with _lock:
        _task_events[result] += 1


def render_prometheus() -> str:
    lines = [
        "# HELP smart_health_http_requests_total Total HTTP requests.",
        "# TYPE smart_health_http_requests_total counter",
    ]
    with _lock:
        for (method, route, status), value in sorted(_requests.items()):
            lines.append(f'smart_health_http_requests_total{{method="{method}",path="{route}",status="{status}"}} {value}')
        lines.extend([
            "# HELP smart_health_http_request_duration_seconds_sum HTTP request latency sum.",
            "# TYPE smart_health_http_request_duration_seconds_sum counter",
        ])
        for route, value in sorted(_latency_sum.items()):
            lines.append(f'smart_health_http_request_duration_seconds_sum{{path="{route}"}} {value:.6f}')
        lines.extend([
            "# HELP smart_health_http_request_duration_seconds_count HTTP request count used for latency.",
            "# TYPE smart_health_http_request_duration_seconds_count counter",
        ])
        for route, value in sorted(_latency_count.items()):
            lines.append(f'smart_health_http_request_duration_seconds_count{{path="{route}"}} {value}')
        lines.extend([
            "# HELP smart_health_http_request_duration_seconds_bucket HTTP request latency histogram buckets.",
            "# TYPE smart_health_http_request_duration_seconds_bucket histogram",
        ])
        for route, bucket in sorted(_latency_buckets, key=lambda item: (item[0], item[1])):
            le = "+Inf" if bucket == float("inf") else f"{bucket:g}"
            value = _latency_buckets[(route, bucket)]
            lines.append(
                f'smart_health_http_request_duration_seconds_bucket{{path="{route}",le="{le}"}} {value}'
            )
        lines.extend([
            "# HELP smart_health_cache_events_total Cache hit/miss events.",
            "# TYPE smart_health_cache_events_total counter",
        ])
        for result, value in sorted(_cache_events.items()):
            lines.append(f'smart_health_cache_events_total{{result="{result}"}} {value}')
        lines.extend([
            "# HELP smart_health_task_events_total Background task outcomes.",
            "# TYPE smart_health_task_events_total counter",
        ])
        for result, value in sorted(_task_events.items()):
            lines.append(f'smart_health_task_events_total{{result="{result}"}} {value}')
    return "\n".join(lines) + "\n"
