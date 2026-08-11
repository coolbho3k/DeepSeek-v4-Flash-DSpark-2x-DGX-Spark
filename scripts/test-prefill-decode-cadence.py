#!/usr/bin/env python3
"""CPU checks for the non-DP interactive prefill/decode cadence."""

import inspect
from types import SimpleNamespace

import vllm.envs as envs
from vllm.v1.core.sched.scheduler import (
    Scheduler,
    _get_prefill_token_threshold,
)
from vllm.v1.engine.core import EngineCore


assert "VLLM_PREFILL_DECODE_CADENCE" in envs.environment_variables
schedule_source = inspect.getsource(Scheduler.schedule)
assert "interactive_cadence or not self.prefill_capacity_bound" in schedule_source
assert "prefill_token_threshold" in schedule_source
assert schedule_source.count("prefill_token_budget - num_new_tokens") == 2
assert "is_waiting_prefill" in schedule_source


# Cadence-disabled images retain the configured upstream per-request cap.
assert _get_prefill_token_threshold(512, 8192, False, 3) == 512


def waterfill(total_budget: int, demands: list[int]) -> list[int]:
    available = total_budget
    remaining = len(demands)
    allocations: list[int] = []
    for demand in demands:
        limit = _get_prefill_token_threshold(512, available, True, remaining)
        scheduled = min(demand, limit)
        allocations.append(scheduled)
        available -= scheduled
        remaining -= 1
    return allocations


# Pure prefill consumes the full 8192-target-token budget with remainder-aware
# fair shares. Earlier short requests donate their unused allocation to later streams.
assert waterfill(8192, [8192] * 3) == [2731, 2731, 2730]
assert waterfill(8192, [8192] * 5) == [1639, 1639, 1638, 1638, 1638]
assert waterfill(8192, [8192] * 6) == [1366, 1366, 1365, 1365, 1365, 1365]
assert waterfill(8192, [8192] * 7) == [1171, 1171, 1170, 1170, 1170, 1170, 1170]
assert waterfill(8192, [8192] * 8) == [1024] * 8
assert waterfill(8192, [100, 8192, 8192]) == [100, 4046, 4046]

# A mixed iteration shares one global 512-token prefill budget.
assert waterfill(512, [8192] * 3) == [171, 171, 170]


def decisions(cadence: int, count: int) -> list[bool]:
    engine = SimpleNamespace(
        prefill_decode_cadence=cadence,
        _prefill_decode_cadence_step=0,
    )
    return [
        EngineCore._should_throttle_prefills(engine)  # type: ignore[arg-type]
        for _ in range(count)
    ]


# False releases prefill; True asks Scheduler.schedule() to defer it when a
# decoder is active. Upstream behavior remains unchanged at cadence one.
assert decisions(1, 8) == [False] * 8
assert decisions(4, 10) == [
    False,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    False,
    True,
]

print("interactive prefill/decode cadence tests: ok")
